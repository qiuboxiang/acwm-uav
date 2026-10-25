"""Exploration baselines v2: collision-checked random / next-best-view / frontier (3D grid + BFS) for a free-flying UAV."""
import os, time, json, argparse, collections, numpy as np, imageio
from uav_sim import UAVSim
from scenes import SCENES, paths, Geometry

CAM_H = 1.5; RADIUS = 0.25; GRID = 0.25
# action set: IndoorUAV's dual-scale forward/turn + vertical (our addition; step sizes not given in the paper)
A_FWD_L, A_FWD_S, A_TURN_L, A_TURN_S, A_UP = 0.9, 0.15, 15.0, 3.0, 0.25

class Agent:
    def __init__(self, sim, geom, seed):
        self.s, self.G = sim, geom; self.rng = np.random.default_rng(seed)
        self.sim_t = 0.0; self.coll = 0; self.dist = 0.0; self.traj = []
        res = 256; f = (res / 2) / np.tan(np.deg2rad(90) / 2)
        u, v = np.meshgrid(np.arange(0, res, 8), np.arange(0, res, 8))
        self.rays = np.stack([(u - res / 2) / f, -(v - res / 2) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        self.seen = set()
        lo = geom.bounds[0] - 1; self.origin = lo; self.shape = np.ceil((geom.bounds[1] + 1 - lo) / GRID).astype(int)
        self.occ = np.zeros(self.shape, np.uint8)  # 0 unknown, 1 free, 2 occupied
    # ---- low level -------------------------------------------------
    def cam(self, p=None): p = self.s.pose()[0] if p is None else p; return p + np.array([0, CAM_H, 0])
    def act(self, kind, mag):
        """Execute one action; blocked moves are rejected (counted as collisions)."""
        p, y = self.s.pose(); self.traj.append(self.cam(p).copy())
        if kind == "yaw": self.s.set_pose(p, (y + mag) % 360); return True
        r = np.deg2rad(y); fwd = np.array([-np.sin(r), 0, -np.cos(r)])
        q = p + (fwd * mag if kind == "fwd" else np.array([0, mag, 0]))
        if self.G.segment_free(self.cam(p), self.cam(q), RADIUS):
            self.s.set_pose(q, y); self.dist += abs(mag); return True
        self.coll += 1; return False
    def map_clear(self, d):
        p, y = self.s.pose(); r = np.deg2rad(y); fwd = np.array([-np.sin(r), 0, -np.cos(r)]); c = self.cam(p)
        for s in np.arange(0.1, d + 1e-6, 0.1):
            i = tuple(np.clip(np.floor((c + fwd * s - self.origin) / GRID).astype(int), 0, self.shape - 1))
            if self.occ[i] != 1: return False
        return True
    def free_dist(self, kind="fwd", maxd=0.9):
        p, y = self.s.pose(); r = np.deg2rad(y); fwd = np.array([-np.sin(r), 0, -np.cos(r)])
        for d in np.arange(0.15, maxd + 1e-6, 0.15):
            if self.G.collides(self.cam(p) + fwd * d, RADIUS): return d - 0.15
        return maxd
    # ---- mapping ----------------------------------------------------
    def observe(self):
        t = time.time(); rgb, d = self.s.obs(); self.sim_t += time.time() - t
        p, y = self.s.pose(); c, s_ = np.cos(np.deg2rad(y)), np.sin(np.deg2rad(y))
        R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]); cp = self.cam(p)
        dd = d[::8, ::8].reshape(-1); ok = (dd > 0.05)
        dd = np.where(ok, dd, 6.0); hit = ok & (dd < 6.0)
        dirs = self.rays @ R.T
        # coverage voxels (full-res subsample stride 4)
        d4 = d[::4, ::4].reshape(-1); res = 256; f = (res / 2) / np.tan(np.pi / 4)
        u, v = np.meshgrid(np.arange(0, res, 4), np.arange(0, res, 4))
        r4 = np.stack([(u - res / 2) / f, -(v - res / 2) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        m = (d4 > 0.05) & (d4 < 8.0); w = (r4[m] * d4[m, None]) @ R.T + cp
        self.seen |= set(map(tuple, np.floor(w / 0.25).astype(np.int32)))
        # occupancy: free along rays, occupied at hit
        for dist in np.arange(0.0, 6.0, GRID / 2):
            pts = cp + dirs * np.minimum(dist, dd - 0.15)[:, None]; msk = dist < dd - 0.1
            idx = np.floor((pts[msk] - self.origin) / GRID).astype(int); idx = np.clip(idx, 0, self.shape - 1)
            self.occ[tuple(idx.T)] = np.where(self.occ[tuple(idx.T)] == 2, 2, 1)
        hp = cp + dirs[hit] * dd[hit, None]; idx = np.clip(np.floor((hp - self.origin) / GRID).astype(int), 0, self.shape - 1)
        self.occ[tuple(idx.T)] = 2
        return rgb, d
    def coverage(self): return len(self.seen & self.G.gt_vox) / max(1, len(self.G.gt_vox))

# ---- policies --------------------------------------------------------
def pol_random(A, d):
    fd = A.free_dist()
    if fd >= 0.9 and A.rng.random() < 0.8: A.act("fwd", A_FWD_L)
    elif fd >= 0.15 and A.rng.random() < 0.5: A.act("fwd", A_FWD_S)
    else: A.act("yaw", float(A.rng.choice([-A_TURN_L, A_TURN_L])))

def pol_nbv(A, d, state={}):
    # greedy gain among yaw candidates, scored by newly-seen voxels (uses ground-truth-free depth only)
    p, y = A.s.pose(); best, bg = 0, -1
    for dy in (-90, -45, -15, 0, 15, 45, 90):
        A.s.set_pose(p, (y + dy) % 360); t = time.time(); _, dd = A.s.obs(); A.sim_t += time.time() - t
        r = np.deg2rad((y + dy) % 360); R = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
        res = 256; f = 128; u, v = np.meshgrid(np.arange(0, res, 8), np.arange(0, res, 8))
        rr = np.stack([(u - 128) / f, -(v - 128) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        d8 = dd[::8, ::8].reshape(-1); m = (d8 > .05) & (d8 < 8); w = (rr[m] * d8[m, None]) @ R.T + A.cam(p)
        g = len(set(map(tuple, np.floor(w / 0.25).astype(np.int32))) - A.seen)
        A.s.set_pose(p, y); fd = A.free_dist()
        g += 30 * (fd >= 0.9)
        if dy != 0: g -= 5
        if g > bg: bg, best = g, dy
    if best: A.act("yaw", best if abs(best) <= 15 else float(np.sign(best)) * A_TURN_L)  # one action per step
    else:
        fd = A.free_dist()
        if fd >= 0.9: A.act("fwd", A_FWD_L)
        elif fd >= 0.15: A.act("fwd", A_FWD_S)
        else: A.act("yaw", A_TURN_L)

class Frontier:
    NB6 = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    def __init__(self): self.path = []; self.target_yaw = None; self.bad = set(); self.stuck = 0; self.replan_in = 0; self.thr = 30
    def plan(self, A):
        occ = A.occ; cp = A.cam(); start = tuple(np.clip(np.floor((cp - A.origin) / GRID).astype(int), 0, np.array(A.shape) - 1))
        from scipy.ndimage import binary_dilation
        blocked = binary_dilation(occ == 2, iterations=1)  # inflate by 1 cell (0.25m) -> 0.25 + radius margin
        for b in self.bad: blocked[b] = True
        free = (occ == 1) & ~blocked; free[start] = True
        unk = (occ == 0)
        from scipy.ndimage import uniform_filter
        ucnt = uniform_filter(unk.astype(np.float32), size=(5, 3, 5), mode='constant', cval=0.0) * 75  # unknown cells in a 5x3x5 block
        q = collections.deque([start]); prev = {start: None}
        while q:
            c = q.popleft()
            if c != start and np.linalg.norm((np.array(c) - np.array(start)) * GRID) >= 1.0 and c not in self.bad and ucnt[c] >= self.thr:
                # frontier: free cell with a horizontal unknown neighbour backed by enough unknown volume
                for n in self.NB6[[0, 1, 4, 5]]:
                    nn = (c[0] + n[0], c[1] + n[1], c[2] + n[2])
                    if all(0 <= nn[i] < A.shape[i] for i in range(3)) and unk[nn]:
                        path = []; k = c
                        while k is not None: path.append(k); k = prev[k]
                        self.path = path[::-1][1:]; self.goal = c
                        dirv = np.array(n, float); self.target_yaw = np.rad2deg(np.arctan2(-dirv[0], -dirv[2])) % 360 if (dirv[0] or dirv[2]) else None
                        return True
            for n in self.NB6:
                nn = (c[0] + n[0], c[1] + n[1], c[2] + n[2])
                if nn in prev or not all(0 <= nn[i] < A.shape[i] for i in range(3)): continue
                if free[nn]: prev[nn] = c; q.append(nn)
        return False

def pol_frontier(A, d, F=Frontier()):
    pass  # replaced by closure in run()

def frontier_step(A, F):
    for _ in range(8):
        st = _frontier_step(A, F)
        if st != "wp": return st
    return st

def _frontier_step(A, F):
    cp = A.cam(); cur = tuple(np.floor((cp - A.origin) / GRID).astype(int))
    if F.replan_in <= 0 or not F.path:
        ok = F.plan(A)
        if not ok: F.thr = 8; F.bad = set(); ok = F.plan(A)
        if not ok:
            F.nf = getattr(F, "nf", 0) + 1
            if F.nf <= 24: A.act("yaw", A_TURN_L)  # scan in place first
            else: pol_random(A, None)
            return "no_frontier"
        F.nf = 0; F.thr = 30
        F.replan_in = 25
    F.replan_in -= 1
    if not F.path:
        F.bad.add(F.goal); return "arrived"
    wp = A.origin + (np.array(F.path[0]) + 0.5) * GRID
    v = wp - cp
    if np.linalg.norm(v) < 0.15 or tuple(np.floor((cp - A.origin) / GRID).astype(int)) == tuple(F.path[0]): F.path.pop(0); return "wp"
    p, y = A.s.pose(); want = np.rad2deg(np.arctan2(-v[0], -v[2])) % 360 if np.hypot(v[0], v[2]) > 0.1 else y
    dy = (want - y + 180) % 360 - 180
    if abs(dy) > 8: A.act("yaw", float(np.clip(dy, -A_TURN_L, A_TURN_L))); return "turn"
    if abs(v[1]) > 0.15 and np.hypot(v[0], v[2]) < 0.3: ok = A.act("up", float(np.sign(v[1]) * A_UP))
    else:
        hd = min(np.hypot(v[0], v[2]), 0.9)
        step = next((s for s in (A_FWD_L, A_FWD_S) if s <= hd + 0.1 and A.map_clear(s)), None)
        if step is None and A.map_clear(A_FWD_S): step = A_FWD_S
        ok = step is not None and A.act("fwd", step)
        if ok and abs(v[1]) > 0.15: A.act("up", float(np.sign(v[1]) * A_UP))
    if not ok:
        F.stuck += 1
        if F.stuck > 1: F.bad.add(F.path[0]); F.path = []; F.stuck = 0; F.replan_in = 0
    else: F.stuck = 0
    return "move"

# ---- rollout + visualisation -----------------------------------------
def top_down(A, G, size=300):
    lo, hi = G.bounds[0][[0, 2]], G.bounds[1][[0, 2]]; sc = size / (hi - lo).max()
    img = np.full((size, size, 3), 255, np.uint8)
    pts = G.pts[::6]; px = ((pts[:, [0, 2]] - lo) * sc).astype(int).clip(0, size - 1); img[px[:, 1], px[:, 0]] = (200, 200, 200)
    return img, lo, sc
def draw_map(base, lo, sc, traj, size=300):
    import cv2
    img = base.copy(); T = np.array(traj)
    for i in range(1, len(T)):
        a = ((T[i - 1][[0, 2]] - lo) * sc).astype(int); b = ((T[i][[0, 2]] - lo) * sc).astype(int)
        col = tuple(int(c) for c in (255 * i / len(T), 60, 255 - 255 * i / len(T)))
        cv2.line(img, tuple(a), tuple(b), col, 1)
    if len(T): cv2.circle(img, tuple(((T[-1][[0, 2]] - lo) * sc).astype(int)), 4, (0, 160, 0), -1)
    return img

def rollout(policy, S, G, steps, seed, video=None, name=""):
    import cv2
    A = Agent(S, G, seed); S.rng = A.rng
    for _ in range(500):
        p = np.array(S.sim.pathfinder.get_random_navigable_point(), float)
        if np.isfinite(p).all() and not G.collides(p + [0, CAM_H, 0], RADIUS + 0.1): break
    S.set_pose(p, float(A.rng.integers(0, 360)))
    F = Frontier(); curve = []; frames = []
    base, lo, sc = top_down(A, G) if video else (None, None, None)
    t0 = time.time(); plan_t = 0.0
    for t in range(steps):
        rgb, d = A.observe()
        if t % 10 == 0: curve.append(A.coverage())
        if video and t % 2 == 0:
            m = draw_map(base, lo, sc, A.traj); r = cv2.resize(rgb, (300, 300)).copy()
            cv2.putText(r, f"{policy} t={t} cov={A.coverage():.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 0), 1)
            frames.append(np.concatenate([r, m], 1))
        tp = time.time()
        if policy == "random": pol_random(A, d)
        elif policy == "nbv": pol_nbv(A, d)
        elif policy == "frontier":
            st = frontier_step(A, F)
            if st == "arrived":
                if F.target_yaw is not None:
                    for _ in range(6):
                        y = S.pose()[1]; dy = (F.target_yaw - y + 180) % 360 - 180
                        if abs(dy) < 8: break
                        A.act("yaw", float(np.clip(dy, -A_TURN_L, A_TURN_L)))
                F.path = []; F.replan_in = 0
        plan_t += time.time() - tp
    wall = time.time() - t0
    if video: imageio.mimsave(video, frames, fps=15)
    return dict(scene=name, policy=policy, seed=seed, steps=steps, coverage=A.coverage(), curve=curve, collisions=A.coll,
                dist=A.dist, wall_s=wall, ms_per_step=1000 * wall / steps, sim_s=A.sim_t, traj=[t.tolist() for t in A.traj[::5]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--policies", default="random,nbv,frontier"); ap.add_argument("--out", default="results")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    g, n = paths(a.scene); S = UAVSim(g); S.sim.pathfinder.load_nav_mesh(n)
    t = time.time(); G = Geometry(g, S.sim.pathfinder.get_bounds()); print("geom", f"{time.time()-t:.1f}s", flush=True)
    out = []
    for pol in a.policies.split(","):
        for sd in range(a.seeds):
            r = rollout(pol, S, G, a.steps, sd, f"{a.out}/{a.scene}__{pol}__{sd}.mp4" if sd == 0 else None, a.scene); out.append(r)
            print(f"{a.scene} {pol} s{sd} cov={r['coverage']:.3f} coll={r['collisions']} dist={r['dist']:.0f}m wall={r['wall_s']:.1f}s ({r['ms_per_step']:.0f} ms/step)", flush=True)
    json.dump(out, open(f"{a.out}/{a.scene}.json", "w"))
