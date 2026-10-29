"""Exploration v4: sensor-only agent with
  (1) frontier2: information-gain + distance scoring, junction 360-degree pan (OPAL-style)
  (2) pose correction: collision-aware dead reckoning (depth-consistency bump detection)
  (3) diversity metrics for world-model data collection.
Agent inputs are unchanged from explore3 (RGB-D or RGB-only + own commands)."""
import os, os, time, json, argparse, collections, numpy as np, imageio, cv2
from scipy.ndimage import binary_dilation, uniform_filter
from uav_sim import UAVSim
from scenes import paths, Geometry
import explore2 as E2
from explore2 import frontier_step, Frontier, top_down, draw_map, CAM_H, RADIUS, GRID, A_FWD_L, A_FWD_S, A_TURN_L, A_UP
from explore3 import Env, SensorAgent, DepthNet, pol_random

NB6 = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
HNB = [(1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
VNB = [(0, 1, 0), (0, -1, 0)]

class Frontier2(Frontier):
    """Scores every reachable frontier bucket by (unknown-volume gain) - lam * (path length); not just the nearest."""
    def __init__(self, lam=0.04, mu=0.0, vfront=False): super().__init__(); self.LAM = lam; self.MU = mu; self.vfront = vfront; self.n_near = 0; self.pan_left = 0; self.pan_pos = None; self.pans = 0
    @staticmethod
    def _turn(A, start, c):  # heading change (0..1) needed to face the candidate; in-place yaw costs flight time
        v = (np.array(c) - np.array(start)).astype(float)
        if np.hypot(v[0], v[2]) < 1e-6: return 0.0
        want = np.rad2deg(np.arctan2(-v[0], -v[2])) % 360
        return abs((want - A.yaw + 180) % 360 - 180) / 180.0
    def plan(self, A):
        occ = A.occ; cp = A.cam(); shp = A.shape
        start = tuple(np.clip(np.floor((cp - A.origin) / GRID).astype(int), 0, np.array(shp) - 1))
        blocked = binary_dilation(occ == 2, iterations=int(os.environ.get('PLAN_DIL', '0'))) if int(os.environ.get('PLAN_DIL', '0')) > 0 else (occ == 2)
        for b in self.bad: blocked[b] = True
        free = (occ == 1) & ~blocked; free[start] = True
        nr = tuple(slice(max(0, s - 2), s + 3) for s in start); free[nr] |= ((occ == 1) & (occ != 2))[nr]
        unk = (occ == 0)
        opt = getattr(self, 'optimistic', False); trav = (free | (unk & ~blocked)) if opt else free
        ucnt = uniform_filter(unk.astype(np.float32), size=(5, 3, 5), mode="constant", cval=0.0) * 75
        dist = np.full(shp, -1, np.int32); par = {}; dist[start] = 0; q = collections.deque([start]); order = []
        while q:
            c = q.popleft(); order.append(c); dc = dist[c]
            for n in NB6:
                nn = (c[0] + n[0], c[1] + n[1], c[2] + n[2])
                if 0 <= nn[0] < shp[0] and 0 <= nn[1] < shp[1] and 0 <= nn[2] < shp[2] and dist[nn] < 0 and trav[nn]:
                    dist[nn] = dc + 1; par[nn] = c; q.append(nn)
        best = {}; cnt = [len(order), 0, 0, 0, 0]
        for c in order:
            if opt and not free[c]: continue
            d_m = dist[c] * GRID; cnt[1] += 1
            if d_m < getattr(self, 'min_d', 1.0) or c in self.bad: continue
            cnt[2] += 1
            if ucnt[c] < self.thr: continue
            cnt[3] += 1
            for n in (HNB + VNB if self.vfront else HNB):
                nn = (c[0] + n[0], c[1] + n[1], c[2] + n[2])
                if 0 <= nn[0] < shp[0] and 0 <= nn[1] < shp[1] and 0 <= nn[2] < shp[2] and unk[nn]:
                    key = (c[0] // 4, c[1] // 2, c[2] // 4); sc = ucnt[c] / 75.0 - self.LAM * d_m - self.MU * self._turn(A, start, c)
                    if key not in best or sc > best[key][0]: best[key] = (sc, c, n, d_m)
                    break
        self.dbg = cnt
        if not best: return False
        self.n_near = sum(1 for v in best.values() if v[3] <= 4.0)
        sc, c, n, d_m = max(best.values(), key=lambda v: v[0])
        path = []; k = c
        while k != start: path.append(k); k = par[k]
        self.path = path[::-1]; self.goal = c
        dv = np.array(n, float); self.target_yaw = (np.rad2deg(np.arctan2(-dv[0], -dv[2])) % 360) if (dv[0] or dv[2]) else None
        return True

class Frontier3(Frontier):
    """Classic 2-D frontier exploration on a height band of the agent's 3-D map (+-0.25 m around the camera): frontier = free cell next to unknown, clustered;
    score = cluster size (info) - lam * path length. Far less sensitive to speckle in 3-D unknown space than Frontier2. Altitude changes come from the altitude behaviour."""
    def __init__(self, lam=0.04, min_cluster=3): super().__init__(); self.LAM = lam; self.minc = min_cluster; self.n_near = 0; self.pans = 0; self.pan_left = 0; self.pan_pos = None
    def plan(self, A):
        from scipy.ndimage import label
        occ = A.occ; cp = A.cam(); shp = np.array(A.shape)
        start = tuple(np.clip(np.floor((cp - A.origin) / GRID).astype(int), 0, shp - 1)); y = start[1]
        band = occ[:, max(0, y - 1):y + 2, :]; o2 = (band == 2).any(1); f2 = (band == 1).any(1) & ~o2; u2 = ~(o2 | f2)
        blocked = binary_dilation(o2, iterations=1)
        for b in self.bad:
            if b[1] == y: blocked[b[0], b[2]] = True
        free = f2 & ~blocked; sx, sz = start[0], start[2]; free[sx, sz] = True
        nr = (slice(max(0, sx - 2), sx + 3), slice(max(0, sz - 2), sz + 3)); free[nr] |= (f2 & ~o2)[nr]  # standing inside the inflated zone: allow leaving through known free cells
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool); fr = binary_dilation(u2, structure=cross) & free
        lab, nl = label(fr, structure=np.ones((3, 3), int)); sizes = np.bincount(lab.ravel()) if nl else np.zeros(1, int)
        dist = np.full(free.shape, -1, np.int32); dist[start[0], start[2]] = 0; par = {}; q = collections.deque([(start[0], start[2])]); W, H = free.shape
        while q:
            c = q.popleft(); dc = dist[c]
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nn = (c[0] + dx, c[1] + dz)
                if 0 <= nn[0] < W and 0 <= nn[1] < H and dist[nn] < 0 and free[nn]:
                    if dx and dz and not (free[c[0] + dx, c[1]] and free[c[0], c[1] + dz]): continue  # no corner cutting
                    dist[nn] = dc + 1; par[nn] = c; q.append(nn)
        best = None; md = getattr(self, "min_d", 1.0); self.dbg = [int(fr.sum()), int(nl), int((sizes[1:] >= self.minc).sum()), int((dist >= 0).sum()), int(free.sum()), int(o2.sum()), int(u2.sum())]
        for cid in range(1, nl + 1):
            if sizes[cid] < self.minc: continue
            cells = np.argwhere(lab == cid); dd = dist[cells[:, 0], cells[:, 1]]; ok = dd >= 0
            if not ok.any(): continue
            k = np.argmin(np.where(ok, dd, 1 << 30)); d_m = dd[k] * GRID; c = (int(cells[k, 0]), int(cells[k, 1]))
            if d_m < md or (c[0], y, c[1]) in self.bad: continue
            sc = min(sizes[cid], 20) / 20.0 - self.LAM * d_m
            if best is None or sc > best[0]: best = (sc, c, d_m, cid)
        if best is None: return False
        c = best[1]; path = []; k = c
        while k != (start[0], start[2]): path.append((k[0], y, k[1])); k = par[k]
        self.path = path[::-1]; self.goal = (c[0], y, c[1]); self.n_near = 0
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # face the unknown
            nn = (c[0] + dx, c[1] + dz)
            if 0 <= nn[0] < W and 0 <= nn[1] < H and u2[nn]: self.target_yaw = float(np.rad2deg(np.arctan2(-dx, -dz)) % 360); break
        else: self.target_yaw = None
        return True

class SensorAgent2(SensorAgent):
    """Adds collision-aware pose correction: a large forward command whose central depth did not shrink was blocked."""
    def __init__(self, env, seed, pose_mode="odom", correct=False):
        super().__init__(env, seed, pose_mode); self.correct = correct and pose_mode == "odom"
        self.pending = None; self.det = dict(tp=0, fp=0, fn=0, tn=0)
    def act(self, kind, mag):
        if kind == "fwd" and mag > 0 and not self.corridor_clear(mag): return False
        if kind == "fwd" and mag < 0 and not self.move_clear(-self.fwd(), -mag): return False  # backwards: not visible, use the map
        if kind == "lat" and not self.move_clear(self.left() * np.sign(mag), abs(mag)): return False
        if kind == "up" and not self.vert_clear(mag): return False
        pre = self.pos.copy(); prev = self._depth.copy(); fwd = self.fwd()
        self.env.step(kind, mag)
        if self.mode == "odom":
            if kind == "yaw": self.yaw = (self.yaw + mag) % 360
            elif kind == "fwd": self.pos = self.pos + fwd * mag
            elif kind == "lat": self.pos = self.pos + self.left() * mag
            else: self.pos = self.pos + np.array([0, mag, 0])
        self.pending = (pre, prev, mag, fwd) if kind == "fwd" and mag >= 0.5 else None
        return True
    def observe(self, rgb, d):
        if self.pending is not None and self.mode == "odom":
            pre, prev, mag, fwd = self.pending; self.pending = None
            a, b = prev[88:168, 88:168], d[88:168, 88:168]; m = (a > 0.3) & (a < 4.5) & (b > 0.05)
            reach = np.percentile(a[a > 0.05], 5) < mag + 0.6 if (a > 0.05).any() else False  # blocked => obstacle within reach
            truth_blocked = self.env.last_blocked
            if m.sum() > 200:
                t_hat = float(np.median((a - b)[m])); det = reach and t_hat < 0.4 * mag
                self.det[("tp" if det else "fn") if truth_blocked else ("fp" if det else "tn")] += 1
                if det and self.correct:
                    self.pos = pre  # blocked: undo the integrated motion
                    idx = np.clip(np.floor((pre + fwd * 0.5 - self.origin) / GRID).astype(int), 0, self.shape - 1); self.occ[tuple(idx)] = 2
        super().observe(rgb, d)

_orig_step = Env.step
def _step(self, kind, mag):
    c0 = self.coll; _orig_step(self, kind, mag); self.last_blocked = self.coll > c0; self.n_act = getattr(self, "n_act", 0) + 1
    kc = self.__dict__.setdefault("kcount", {"yaw": 0, "fwd": 0, "up": 0}); kc[kind] += 1
    self.t_flight = getattr(self, "t_flight", 0.0) + (abs(mag) / 1.0 if kind == "fwd" else abs(mag) / 90.0 if kind == "yaw" else abs(mag) / 0.5)  # 1 m/s forward, 90 deg/s yaw, 0.5 m/s vertical
Env.step = _step
Env.last_blocked = False

def pick_alt(A, z0):
    """New altitude target (+-0.5..2.0 m) that the agent's map says is clear, within [z0-1.5, z0+2.0]. Official VLN trajectories: ~10% of labelled actions are up/down, |net dz|>1 m in 54% of trajectories."""
    c = [dz for dz in (-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0) if z0 - 1.5 <= A.pos[1] + dz <= z0 + 2.0 and A.vert_clear(dz)]
    return float(A.pos[1] + A.rng.choice(c)) if c else None

class _View:
    """Cropped view of the agent's map (known bounding box + margin) so planning cost scales with the explored area, not the full 96 m map."""
    def __init__(self, A, lo, hi): self._A = A; self.lo = lo; self.occ = A.occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; self.shape = np.array(self.occ.shape); self.origin = A.origin + lo * GRID
    def __getattr__(self, k): return getattr(self._A, k)
    def cam(self): return self._A.cam()

def crop_plan(F, A, raw):
    lo = np.maximum(A.bb_lo - 6, 0); hi = np.minimum(A.bb_hi + 7, np.array(A.shape)); V = _View(A, lo, hi)
    gbad = F.bad; F.bad = {tuple(np.array(b) - lo) for b in gbad if all(lo[i] <= b[i] < hi[i] for i in range(3))}
    try: ok = raw(V)
    finally: F.bad = gbad
    if ok: F.path = [tuple(np.array(c) + lo) for c in F.path]; F.goal = tuple(np.array(F.goal) + lo)
    return ok

def diversity(pos, yaw, thumbs, nav_cells):
    pos = np.array(pos); yaw = np.array(yaw); n = len(pos)
    cell05 = [tuple(c) for c in np.floor(pos / 0.5).astype(int)]; last = {}; rev = 0
    for i, c in enumerate(cell05):
        rev += (c in last) and (i - last[c] > 40); last[c] = i
    xz = [tuple(c) for c in np.floor(pos[:, [0, 2]]).astype(int)]
    hist = np.bincount((yaw // 30).astype(int) % 12, minlength=12).astype(float); p = hist / hist.sum(); p = p[p > 0]
    views = {(c, int(y // 30) % 12) for c, y in zip(xz, yaw)}
    T = np.array(thumbs, np.float32); nn = []
    for i in range(1, len(T)): nn.append(float(np.min(np.sqrt(np.mean((T[:i] - T[i]) ** 2, 1)))))
    zz = pos[:, 1]; zdiv = dict(z_range=float(np.percentile(zz, 95) - np.percentile(zz, 5)), z_std=float(zz.std()), alt_bins=len(set(np.floor(zz / 0.5).astype(int))))
    return dict(**zdiv, nav_cov=len(set(xz) & nav_cells) / max(1, len(nav_cells)), revisit=rev / n, yaw_entropy=float(-(p * np.log2(p)).sum() / np.log2(12)),
                views_per_100=len(views) / n * 100, frame_nn=float(np.mean(nn)) if nn else 0.0)

def rollout(policy, S, G, steps, seed, pose_mode, correct, nav_cells, video=None, name="", depth_net=None, lam=0.04, pan_dist=3.0, dist_budget=1e9, smooth=False, mu=0.0, vert=False, p_alt=0.4, cov_stop=2.0):
    rng = np.random.default_rng(seed); env = Env(S, G, rng, depth_net=depth_net); A = SensorAgent2(env, seed, pose_mode, correct)
    p, y = env.reset(); A.init(p, y)
    F = Frontier3(lam) if policy.startswith("frontier3") else (Frontier2(lam, mu, vfront=vert) if policy.startswith("frontier2") else Frontier()); F.smooth = smooth; F.vfront = vert; A.relax = smooth; F.nplan = 0; _raw = F.plan; _pl = lambda A_, F=F, _raw=_raw: crop_plan(F, A_, _raw)
    def _counted(A_, _pl=_pl, F=F): F.nplan += 1; return _pl(A_)
    F.plan = _counted; use_pan = policy == "frontier2"; F.pan_pos = None
    curve, dcurve, frames, P, Y, thumbs, drift = [], [], [], [], [], [], []
    base, lo, sc = top_down(None, G) if video else (None, None, None); t0 = time.time()
    tag = "rgb" if depth_net else "rgbd"; label = f"{policy}{'+corr' if correct else ''}/{pose_mode}"
    env.n_act = 0; t = -1; nxt = 0; acurve = []; idle = 0; escapes = 0; last_act = -1; tcurve = []; alt = None; z0 = A.pos[1]; stc = {}
    while env.n_act < steps and env.dist < dist_budget and t < 3 * steps:
        t += 1
        rgb, d = env.sense(); A.observe(rgb, d); tp = S.pose()
        P.append(tp[0] + [0, CAM_H, 0]); Y.append(tp[1])
        if t % 5 == 0: thumbs.append(cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (16, 16), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0)
        if env.n_act >= nxt:
            nxt += 10; curve.append(env.coverage()); dcurve.append(env.dist); acurve.append(env.n_act); tcurve.append(getattr(env, 't_flight', 0.0)); drift.append(float(np.linalg.norm(A.pos - P[-1])))
            if curve[-1] >= cov_stop: break
        if video and t % 3 == 0:
            m = draw_map(base, lo, sc, env.traj); r = cv2.resize(rgb, (300, 300)).copy()
            cv2.putText(r, f"{label}/{tag} t={t} cov={env.coverage():.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1); frames.append(np.concatenate([r, m], 1))
        if vert and alt is not None:  # executing an altitude change (one 0.25 m step per iteration)
            dz = alt - A.pos[1]
            if abs(dz) < 0.13 or not A.act("up", float(np.sign(dz) * A_UP)):
                alt = None; F.path = []; F.replan_in = 0
            continue
        if vert and policy == "random" and A.rng.random() < 0.025: alt = pick_alt(A, z0)
        idle = idle + 1 if env.n_act == last_act else 0; last_act = env.n_act
        if idle >= 4 and policy != "random":  # stall breaker: the policy keeps proposing moves our depth-safety refuses -> turn away and re-plan
            A.act("yaw", float(A.rng.choice([-1, 1]) * A_TURN_L)); escapes += 1
            F.path = []; F.replan_in = 0
            if idle >= 12: F.bad = set(); idle = 0
            continue
        if policy == "random": pol_random(A); continue  # budget is executed actions
        if policy == "frontier2":
            if F.pan_left > 0:
                A.act("yaw", A_TURN_L); F.pan_left -= 1
                if F.pan_left == 0: F.path = []; F.replan_in = 0
                continue
            if F.replan_in <= 0 or not F.path:
                F.thr = 30
                if F.plan(A):
                    F.replan_in = 25
                    far = F.pan_pos is None or np.linalg.norm(A.pos - F.pan_pos) >= pan_dist
                    if use_pan and far and F.n_near >= 2:
                        F.pan_left = 24; F.pan_pos = A.pos.copy(); F.pans += 1; continue
        elif policy == "frontier2nopan":
            pass
        st = frontier_step(A, F); stc[st] = stc.get(st, 0) + 1
        if os.environ.get("DBG_NF") and st == "no_frontier" and stc[st] == 40: np.savez("/tmp/stuck_map.npz", occ=A.occ, origin=A.origin, pos=A.pos, traj=np.array(env.traj)); print("DBG saved", flush=True)
        if os.environ.get("DBG_NF") and st == "no_frontier" and stc[st] in (1, 30, 300):
            c = np.floor((A.pos - A.origin) / GRID).astype(int); w = A.occ[c[0]-3:c[0]+4, c[1]-3:c[1]+4, c[2]-3:c[2]+4]
            lo_ = np.maximum(A.bb_lo - 6, 0); hi_ = np.minimum(A.bb_hi + 7, np.array(A.shape)); oc = A.occ[lo_[0]:hi_[0], lo_[1]:hi_[1], lo_[2]:hi_[2]]
            print("DBG", stc[st], "t", t, "pos", np.round(A.pos, 1), "z0", round(float(z0), 1), "near 7^3: free", int((w == 1).sum()), "unk", int((w == 0).sum()), "occ", int((w == 2).sum()), "| crop free", int((oc == 1).sum()), "unk", int((oc == 0).sum()), "bad", len(F.bad), "opt", getattr(F, "optimistic", None), "nf", getattr(F, "nf", None), "plan-stages", getattr(F, "dbg", None), flush=True)
        if st == "arrived" and vert and A.rng.random() < p_alt: alt = pick_alt(A, z0)
        if st == "arrived" and F.target_yaw is not None:
            for _ in range(6):
                dy = (F.target_yaw - A.yaw + 180) % 360 - 180
                if abs(dy) < 8: break
                A.act("yaw", float(np.clip(dy, -A_TURN_L, A_TURN_L)))
            F.path = []; F.replan_in = 0
    wall = time.time() - t0
    if video: imageio.mimsave(video, frames, fps=15)
    dv = diversity(P, Y, thumbs, nav_cells); dt = A.det
    return dict(scene=name, policy=policy + ("+corr" if correct else ""), pose=pose_mode, sensor=tag, seed=seed, steps=steps, coverage=env.coverage(),
                curve=curve, dcurve=dcurve, acurve=acurve, iters=t + 1, escapes=escapes, lam=lam, pan_dist=pan_dist, mu=mu, vert=bool(vert), oob=int(A.oob), stc=stc, t_flight=getattr(env, 't_flight', 0.0), tcurve=tcurve, smooth=bool(smooth), nplan=F.nplan, kcount=dict(env.kcount), collisions=env.coll, dist=env.dist, wall_s=wall, ms_per_step=1000 * wall / max(1, env.n_act), final_drift_m=drift[-1], mean_drift_m=float(np.mean(drift)),
                actions=getattr(env, "n_act", 0), pans=getattr(F, "pans", 0), det=dt, div=dv, depth_absrel=float(np.mean(env.absrel)) if env.absrel else None,
                traj=[list(map(float, q)) for q in np.array(P)[::4]], yaws=[float(v) for v in np.array(Y)[::4]])

def nav_cell_set(S, n=4000):
    pts = np.array([S.sim.pathfinder.get_random_navigable_point() for _ in range(n)], float); pts = pts[np.isfinite(pts).all(1)]
    return {tuple(c) for c in np.floor(pts[:, [0, 2]]).astype(int)}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sensor", default="rgbd", choices=["rgbd", "rgb"]); ap.add_argument("--out", default="results_v4")
    ap.add_argument("--configs", default="random:odom:0,frontier1:odom:0,frontier1:odom:1,frontier2nopan:odom:1,frontier2:odom:0,frontier2:odom:1,frontier2:gt:0")
    ap.add_argument("--dist_budget", type=float, default=1e9); ap.add_argument("--cov_stop", type=float, default=2.0); ap.add_argument("--video", default="random:odom:0,frontier1:odom:0,frontier2:odom:1"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    g, n = paths(a.scene); S = UAVSim(g); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds()); nav = nav_cell_set(S)
    dn = DepthNet() if a.sensor == "rgb" else None; out = []; vids = set(a.video.split(","))
    for cfg in a.configs.split(","):
        parts = cfg.split(":"); pol, pm, cr = parts[:3]; kw = dict(x.split("=") for x in parts[3:]); lam = float(kw.get("lam", 0.04)); pd = float(kw.get("pan", 3.0)); sm = kw.get("smooth", "0") == "1"; mu = float(kw.get("mu", 0.0)); vt = kw.get("vert", "0") == "1"; tag_ = "".join(f"_{k}{v}" for k, v in kw.items())
        for sd in range(a.seeds):
            vp = f"{a.out}/{a.scene}__{pol}{'+corr' if cr == '1' else ''}-{pm}-{a.sensor}__{sd}.mp4" if (sd == 0 and cfg in vids) else None
            r = rollout(pol, S, G, a.steps, sd, pm, cr == "1", nav, vp, a.scene, dn, lam, pd, a.dist_budget, sm, mu, vt, 0.4, a.cov_stop); r["cfg"] = cfg; out.append(r)
            print(f"{a.scene} {r['policy']}/{pm}/{a.sensor} s{sd} cov={r['coverage']:.3f} coll={r['collisions']} dist={r['dist']:.0f}m drift={r['final_drift_m']:.2f} pans={r['pans']} det={r['det']} wall={r['wall_s']:.1f}s", flush=True)
    json.dump(out, open(f"{a.out}/{a.scene}__{a.sensor}.json", "w"))
