"""Sensor-only exploration agent (no privileged info).
Agent input : RGB-D from the onboard camera (depth optionally noisy), and its own commanded actions.
Agent state : pose estimated by dead-reckoning (--pose odom) -- or GT pose for ablation (--pose gt).
Agent never sees: mesh / KD-tree / collision flags / GT pose / GT coverage / sim rendering of hypothetical views.
Env (privileged, evaluation + physics only): executes the true motion with noise, blocks motion on real collisions, scores coverage."""
import os, time, json, argparse, numpy as np, imageio, cv2
from uav_sim import UAVSim
from scenes import paths, Geometry
import explore2 as E2
from explore2 import Frontier, frontier_step, top_down, draw_map, CAM_H, RADIUS, GRID, A_FWD_L, A_FWD_S, A_TURN_L, A_UP

class DepthNet:
    """Monocular metric depth (Depth-Anything-V2 Metric-Indoor-Small) -- the only depth source in RGB-only mode."""
    SCALE = 0.41  # one global scalar: our 90deg-HFOV camera is far from the model's training FOV, so raw metric output is ~2.5x too far (median ratio 2.4-2.9 over 4 scenes). Same role as a one-time depth calibration on a real robot.
    def __init__(self):
        import torch; from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        mid = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"; self.torch = torch
        self.dev = "mps" if torch.backends.mps.is_available() else "cpu"
        self.proc = AutoImageProcessor.from_pretrained(mid); self.net = AutoModelForDepthEstimation.from_pretrained(mid).to(self.dev).eval()
    def __call__(self, rgb):
        torch = self.torch; x = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(self.dev)
        with torch.no_grad(): out = self.net(pixel_values=x).predicted_depth
        d = torch.nn.functional.interpolate(out[:, None], size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        return d.float().cpu().numpy() * self.SCALE

class Env:
    def __init__(self, S, G, rng, fwd_noise=0.03, yaw_noise=0.5, depth_noise=0.01, depth_net=None):
        self.depth_net = depth_net; self.absrel = []
        self.S, self.G, self.rng = S, G, rng; self.fn, self.yn, self.dn = fwd_noise, yaw_noise, depth_noise
        self.coll = 0; self.dist = 0.0; self.traj = []; self.seen = set()
    def reset(self):
        for _ in range(500):
            p = np.array(self.S.sim.pathfinder.get_random_navigable_point(), float)
            if np.isfinite(p).all() and not self.G.collides(p + [0, CAM_H, 0], RADIUS + 0.1): break
        y = float(self.rng.integers(0, 360)); self.S.set_pose(p, y); return p, y
    def step(self, kind, mag):
        p, y = self.S.pose(); cp = lambda q: q + np.array([0, CAM_H, 0]); self.traj.append(cp(p).copy())
        if kind == "yaw": self.S.set_pose(p, (y + mag + self.rng.normal(0, self.yn)) % 360); return
        m = mag * (1 + self.rng.normal(0, self.fn)); r = np.deg2rad(y)
        q = p + (np.array([-np.sin(r), 0, -np.cos(r)]) * m if kind == "fwd" else np.array([0, m, 0]))
        if self.G.segment_free(cp(p), cp(q), RADIUS): self.S.set_pose(q, y); self.dist += abs(m)
        else: self.coll += 1  # crash: motion rejected; agent is NOT told
    def sense(self):
        rgb, d = self.S.obs(); d = d.copy(); v = d > 0.05
        if self.depth_net is not None:  # RGB-only: agent's depth comes from the monocular estimator, never from the sensor
            dp = self.depth_net(rgb); self.absrel.append(float(np.median(np.abs(dp[v] - d[v]) / d[v]))); d = dp
        else: d[v] *= 1 + self.rng.normal(0, self.dn, d.shape)[v]  # multiplicative depth noise
        # score true coverage (privileged, evaluation only)
        p, y = self.S.pose(); c, s_ = np.cos(np.deg2rad(y)), np.sin(np.deg2rad(y)); R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        f = 128; u, vv = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4)); r4 = np.stack([(u - 128) / f, -(vv - 128) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        d4 = self.S.obs()[1][::4, ::4].reshape(-1); m = (d4 > .05) & (d4 < 8); w = (r4[m] * d4[m, None]) @ R.T + p + [0, CAM_H, 0]
        self.seen |= set(map(tuple, np.floor(w / 0.25).astype(np.int32)))
        return rgb, d
    def coverage(self): return len(self.seen & self.G.gt_vox) / max(1, len(self.G.gt_vox))

class PoseProxy:  # what E2.frontier_step calls as A.s.pose()
    def __init__(self, a): self.a = a
    def pose(self): return self.a.pos - np.array([0, CAM_H, 0]), self.a.yaw

class SensorAgent:
    """Mapping + planning from RGB-D and estimated pose only."""
    def __init__(self, env, seed, pose_mode="odom", extent=(24, 6, 24)):
        self.env, self.mode = env, pose_mode; self.rng = np.random.default_rng(seed + 1000); self.s = PoseProxy(self)
        self.pos = None; self.yaw = None; self.f = 128
        u, v = np.meshgrid(np.arange(0, 256, 8), np.arange(0, 256, 8)); self.rays = np.stack([(u - 128) / self.f, -(v - 128) / self.f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        self.extent = extent; self.est_traj = []
    def init(self, p, y):  # anchor the map frame at the start pose (world-frame origin is arbitrary in reality)
        self.pos = p + np.array([0, CAM_H, 0]); self.yaw = y
        ex = np.array(self.extent, float); self.origin = self.pos - ex / 2 * np.array([1, 1, 1]); self.shape = np.ceil(ex / GRID).astype(int)
        self.occ = np.zeros(self.shape, np.uint8)
    def cam(self, p=None): return self.pos.copy()
    # ---- perception ----
    def observe(self, rgb, d):
        if self.mode == "gt": p, y = self.env.S.pose(); self.pos = p + np.array([0, CAM_H, 0]); self.yaw = y
        c, s_ = np.cos(np.deg2rad(self.yaw)), np.sin(np.deg2rad(self.yaw)); R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
        dd = d[::8, ::8].reshape(-1); ok = dd > 0.05; dd = np.where(ok, dd, 6.0); hit = ok & (dd < 6.0); dirs = self.rays @ R.T
        for dist in np.arange(0.0, 6.0, GRID / 2):
            m = dist < dd - 0.1; pts = self.pos + dirs[m] * dist
            idx = np.clip(np.floor((pts - self.origin) / GRID).astype(int), 0, self.shape - 1)
            self.occ[tuple(idx.T)] = np.where(self.occ[tuple(idx.T)] == 2, 2, 1)
        hp = self.pos + dirs[hit] * dd[hit, None]; idx = np.clip(np.floor((hp - self.origin) / GRID).astype(int), 0, self.shape - 1); self.occ[tuple(idx.T)] = 2
        self._depth = d; self.est_traj.append(self.pos.copy())
    def corridor_clear(self, dist, axis="fwd"):
        """Depth-only safety check: any depth return inside the UAV's swept box (|lateral|<0.35, |vertical|<0.35) closer than dist+0.3?"""
        d = self._depth[::4, ::4]; u, v = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4)); ok = d > 0.05
        x = (u - 128) / self.f * d; y = -(v - 128) / self.f * d
        if axis == "fwd": box = ok & (np.abs(x) < 0.35) & (np.abs(y) < 0.35) & (d < dist + 0.3)
        else:  # vertical move: use image band above/below centre, require far depth
            box = ok & (np.abs(x) < 0.35) & ((y * np.sign(dist) > 0) & (np.abs(y) < abs(dist) + 0.4)) & (d < 1.2)
        return not box.any()
    def map_clear(self, d):
        for s in np.arange(0.1, d + 1e-6, 0.1):
            i = tuple(np.clip(np.floor((self.pos + self.fwd() * s - self.origin) / GRID).astype(int), 0, self.shape - 1))
            if self.occ[i] != 1: return False
        return self.corridor_clear(d)
    def fwd(self): r = np.deg2rad(self.yaw); return np.array([-np.sin(r), 0, -np.cos(r)])
    def free_dist(self):
        return A_FWD_L if self.corridor_clear(A_FWD_L) else (A_FWD_S if self.corridor_clear(A_FWD_S) else 0.0)
    # ---- control: command the env, update dead-reckoned pose with the COMMANDED motion ----
    def act(self, kind, mag):
        if kind == "up" and not self.corridor_clear(mag, "vert"): return False
        if kind == "fwd" and not self.corridor_clear(mag): return False
        self.env.step(kind, mag)
        if self.mode == "odom":
            if kind == "yaw": self.yaw = (self.yaw + mag) % 360
            elif kind == "fwd": self.pos = self.pos + self.fwd() * mag
            else: self.pos = self.pos + np.array([0, mag, 0])
        return True

def pol_random(A):
    if A.free_dist() >= 0.9 and A.rng.random() < 0.8: A.act("fwd", A_FWD_L)
    elif A.free_dist() >= 0.15 and A.rng.random() < 0.5: A.act("fwd", A_FWD_S)
    else: A.act("yaw", float(A.rng.choice([-A_TURN_L, A_TURN_L])))

def rollout(policy, S, G, steps, seed, pose_mode, video=None, name="", depth_net=None):
    rng = np.random.default_rng(seed); env = Env(S, G, rng, depth_net=depth_net); A = SensorAgent(env, seed, pose_mode)
    p, y = env.reset(); A.init(p, y); F = Frontier(); curve = []; frames = []; drift = []
    tag = "rgb" if depth_net else "rgbd"
    base, lo, sc = top_down(None, G) if video else (None, None, None); t0 = time.time()
    for t in range(steps):
        rgb, d = env.sense(); A.observe(rgb, d)
        if t % 10 == 0: curve.append(env.coverage()); drift.append(float(np.linalg.norm(A.pos - (S.pose()[0] + [0, CAM_H, 0]))))
        if video and t % 2 == 0:
            m = draw_map(base, lo, sc, env.traj)
            for i in range(1, len(A.est_traj)): cv2.line(m, tuple(((A.est_traj[i - 1][[0, 2]] - lo) * sc).astype(int)), tuple(((A.est_traj[i][[0, 2]] - lo) * sc).astype(int)), (0, 150, 0), 1)
            r = cv2.resize(rgb, (300, 300)).copy(); cv2.putText(r, f"{policy}/{pose_mode}/{tag} t={t} cov={env.coverage():.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 0), 1)
            frames.append(np.concatenate([r, m], 1))
        if policy == "random": pol_random(A)
        else:
            st = frontier_step(A, F)
            if st == "arrived":
                if F.target_yaw is not None:
                    for _ in range(6):
                        dy = (F.target_yaw - A.yaw + 180) % 360 - 180
                        if abs(dy) < 8: break
                        A.act("yaw", float(np.clip(dy, -A_TURN_L, A_TURN_L)))
                F.path = []; F.replan_in = 0
    wall = time.time() - t0
    if video: imageio.mimsave(video, frames, fps=15)
    return dict(scene=name, policy=f"{policy}-{pose_mode}" + ("-rgb" if depth_net else "-rgbd"), depth_absrel=float(np.mean(env.absrel)) if env.absrel else None, seed=seed, steps=steps, coverage=env.coverage(), curve=curve, collisions=env.coll,
                dist=env.dist, wall_s=wall, ms_per_step=1000 * wall / steps, final_drift_m=drift[-1], mean_drift_m=float(np.mean(drift)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--out", default="results_sensor"); ap.add_argument("--sensor", default="rgbd", choices=["rgbd", "rgb"]); ap.add_argument("--configs", default="random:odom,frontier:gt,frontier:odom"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    g, n = paths(a.scene); S = UAVSim(g); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds()); out = []
    dn = DepthNet() if a.sensor == "rgb" else None
    for pol, pm in [c.split(":") for c in a.configs.split(",")]:
        for sd in range(a.seeds):
            r = rollout(pol, S, G, a.steps, sd, pm, f"{a.out}/{a.scene}__{pol}-{pm}-{a.sensor}__{sd}.mp4" if sd == 0 else None, a.scene, dn); out.append(r)
            print(f"{a.scene} {r['policy']} s{sd} cov={r['coverage']:.3f} coll={r['collisions']} dist={r['dist']:.0f}m drift={r['final_drift_m']:.2f}m absrel={r['depth_absrel']} wall={r['wall_s']:.1f}s", flush=True)
    json.dump(out, open(f"{a.out}/{a.scene}__{a.sensor}.json", "w"))
