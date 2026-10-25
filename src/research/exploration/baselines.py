"""Exploration baselines for the free-flying UAV. Metric: # unique 0.25m surface voxels observed (3D coverage)."""
import sys, json, time, argparse, numpy as np, imageio
from uav_sim import UAVSim

VOX = 0.25
class Coverage:
    def __init__(self, hfov=90, res=256):
        f = (res / 2) / np.tan(np.deg2rad(hfov) / 2)
        u, v = np.meshgrid(np.arange(res), np.arange(res))
        self.rays = np.stack([(u - res / 2) / f, -(v - res / 2) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        self.seen = set()
    def voxels(self, depth, pos, yaw, cam_h=1.5, stride=4):
        d = depth[::stride, ::stride].reshape(-1); r = self.rays.reshape(256, 256, 3)[::stride, ::stride].reshape(-1, 3)
        m = (d > 0.05) & (d < 8.0); pc = r[m] * d[m, None]
        c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])  # yaw about +y
        w = pc @ R.T + pos + np.array([0, cam_h, 0])
        return set(map(tuple, np.floor(w / VOX).astype(np.int32)))
    def gain(self, vox): return len(vox - self.seen)
    def add(self, vox): self.seen |= vox

def clear_ahead(depth, dist, half=0.2):
    c = depth[96:160, 96:160]; c = c[c > 0.05]
    return c.size == 0 or np.percentile(c, 5) > dist + half

def run(policy, s, steps, seed, save_video=None):
    s.rng = np.random.default_rng(seed)
    s.set_pose(s.sample_start(), float(s.rng.integers(0, 360)))
    cov = Coverage(); curve = []; frames = []; coll = 0
    for t in range(steps):
        rgb, d = s.obs(); p, y = s.pose(); cov.add(cov.voxels(d, p, y))
        if t % 10 == 0: curve.append(len(cov.seen))
        if save_video is not None: frames.append(rgb)
        if policy == "random":
            if clear_ahead(d, 0.9) and s.rng.random() < 0.8: s.step_move(0.9)
            elif clear_ahead(d, 0.15) and s.rng.random() < 0.5: s.step_move(0.15)
            else: s.step_move(0, float(s.rng.choice([-15, 15])))
        elif policy == "nbv":  # next-best-view: evaluate yaw candidates by unseen-voxel gain, then fly
            best, bg = 0, -1
            for dy in (-90, -45, -15, 0, 15, 45, 90):
                s.set_pose(p, (y + dy) % 360); _, dd = s.obs()
                g = cov.gain(cov.voxels(dd, p, (y + dy) % 360)) + (50 if clear_ahead(dd, 0.9) else 0)
                if g > bg: bg, best = g, dy
            s.set_pose(p, (y + best) % 360); _, dd = s.obs()
            dz = float(s.rng.choice([-0.3, 0, 0.3])) if t % 5 == 0 else 0.0
            if clear_ahead(dd, 0.9): s.step_move(0.9, 0, dz)
            elif clear_ahead(dd, 0.15): s.step_move(0.15, 0, dz)
            else: s.step_move(0, 15); coll += 1
    if save_video: imageio.mimsave(save_video, frames, fps=10)
    return {"policy": policy, "seed": seed, "steps": steps, "voxels": len(cov.seen), "curve": curve}

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--scene", default="gibson/Adrian"); a.add_argument("--steps", type=int, default=300)
    a.add_argument("--seeds", type=int, default=3); a.add_argument("--video", action="store_true"); a = a.parse_args()
    base = f"../scenes/scene_datasets/{a.scene}"
    S = UAVSim(base + ".glb"); S.sim.pathfinder.load_nav_mesh(base + ".navmesh")  # one Simulator per process (macOS GL limitation)
    for pol in ("random", "nbv"):
        res = []
        for sd in range(a.seeds):
            t = time.time(); r = run(pol, S, a.steps, sd, f"{pol}_{sd}.mp4" if a.video and sd == 0 else None)
            res.append(r["voxels"]); print(pol, sd, r["voxels"], f"{time.time()-t:.1f}s", flush=True)
        print(f"== {pol}: mean {np.mean(res):.0f} ± {np.std(res):.0f}")
