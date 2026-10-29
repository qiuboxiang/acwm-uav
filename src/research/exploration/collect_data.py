"""Data collection with the sensor-only exploration agent (frontier3 + random fallback + altitude changes), saved in the IndoorUAV layout.

Per trajectory  <out>/<group>/<scene>/traj_<k>/ :
  screenshots/<i>.jpg           recording camera, 720x1280, hfov 90, i = 0..N (frame 0 = initial pose)
  depth_images/<i>.png          optional (--save_depth): uint16 millimetres, 0 = invalid
  saved_transformations/<i>.txt JSON {"action": [x, y, z, yaw_rad], ...}  (same reader as the official load_position)
  posture.txt                   JSON list of [x, y, z, yaw_deg], frames 0..N
  real_action.json              {"frame": [{"frame": i, "actions": {"fly_forward": metres}}, ...]}, i = 1..N, the action that produced frame i
  annotation.json               {"fly_forward_small": [frames], ..., "stop": [N]}; a discrete action is labelled on its last frame
  episode.npz                   states (N+1,4), actions_abs (N,4) = next state, action_class (N,), blocked (N,)
  meta.json                     scene, seed, agent configuration, sensor spec, summary statistics

Coordinates follow the official files: x = habitat x, y = habitat z, z = agent base height (habitat y; the camera sits 1.5 m above it), yaw 0-360 deg
about the vertical axis (positive = turn left).
"""
import os, json, time, shutil, subprocess, argparse, numpy as np, cv2
from uav_sim import UAVSim
from scenes import paths, Geometry
import explore2 as E2
from explore2 import frontier_step, CAM_H, RADIUS, GRID, A_FWD_L, A_FWD_S, A_TURN_L, A_UP
from explore3 import Env, pol_wander, A_LAT
from explore4 import Frontier3, SensorAgent2, crop_plan, pick_alt, nav_cell_set

CLASSES = ["fly_forward_small", "fly_forward_large", "turn_left_small", "turn_left_large", "turn_right_small", "turn_right_large", "fly_up", "fly_down", "fly_left", "fly_right", "fly_backward", "stop"]
MAX_SUB = dict(fwd=0.45, yaw=6.0, up=0.45, lat=0.3)  # per-frame displacement caps, matching the official per-frame real_action magnitudes (forward <=0.49 m, turn <=6.3 deg)


def label_of(kind, mag):
    if kind == "fwd": return ("fly_forward_large" if abs(mag) >= 0.5 else "fly_forward_small") if mag > 0 else "fly_backward"
    if kind == "lat": return "fly_left" if mag > 0 else "fly_right"
    if kind == "yaw": return ("turn_left_" if mag > 0 else "turn_right_") + ("large" if abs(mag) >= 9 else "small")
    return "fly_up" if mag > 0 else "fly_down"


UNION = set()  # surface voxels seen across episodes (evaluation only)
GT_INDEX = {}; VIEWSET = set()  # (gt voxel index * 24 + viewing-direction bin) pairs seen across episodes (dry-run diagnostics)

class Recorder:
    def __init__(self, root, save_depth, jpeg_q=92, video=False, crf=23):
        self.root, self.save_depth, self.q, self.video = root, save_depth, jpeg_q, video; self.i = -1; self.ff = None; self.crf = crf
        for d in (() if video else ("screenshots", "saved_transformations")) + (("depth_images",) if save_depth else ()): os.makedirs(f"{root}/{d}", exist_ok=True)
        os.makedirs(root, exist_ok=True)
        self.states, self.actions, self.cls, self.blocked, self.real, self.ann = [], [], [], [], [], {c: [] for c in CLASSES}

    def frame(self, rgb, depth, base_pos, yaw_deg, real=None, blocked=False):
        self.i += 1; i = self.i; yaw = float(yaw_deg) % 360
        if rgb is None:  # dry run: keep poses only
            self.states.append([float(base_pos[0]), float(base_pos[2]), float(base_pos[1]), float(np.deg2rad(yaw))])
            if i > 0 and real is not None: self.real.append({"frame": i, "actions": real}); self.blocked.append(bool(blocked))
            return
        if self.video:  # one H.264 mp4 per trajectory (frame i = i-th decoded frame); poses live in posture.txt / episode.npz
            if self.ff is None:
                h, w = rgb.shape[:2]
                self.ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "10", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-g", "30", "-threads", "2", "-movflags", "+faststart", f"{self.root}/video.mp4"], stdin=subprocess.PIPE)
            self.ff.stdin.write(np.ascontiguousarray(rgb).tobytes())
        else: cv2.imwrite(f"{self.root}/screenshots/{i}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.q])
        if self.save_depth: cv2.imwrite(f"{self.root}/depth_images/{i}.png", np.clip(depth * 1000, 0, 65535).astype(np.uint16))
        st = [float(base_pos[0]), float(base_pos[2]), float(base_pos[1]), float(np.deg2rad(yaw))]  # official [x, y, z(height), yaw_rad]
        if not self.video: json.dump({"action": st, "agent_state": {"position_habitat": [float(v) for v in base_pos], "yaw_deg": yaw}}, open(f"{self.root}/saved_transformations/{i}.txt", "w"))
        self.states.append(st)
        if i > 0 and real is not None: self.real.append({"frame": i, "actions": real}); self.blocked.append(bool(blocked))

    def close(self):
        if self.ff is not None: self.ff.stdin.close(); self.ff.wait(); self.ff = None

    def label(self, name): self.ann[name].append(self.i)  # last frame of the discrete action


class RecEnv(Env):
    """Env that splits every executed action into per-frame sub-steps (each rendered and recorded) and keeps the agent's small RGB-D observation."""
    def __init__(self, S, G, rng, rec):
        super().__init__(S, G, rng); self.rec = rec; self.n_act = 0; self.kcount = {"yaw": 0, "fwd": 0, "up": 0, "lat": 0}; self.last_blocked = False; self.t_flight = 0.0; self._cur = None
        self.blocked_total = 0
    def _render(self):
        o = self.S.sim.get_sensor_observations(); self._cur = o; return o
    def sense(self):
        o = self._render(); rgb = o["rgb"][..., :3]; d = o["depth"].copy(); v = d > 0.05
        if self.depth_net is not None: dp = self.depth_net(rgb); d = dp
        else: d[v] *= 1 + self.rng.normal(0, self.dn, d.shape)[v]
        p, y = self.S.pose(); c, s_ = np.cos(np.deg2rad(y)), np.sin(np.deg2rad(y)); R = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])  # coverage bookkeeping (evaluation only)
        f = 128 / np.tan(np.deg2rad(self.S.hfov) / 2); u, vv = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4)); r4 = np.stack([(u - 128) / f, -(vv - 128) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        d4 = o["depth"][::4, ::4].reshape(-1); m = (d4 > .05) & (d4 < 8); w = (r4[m] * d4[m, None]) @ R.T + p + [0, CAM_H, 0]
        self.seen |= set(map(tuple, np.floor(w / 0.25).astype(np.int32)))
        if GT_INDEX:  # which directions was each surface voxel seen from (8 azimuth x 3 elevation bins)
            ws = w[::8]; cam = p + [0, CAM_H, 0]; dv = cam - ws; dv /= (np.linalg.norm(dv, axis=1, keepdims=True) + 1e-9)
            b = (((np.arctan2(dv[:, 0], dv[:, 2]) + np.pi) / (2 * np.pi) * 8).astype(int) % 8) * 3 + np.digitize(np.arcsin(np.clip(dv[:, 1], -1, 1)), [-0.35, 0.35])
            for vx, bb in zip(map(tuple, np.floor(ws / 0.25).astype(np.int32)), b):
                gi = GT_INDEX.get(vx)
                if gi is not None: VIEWSET.add(gi * 24 + int(bb))
        return rgb, d
    def snapshot_frame(self, real=None, blocked=False):
        p, y = self.S.pose()
        if self.S.rec is None: self.rec.frame(None, None, p, y, real, blocked); return  # dry run
        o = self._render(); self.rec.frame(o["rec_rgb"][..., :3], o["rec_depth"], p, y, real, blocked)
    def step(self, kind, mag):
        p0, y0 = self.S.pose(); cp = lambda q: q + np.array([0, CAM_H, 0]); self.traj.append(cp(p0).copy())
        total = mag * (1 + self.rng.normal(0, self.fn)) if kind != "yaw" else mag + self.rng.normal(0, self.yn)
        n = max(1, int(np.ceil(abs(total) / MAX_SUB[kind]))); blocked_any = False; done = 0.0
        for k in range(n):
            sub = total / n; p, y = self.S.pose(); r = np.deg2rad(y)
            if kind == "yaw": self.S.set_pose(p, (y + sub) % 360); ok = True
            else:
                vec = {"fwd": np.array([-np.sin(r), 0, -np.cos(r)]), "lat": np.array([-np.cos(r), 0, np.sin(r)]), "up": np.array([0, 1.0, 0])}[kind]
                q = p + vec * sub; ok = self.G.segment_free(cp(p), cp(q), RADIUS)
                if ok: self.S.set_pose(q, y); self.dist += abs(sub)
            if not ok: blocked_any = True; self.coll += 1; self.blocked_total += 1; break
            done += sub
            name = {"fwd": "fly_forward" if sub > 0 else "fly_backward", "lat": "fly_left" if sub > 0 else "fly_right", "up": "fly_up" if sub > 0 else "fly_down", "yaw": "turn_left" if sub > 0 else "turn_right"}[kind]
            self.snapshot_frame({name: round(abs(sub), 4)})
        self.last_blocked = blocked_any; self.n_act += 1; self.kcount[kind] += 1
        self.t_flight += abs(mag) / (1.0 if kind in ("fwd", "lat") else 90.0 if kind == "yaw" else 0.5)
        if done != 0.0 or not blocked_any:  # one label per executed (possibly truncated) discrete action, on its last recorded frame
            self.rec.label(label_of(kind, mag))
            self.rec.cls.append(CLASSES.index(label_of(kind, mag)))


def collect_episode(S, G, seed, root, a, nav):
    rec = Recorder(root, a.save_depth, video=a.video, crf=a.crf); rng = np.random.default_rng(seed); env = RecEnv(S, G, rng, rec); A = SensorAgent2(env, seed, "odom", True)
    p, y = env.reset(); A.init(p, y); A.relax = bool(a.relax); env.snapshot_frame()  # frame 0
    F = Frontier3(a.lam); F.vfront = False; F.align_tol = a.align_tol; F.scan_turns = a.scan_turns; F.translate = bool(a.translate); F.fallback = pol_wander if a.translate else None
    if not a.translate: del F.fallback
    F.nplan = 0; raw = F.plan; F.plan = lambda A_, F=F, raw=raw: crop_plan(F, A_, raw)
    t0 = time.time(); alt = None; z0 = A.pos[1]; idle = 0; last_act = -1; t = -1; stc = {}; escapes = 0; aborted = False; altc = {"trig": 0, "started": 0, "fail_first": 0, "steps": 0}; turn_run = 0; forced = 0
    while rec.i < a.max_frames and env.dist < a.dist_budget and t < 6 * a.max_frames:
        t += 1; rgb, d = env.sense(); A.observe(rgb, d)
        if t == 150 and env.dist < a.min_flight_early: aborted = True; break  # trapped at the start: restart from a new pose
        if alt is not None:
            dz = alt - A.pos[1]
            if abs(dz) < 0.13: alt = None; F.path = []; F.replan_in = 0
            elif not A.act("up", float(np.sign(dz) * A_UP)): alt = None; altc["fail_first"] += 1; F.path = []; F.replan_in = 0
            else: altc["steps"] += 1
            continue
        if A.rng.random() < a.p_alt_step:  # altitude changes also happen mid-flight (official data: ~10% of labelled actions are up/down)
            altc["trig"] += 1; alt = pick_alt(A, z0); altc["started"] += alt is not None
        if turn_run >= a.max_turn_run:  # keep the action mix balanced: after a run of pure turns try to translate
            turn_run = 0
            for s in (A_FWD_L, A_FWD_S):
                if A.corridor_clear(s) and A.act("fwd", s): forced += 1; break
            else:
                if a.translate: pol_wander(A); forced += 1
            continue
        idle = idle + 1 if env.n_act == last_act else 0; last_act = env.n_act
        if idle >= 4:  # stall breaker (see explore4)
            A.act("yaw", float(A.rng.choice([-1, 1]) * A_TURN_L)); escapes += 1; F.path = []; F.replan_in = 0
            if idle >= 12: F.bad = set(); idle = 0
            continue
        y0, f0 = env.kcount["yaw"], env.kcount["fwd"] + env.kcount["up"] + env.kcount["lat"]
        st = frontier_step(A, F); stc[st] = stc.get(st, 0) + 1  # frontier3 planner; falls back to the random policy when it finds no frontier
        if st == "arrived" and A.rng.random() < a.p_alt: alt = pick_alt(A, z0)
        turn_run = turn_run + (env.kcount["yaw"] - y0) if env.kcount["fwd"] + env.kcount["up"] + env.kcount["lat"] == f0 else 0
        if st == "arrived" and F.target_yaw is not None:
            for _ in range(6):
                dy = (F.target_yaw - A.yaw + 180) % 360 - 180
                if abs(dy) < 8: break
                A.act("yaw", float(np.clip(dy, -A_TURN_L, A_TURN_L)))
            F.path = []; F.replan_in = 0
    rec.close()
    if aborted or env.dist < a.min_flight: shutil.rmtree(root, ignore_errors=True); return None  # low-motion episode: discard
    UNION.update(env.seen)
    N = rec.i; rec.label("stop")
    json.dump({"frame": rec.real}, open(f"{root}/real_action.json", "w")); json.dump({k: v for k, v in rec.ann.items()}, open(f"{root}/annotation.json", "w"))
    S_arr = np.array(rec.states, np.float32); json.dump([[s[0], s[1], s[2], float(np.rad2deg(s[3]))] for s in rec.states], open(f"{root}/posture.txt", "w"))
    np.savez_compressed(f"{root}/episode.npz", states=S_arr, actions_abs=S_arr[1:], blocked=np.array(rec.blocked, bool), action_class=np.array(rec.cls, np.int8), classes=np.array(CLASSES))
    ver = sum(1 for r in rec.real if any(k in r["actions"] for k in ("fly_up", "fly_down")))
    meta = dict(scene=a.scene, seed=seed, frames=N + 1, fps=10, rec_resolution=[720, 1280], hfov=90, camera_height=CAM_H, depth_saved=a.save_depth, storage=('video.mp4 (H.264 crf%d, 10 fps)' % a.crf) if a.video else 'screenshots/*.jpg',
                policy="frontier3 (2-D band frontier, lambda=%g) + random fallback + altitude changes (p_alt=%g)" % (a.lam, a.p_alt), pose="dead-reckoned (sensor-only agent)", flight_m=round(env.dist, 1),
                coverage=round(env.coverage(), 3), collisions=env.coll, escapes=escapes, planner_states=stc, altitude=altc, forced_forward=forced, vertical_frame_frac=round(ver / max(1, N), 3), wall_s=round(time.time() - t0, 1),
                coordinate_note="x=habitat x, y=habitat z, z=agent base height (habitat y), yaw deg 0-360 positive=left; camera 1.5 m above base", label_note="one label per executed discrete action on its last frame; sub-steps split to <=0.45 m / <=6 deg per frame")
    json.dump(meta, open(f"{root}/meta.json", "w"), indent=1); return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--out", default="../../../data/collected"); ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--max_frames", type=int, default=600); ap.add_argument("--dist_budget", type=float, default=80.0)
    ap.add_argument("--lam", type=float, default=0.04); ap.add_argument("--p_alt", type=float, default=0.4); ap.add_argument("--save_depth", action="store_true"); ap.add_argument("--relax", type=int, default=1); ap.add_argument("--align_tol", type=float, default=8.0); ap.add_argument("--p_alt_step", type=float, default=0.05); ap.add_argument("--scan_turns", type=int, default=4); ap.add_argument("--max_turn_run", type=int, default=4); ap.add_argument("--translate", type=int, default=1); ap.add_argument("--dry", action="store_true"); ap.add_argument("--video", action="store_true"); ap.add_argument("--crf", type=int, default=23); ap.add_argument("--min_flight", type=float, default=15.0); ap.add_argument("--min_flight_early", type=float, default=3.0);
    a = ap.parse_args(); g, n = paths(a.scene); S = UAVSim(g, rec=None if a.dry else (720, 1280)); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds()); nav = nav_cell_set(S, 500)
    if a.dry: GT_INDEX.update({v: i for i, v in enumerate(G.gt_vox)})
    grp, sc = a.scene.split("_", 1); base = f"{a.out}/{grp}/{sc}"; os.makedirs(base, exist_ok=True)
    ex = [d for d in os.listdir(base) if d.startswith("traj_")]; k0 = len(ex)
    for k in range(a.episodes):
        root = f"{base}/traj_{k0 + k}"; m = None; tries = 0
        while m is None and tries < 8:
            m = collect_episode(S, G, a.seed0 + k + 1000 * tries, root, a, nav); tries += 1
            if m is None: print(f"  discarded low-motion episode (try {tries})", flush=True)
        if m is None: continue
        m["seed"] = a.seed0 + k + 1000 * (tries - 1)
        with open(f"{a.out}/index.jsonl", "a") as f: f.write(json.dumps(dict(path=f"{grp}/{sc}/traj_{k0 + k}", **{x: m[x] for x in ("frames", "flight_m", "coverage", "collisions", "vertical_frame_frac", "seed")})) + "\n")
        cum_frames = locals().get('cum_frames', 0) + m['frames']
        if a.dry and VIEWSET:
            vc = np.bincount(np.fromiter((k // 24 for k in VIEWSET), int), minlength=len(GT_INDEX)); vstat = " views>=1/3/6: %.3f/%.3f/%.3f" % tuple((vc >= q).mean() for q in (1, 3, 6))
        else: vstat = ""
        if a.dry: print(f"UNION episode {k + 1} cum_frames {cum_frames} cum_hours {cum_frames / 36000:.2f} union_coverage {len(UNION & G.gt_vox) / len(G.gt_vox):.3f} this_episode {m['coverage']}{vstat}", flush=True)
        print(a.scene, f"traj_{k0 + k}", "frames", m["frames"], "flight", m["flight_m"], "cov", m["coverage"], "coll", m["collisions"], "vert", m["vertical_frame_frac"], "states", m["planner_states"], f"{m['wall_s']}s", flush=True)
