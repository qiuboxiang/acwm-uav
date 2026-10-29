"""Data collection with the v5 sensor-only agent, saved in the same IndoorUAV layout as collect_data.py (see that file's docstring).

Agent (no privileged information):
  log-odds map + 4-DoF depth-ICP odometry (proposal 1), hierarchical frontier planner (proposal 2), view-diversity mode with a
  visit-count archive once frontiers run out (proposal 4b/4c), optional learned frontier value (proposal 3a).
Session mode (proposal 4c, ``--session_len K``): K consecutive episodes form one continuous flight. Each episode starts where the
previous one ended and the agent keeps its map, odometry keyframes, view bitmask and visit archive, so later episodes go where the
earlier ones did not. A new session starts from a fresh random pose with an empty map (a new battery / take-off), so no
relocalisation against a prior map is needed.
Simulator side: identical to collect_data.py (per-frame sub-steps, labels), plus a geofence at the scene bounds (scans have holes
to the void outside; motion out of the bounds is blocked like a collision and the agent is not told).
"""
import os, json, time, shutil, argparse, collections
import numpy as np

import collect_data as C
from collect_data import Recorder, RecEnv, CLASSES, label_of
from uav_sim import UAVSim
from scenes import paths, Geometry
from explore2 import CAM_H, RADIUS, A_FWD_L, A_FWD_S
from explore3 import Env
from explore4 import nav_cell_set
from explore5 import SensorAgent5, Explorer, HierPlanner, pol_wander5
import mapping5 as M


class RecEnv5(RecEnv):
    def __init__(self, S, G, rng, rec):
        super().__init__(S, G, rng, rec); self.n_fence = 0; self.n_back = 0

    def reset(self, seed=None):
        self.G.ensure_main(self.S.sim.pathfinder)
        if seed is not None: self.S.sim.pathfinder.seed(int(seed) + 12345)
        return Env.reset(self)

    def step(self, kind, mag):  # collect_data.RecEnv.step + geofence
        p0, y0 = self.S.pose(); cp = lambda q: q + np.array([0, CAM_H, 0]); self.traj.append(cp(p0).copy())
        total = mag * (1 + self.rng.normal(0, self.fn)) if kind != "yaw" else mag + self.rng.normal(0, self.yn)
        n = max(1, int(np.ceil(abs(total) / C.MAX_SUB[kind]))); blocked_any = False; done = 0.0
        for k in range(n):
            sub = total / n; p, y = self.S.pose(); r = np.deg2rad(y)
            if kind == "yaw": self.S.set_pose(p, (y + sub) % 360); ok = True
            else:
                vec = {"fwd": np.array([-np.sin(r), 0, -np.cos(r)]), "lat": np.array([-np.cos(r), 0, np.sin(r)]), "up": np.array([0, 1.0, 0])}[kind]
                q = p + vec * sub; c = cp(q); inside = bool(((c >= self.G.bounds[0] + 0.3) & (c <= self.G.bounds[1] - 0.3)).all())
                ok = inside and self.G.segment_free(cp(p), c, RADIUS); self.n_fence += int(not inside)
                if ok: self.S.set_pose(q, y); self.dist += abs(sub)
            if not ok: blocked_any = True; self.coll += 1; self.blocked_total += 1; break
            done += sub
            name = {"fwd": "fly_forward" if sub > 0 else "fly_backward", "lat": "fly_left" if sub > 0 else "fly_right", "up": "fly_up" if sub > 0 else "fly_down", "yaw": "turn_left" if sub > 0 else "turn_right"}[kind]
            self.snapshot_frame({name: round(abs(sub), 4)})
        self.last_blocked = blocked_any; self.n_act += 1; self.kcount[kind] += 1; self.n_back += int(kind == "fwd" and mag < 0)
        self.t_flight += abs(mag) / (1.0 if kind in ("fwd", "lat") else 90.0 if kind == "yaw" else 0.5)
        if done != 0.0 or not blocked_any:
            self.rec.label(label_of(kind, mag)); self.rec.cls.append(CLASSES.index(label_of(kind, mag)))


def policy_cfg(a):
    return dict(plan="hier", pose="vo", map="logodds", sel=a.sel, div=bool(a.div), div_beta=a.div_beta, translate=bool(a.translate), fallback="wander5",
                p_alt_step=a.p_alt_step, p_alt=a.p_alt, vert=False, infl=1, kappa=a.kappa, align_tol=a.align_tol, arrive_tol=a.arrive_tol, yaw_w=a.yaw_w,
                door=int(getattr(a, "door", 0)), narrow_ttl=int(getattr(a, "narrow_ttl", 400)), narrow_reset=int(getattr(a, "narrow_reset", 0)))


def collect_episode(S, G, seed, root, a, state, value_fn=None):
    """One trajectory. ``state`` carries the agent across episodes of a session (None = start a new session)."""
    rec = Recorder(root, a.save_depth, video=a.video, crf=a.crf); rng = np.random.default_rng(seed); env = RecEnv5(S, G, rng, rec)
    new = state is None
    if new:
        p, y = env.reset(seed); A = SensorAgent5(env, seed, "vo", False, "logodds"); A.fwm = M.ForwardEnsemble(seed=seed) if a.wm else None; A.init(p, y, (G.bounds[0] + 0.3, G.bounds[1] - 0.3)); A.relax = True
        X = Explorer(policy_cfg(a), seed, value_fn); state = dict(A=A, X=X, k=0, sid=seed)
    else:
        A, X = state["A"], state["X"]; A.env = env; X.F.path = []; X.F.replan_in = 0; X.q.clear(); X.alt = None
    env.snapshot_frame()  # frame 0
    F = X.F; t0 = time.time(); idle = 0; last = -1; t = -1; aborted = False; turn_run = 0; forced = 0
    nd0, nf0, nfail0 = F.n_div, F.n_front, F.n_fail; stc0 = collections.Counter(X.stc); esc0 = X.escapes
    while rec.i < a.max_frames and env.dist < a.dist_budget and t < 6 * a.max_frames:
        t += 1; rgb, d = env.sense(); A.observe(rgb, d)
        if new and t == 150 and env.dist < a.min_flight_early: aborted = True; break  # trapped at the start: restart from a new pose
        idle = idle + 1 if env.n_act == last else 0; last = env.n_act
        if a.max_turn_run and turn_run >= a.max_turn_run:  # optional action-mix balancing (as collect_data.py)
            turn_run = 0
            for s_ in (A_FWD_L, A_FWD_S):
                if A.corridor_clear(s_) and A.act("fwd", s_): forced += 1; break
            continue
        y0 = env.kcount["yaw"]; m0 = env.kcount["fwd"] + env.kcount["up"] + env.kcount["lat"]
        X.step(A, idle)
        turn_run = turn_run + (env.kcount["yaw"] - y0) if env.kcount["fwd"] + env.kcount["up"] + env.kcount["lat"] == m0 else 0
    rec.close()
    if aborted or env.dist < a.min_flight: shutil.rmtree(root, ignore_errors=True); return None, None
    C.UNION.update(env.seen)
    N = rec.i; rec.label("stop")
    json.dump({"frame": rec.real}, open(f"{root}/real_action.json", "w")); json.dump({k: v for k, v in rec.ann.items()}, open(f"{root}/annotation.json", "w"))
    S_arr = np.array(rec.states, np.float32); json.dump([[s[0], s[1], s[2], float(np.rad2deg(s[3]))] for s in rec.states], open(f"{root}/posture.txt", "w"))
    np.savez_compressed(f"{root}/episode.npz", states=S_arr, actions_abs=S_arr[1:], blocked=np.array(rec.blocked, bool), action_class=np.array(rec.cls, np.int8), classes=np.array(CLASSES))
    ver = sum(1 for r in rec.real if any(k in r["actions"] for k in ("fly_up", "fly_down")))
    tp = S.pose(); drift = float(np.linalg.norm(A.pos - (tp[0] + [0, CAM_H, 0])))
    stc = collections.Counter(X.stc); stc.subtract(stc0)
    meta = dict(scene=a.scene, seed=seed, session=state["sid"], episode_in_session=state["k"], frames=N + 1, fps=10, rec_resolution=[720, 1280], hfov=90, camera_height=CAM_H,
                depth_saved=a.save_depth, storage=('video.mp4 (H.264 crf%d, 10 fps)' % a.crf) if a.video else 'screenshots/*.jpg',
                policy=f"v5: log-odds map + depth-ICP odometry + hierarchical frontier (sel={a.sel}) + view-diversity mode (div={a.div}) + session_len={a.session_len}" + (f" + door mode, narrow_ttl={a.narrow_ttl}, narrow_reset={a.narrow_reset}" if getattr(a, "door", 0) else ""),
                pose="depth-ICP odometry (sensor-only agent)", flight_m=round(env.dist, 1), coverage=round(env.coverage(), 3), collisions=env.coll, fence_blocks=env.n_fence, backward=env.n_back,
                escapes=X.escapes - esc0, planner_states={k: v for k, v in stc.items() if v}, frontier_targets=F.n_front - nf0, diversity_targets=F.n_div - nd0, plan_failures=F.n_fail - nfail0,
                forced_forward=forced, watchdog=X.n_watchdog, vertical_frame_frac=round(ver / max(1, N), 3), final_pose_err_m=round(drift, 3), vo=dict(A.vo.stats), wall_s=round(time.time() - t0, 1),
                coordinate_note="x=habitat x, y=habitat z, z=agent base height (habitat y), yaw deg 0-360 positive=left; camera 1.5 m above base",
                label_note="one label per executed discrete action on its last frame; sub-steps split to <=0.45 m / <=6 deg per frame")
    json.dump(meta, open(f"{root}/meta.json", "w"), indent=1)
    state["k"] += 1
    return meta, state


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--out", default="../../../data/collected_v5"); ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--max_frames", type=int, default=600); ap.add_argument("--dist_budget", type=float, default=80.0)
    ap.add_argument("--sel", default="tsp"); ap.add_argument("--kappa", type=float, default=0.4); ap.add_argument("--div", type=int, default=1); ap.add_argument("--div_beta", type=float, default=1.0)
    ap.add_argument("--session_len", type=int, default=1); ap.add_argument("--translate", type=int, default=1); ap.add_argument("--p_alt_step", type=float, default=0.02); ap.add_argument("--p_alt", type=float, default=0.3)
    ap.add_argument("--max_turn_run", type=int, default=0); ap.add_argument("--save_depth", action="store_true"); ap.add_argument("--dry", action="store_true"); ap.add_argument("--video", action="store_true")
    ap.add_argument("--crf", type=int, default=23); ap.add_argument("--min_flight", type=float, default=15.0); ap.add_argument("--min_flight_early", type=float, default=3.0); ap.add_argument("--value_model", default=None)
    ap.add_argument("--align_tol", type=float, default=20.0, help="fly forward when the waypoint is within this heading error (deg)")
    ap.add_argument("--arrive_tol", type=float, default=40.0, help="skip the look-at-target turn when the target is already this close to the heading")
    ap.add_argument("--yaw_w", type=float, default=6.0, help="planner cost (m) of a 180 deg heading change towards a target")
    ap.add_argument("--wm", type=int, default=0, help="4a: forward-model-ensemble novelty in the diversity mode")
    ap.add_argument("--rank_model", default=None)
    ap.add_argument("--door", type=int, default=0, help="door / narrow-passage traversal in the executor: 0 off, 1 whenever the normal check refuses, 2 only at passages refused before")
    ap.add_argument("--narrow_ttl", type=int, default=400, help="observations a passage stays marked too narrow after a refused step")
    ap.add_argument("--narrow_reset", type=int, default=0, help="1 = clear narrow marks when the map grows and retry them before giving up")
    ap.add_argument("--stats", default=None, help="jsonl file for per-episode cumulative ground-truth diversity (evaluation only)")
    a = ap.parse_args(); g, n = paths(a.scene); S = UAVSim(g, rec=None if a.dry else (720, 1280)); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds())
    C.GT_INDEX.update({v: i for i, v in enumerate(G.gt_vox)})  # evaluation-only bookkeeping of (voxel, view-direction) pairs
    vf = None
    if a.value_model or a.rank_model:
        import frontier_value; vf = frontier_value.load(a.rank_model if a.sel == "rank" else a.value_model)
    grp, sc = a.scene.split("_", 1); base = f"{a.out}/{grp}/{sc}"; os.makedirs(base, exist_ok=True)
    ex = [d for d in os.listdir(base) if d.startswith("traj_")]; k0 = len(ex); state = None; cum_frames = 0
    for k in range(a.episodes):
        root = f"{base}/traj_{k0 + k}"; m = None; tries = 0
        if state is not None and state["k"] >= a.session_len: state = None  # battery change: new session from a fresh pose
        while m is None and tries < 8:
            seed = a.seed0 + k + 1000 * tries
            m, st = collect_episode(S, G, seed, root, a, state if tries == 0 else None, vf); tries += 1
            if m is None: state = None; print(f"  discarded low-motion episode (try {tries})", flush=True)
        if m is None: continue
        state = st; cum_frames += m["frames"]
        with open(f"{a.out}/index.jsonl", "a") as f: f.write(json.dumps(dict(path=f"{grp}/{sc}/traj_{k0 + k}", **{x: m[x] for x in ("frames", "flight_m", "coverage", "collisions", "vertical_frame_frac", "seed", "session", "episode_in_session")})) + "\n")
        vc = np.bincount(np.fromiter((v // 24 for v in C.VIEWSET), int, len(C.VIEWSET)), minlength=len(C.GT_INDEX)); vs = tuple(float((vc >= q).mean()) for q in (1, 3, 6))
        uc = len(C.UNION & G.gt_vox) / len(G.gt_vox)
        if a.stats:
            with open(a.stats, "a") as f: f.write(json.dumps(dict(episode=k, cum_frames=cum_frames, union_coverage=uc, views_ge1=vs[0], views_ge3=vs[1], views_ge6=vs[2], **{x: m[x] for x in ("coverage", "flight_m", "collisions", "frames", "session", "episode_in_session", "diversity_targets", "frontier_targets")})) + "\n")
        print(a.scene, f"traj_{k0 + k}", "frames", m["frames"], "flight", m["flight_m"], "cov", m["coverage"], "coll", m["collisions"], "fence", m["fence_blocks"], "back", m["backward"], "vert", m["vertical_frame_frac"],
              "sess", m["session"], m["episode_in_session"], "front/div", m["frontier_targets"], m["diversity_targets"], "err", m["final_pose_err_m"], f"union {uc:.3f} views>=1/3/6 {vs[0]:.3f}/{vs[1]:.3f}/{vs[2]:.3f}", f"{m['wall_s']}s", flush=True)
