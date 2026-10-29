"""FrontierNet baseline driven by our habitat-sim UAV (sensor-only: RGB + depth + estimated pose; no mesh / GT pose / collision flags).
Runs in the py3.9 habitat env; FrontierNet itself runs in the `frontiernet` conda env via fn_server.py (RPC)."""
import os, sys, time, json, argparse, subprocess, socket, numpy as np, imageio, cv2
from multiprocessing.connection import Client
from uav_sim import UAVSim
from scenes import paths, Geometry
from explore2 import top_down, draw_map, CAM_H, A_FWD_L, A_FWD_S, A_TURN_L, A_UP
from explore3 import Env, DepthNet
from explore4 import SensorAgent2, diversity, nav_cell_set

FN_PY = "/opt/anaconda3/envs/frontiernet/bin/python"
HFOV = 77.3  # FrontierNet's training/demo camera (f=300 @ 480px)
Mz = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)  # habitat (x,y_up,z_back) -> FrontierNet world (z up)

OFF = np.zeros(3)  # world origin = start (x, floor_y, z) in habitat coordinates, set per rollout

def pose_to_W_T_C(pos, yaw_deg, floor_y):
    r = np.deg2rad(yaw_deg); Ry = np.array([[np.cos(r), 0, np.sin(r)], [0, 1, 0], [-np.sin(r), 0, np.cos(r)]])
    T = np.eye(4); T[:3, :3] = Mz @ Ry @ np.diag([1, -1, -1]); T[:3, 3] = Mz @ (pos - OFF); return T

def W_to_hab(p, floor_y): return np.array([p[0], p[2], -p[1]]) + OFF

def pack(rgb, depth, T):
    return dict(cmd="iter", rgb_b=np.ascontiguousarray(rgb, np.uint8).tobytes(), rgb_s=rgb.shape, depth_b=np.ascontiguousarray(depth, np.float32).tobytes(), depth_s=depth.shape,
                pose_b=T.astype(np.float64).tobytes(), pose_s=(4, 4))

def start_server(port):
    p = subprocess.Popen([FN_PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fn_server.py"), "--port", str(port)], stdout=subprocess.DEVNULL, stderr=open(f"/tmp/fn_server_{port}.log", "w"))
    for _ in range(240):
        try: return p, Client(("127.0.0.1", port), authkey=b"acwm")
        except Exception: time.sleep(1)
    raise RuntimeError("fn_server did not start")

def follow(A, tgt_hab, final_yaw):
    """One action toward tgt (habitat world, camera position) using only depth-corridor safety; returns True if an action executed."""
    v = tgt_hab - A.pos; hd = np.hypot(v[0], v[2]); y = A.yaw
    want = np.rad2deg(np.arctan2(-v[0], -v[2])) % 360 if hd > 0.15 else (final_yaw if final_yaw is not None else y)
    dy = (want - y + 180) % 360 - 180
    if abs(dy) > 8: return A.act("yaw", float(np.clip(dy, -A_TURN_L, A_TURN_L)))
    if abs(v[1]) > 0.2 and (hd < 0.4 or abs(v[1]) > 0.5) and A.act("up", float(np.sign(v[1]) * A_UP)): return True  # vertical refused -> continue horizontally
    for s in (A_FWD_L, A_FWD_S):
        if s <= hd + 0.1 and A.corridor_clear(s): return A.act("fwd", s)
    if A.corridor_clear(A_FWD_S): return A.act("fwd", A_FWD_S)
    return A.act("yaw", A_TURN_L)  # blocked ahead: turn and let FrontierNet replan

def rollout(S, G, steps, seed, nav, srv, conn, correct=True, depth_net=None, video=None, name=""):
    rng = np.random.default_rng(seed); env = Env(S, G, rng, depth_net=depth_net); A = SensorAgent2(env, seed, "odom", correct)
    p, y = env.reset(); A.init(p, y); floor_y = p[1]; b = G.bounds; OFF[:] = [p[0], floor_y, p[2]]
    bounds = [float(b[0][0] - p[0]), float(b[1][0] - p[0]), float(-(b[1][2] - p[2])), float(-(b[0][2] - p[2])), 0.5, 2.2]
    bounds = [max(bounds[0], -19.0), min(bounds[1], 19.0), max(bounds[2], -19.0), min(bounds[3], 19.0), 0.5, 2.2]
    f = 128 / np.tan(np.deg2rad(HFOV) / 2)
    conn.send(dict(cmd="init", bounds=bounds, min_dist2occ=0.4, K=[f, 0, 128, 0, f, 128, 0, 0, 1], H=256, W=256)); r = conn.recv()
    assert r.get("ok"), r
    env.n_act = 0; t = -1; nxt = 0; curve, dcurve, acurve, P, Y, thumbs, drift = [], [], [], [], [], [], []; frames = []; ndone = 0; final_done = False; nofront = 0; ndbg = []; hist = []; last_escape = -99; nesc = 0
    base, lo, sc = top_down(None, G) if video else (None, None, None); t0 = time.time(); tfn = 0.0; nfront = []
    while env.n_act < steps and t < 3 * steps:
        t += 1; rgb, d = env.sense(); A.observe(rgb, d); tp = S.pose(); P.append(tp[0] + [0, CAM_H, 0]); Y.append(tp[1])
        if t % 5 == 0: thumbs.append(cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (16, 16), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0)
        if env.n_act >= nxt:
            nxt += 10; curve.append(env.coverage()); dcurve.append(env.dist); acurve.append(env.n_act); drift.append(float(np.linalg.norm(A.pos - P[-1])))
        if video and t % 3 == 0:
            m = draw_map(base, lo, sc, env.traj); r_ = cv2.resize(rgb, (300, 300)).copy()
            cv2.putText(r_, f"FrontierNet t={t} cov={env.coverage():.2f}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1); frames.append(np.concatenate([r_, m], 1))
        t1 = time.time(); conn.send(pack(rgb, d, pose_to_W_T_C(A.pos, A.yaw, floor_y))); res = conn.recv(); tfn += time.time() - t1
        if "error" in res: print(res["error"][-800:]); break
        nfront.append(res.get("nfront", 0))
        if os.environ.get('FN_DEBUG') and t % 15 == 0: print('dbg', t, env.n_act, 'nfront', res.get('nfront'), 'npath', res.get('n_path'), 'ndet', res.get('n_detect'), 'pos', np.round(A.pos, 1), 'yaw', int(A.yaw), 'coll', env.coll)
        if res["done"]:
            nofront += 1; ndbg.append((t, res.get('nfts'), res.get('n')))
            if nofront > 80: final_done = True; break
            A.act("yaw", A_TURN_L); continue
        nofront = 0
        if res.get("next_b") is None: A.act("yaw", A_TURN_L); continue  # no path yet: scan
        T = np.frombuffer(res["next_b"], np.float64).reshape(4, 4); tgt = W_to_hab(T[:3, 3], floor_y)
        fwd_w = T[:3, :3] @ np.array([0, 0, 1.0]); fh = np.array([fwd_w[0], fwd_w[2], -fwd_w[1]])
        fy = float(np.rad2deg(np.arctan2(-fh[0], -fh[2])) % 360) if res["last"] else None
        hist.append(A.pos.copy())
        if len(hist) > 14 and np.linalg.norm(hist[-1] - hist[-14]) < 0.15 and env.n_act - last_escape > 14:  # stalled: path not executable with our safety margin
            conn.send(dict(cmd="reset_path")); conn.recv(); last_escape = env.n_act; nesc += 1
            A.act("yaw", float(rng.choice([-1, 1]) * rng.choice([45, 60, 90]) // 15 * 15 if False else rng.choice([-1, 1]) * A_TURN_L))
            for _ in range(int(rng.integers(3, 7))): A.act("yaw", A_TURN_L * float(np.sign(rng.random() - .5)))
            for _ in range(3):
                if A.corridor_clear(A_FWD_L): A.act("fwd", A_FWD_L)
            hist.clear(); continue
        follow(A, tgt, fy)
    wall = time.time() - t0
    if os.environ.get('FN_DEBUG'): print('nofront events', len(ndbg), ndbg[:5], 'last nfront', nfront[-5:])
    if video and frames: imageio.mimsave(video, frames, fps=15)
    return dict(scene=name, policy="frontiernet" + ("+corr" if correct else ""), pose="odom", sensor="rgb" if depth_net else "rgbd", seed=seed, steps=steps, coverage=env.coverage(), curve=curve, dcurve=dcurve,
                acurve=acurve, iters=t + 1, collisions=env.coll, dist=env.dist, wall_s=wall, fn_s=tfn, ms_per_step=1000 * wall / max(1, env.n_act), final_drift_m=drift[-1] if drift else 0, mean_drift_m=float(np.mean(drift)) if drift else 0,
                actions=getattr(env, "n_act", 0), pans=0, det=A.det, div=diversity(P, Y, thumbs, nav), depth_absrel=float(np.mean(env.absrel)) if env.absrel else None, finished=final_done, escapes=nesc,
                traj=[list(map(float, q)) for q in np.array(P)[::4]], yaws=[float(v) for v in np.array(Y)[::4]])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--steps", type=int, default=1000); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sensor", default="rgbd", choices=["rgbd", "rgb"]); ap.add_argument("--out", default="results_v4"); ap.add_argument("--port", type=int, default=0); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    g, n = paths(a.scene); S = UAVSim(g, hfov=HFOV); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds()); nav = nav_cell_set(S)
    dn = DepthNet() if a.sensor == "rgb" else None; out = []
    for sd in range(a.seeds):
        port = a.port or (20000 + np.random.randint(0, 20000)); proc, conn = start_server(port)
        try:
            r = rollout(S, G, a.steps, sd, nav, conn, conn, True, dn, f"{a.out}/{a.scene}__frontiernet-odom-{a.sensor}__{sd}.mp4" if sd == 0 else None, a.scene) if False else rollout(S, G, a.steps, sd, nav, None, conn, True, dn, f"{a.out}/{a.scene}__frontiernet-odom-{a.sensor}__{sd}.mp4" if sd == 0 else None, a.scene)
        finally:
            try: conn.send(dict(cmd="quit")); conn.close()
            except Exception: pass
            proc.terminate()
        out.append(r); print(f"{a.scene} {r['policy']}/{a.sensor} s{sd} cov={r['coverage']:.3f} coll={r['collisions']} dist={r['dist']:.0f}m actions={r['actions']} fn={r['fn_s']:.0f}s wall={r['wall_s']:.0f}s finished={r['finished']}", flush=True)
    json.dump(out, open(f"{a.out}/{a.scene}__frontiernet_{a.sensor}.json", "w"))
