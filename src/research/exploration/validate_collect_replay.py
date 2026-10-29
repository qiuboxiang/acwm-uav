"""Paired validation of collection-policy changes: rerun recorded collection episodes by seed (dry, no video) with given door /
narrow settings and print one JSON line per seed. With default flags the episodes reproduce the recorded data frame for frame.

  python validate_collect_replay.py --seeds 400012,400150 --door 2 --narrow_ttl 60 --narrow_reset 1 --out ../exploration_v5/door_val/tmp
"""
import sys, os, json, argparse, shutil, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); os.chdir(HERE)
import collect_data5 as CD, collect_data as C
from uav_sim import UAVSim; from scenes import paths, Geometry
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", required=True); ap.add_argument("--door", type=int, default=0); ap.add_argument("--narrow_ttl", type=int, default=400)
ap.add_argument("--narrow_reset", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--keep", action="store_true"); o = ap.parse_args()
a = argparse.Namespace(scene="gibson_Adrian", max_frames=600, dist_budget=80.0, sel="tsp", kappa=0.4, div=1, div_beta=1.0, session_len=1, translate=1, p_alt_step=0.02, p_alt=0.3, max_turn_run=0,
                       save_depth=False, dry=True, video=False, crf=23, min_flight=0.0, min_flight_early=-1, value_model=None, align_tol=20.0, arrive_tol=40.0, yaw_w=6.0, wm=0, rank_model=None, stats=None,
                       door=o.door, narrow_ttl=o.narrow_ttl, narrow_reset=o.narrow_reset)
g, n = paths(a.scene); S = UAVSim(g, rec=None); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds()); C.GT_INDEX.update({v: i for i, v in enumerate(G.gt_vox)})
os.makedirs(o.out, exist_ok=True)
for sd in map(int, o.seeds.split(",")):
    root = f"{o.out}/ep{sd}"; m, st = CD.collect_episode(S, G, sd, root, a, None)
    Q = np.array(json.load(open(root + "/posture.txt"))); dd = np.linalg.norm(Q[:, :2] - Q[0, :2], axis=1)
    X = st["X"]; F = X.F
    rec = dict(seed=sd, cov=m["coverage"], flight=m["flight_m"], coll=m["collisions"], maxd=round(float(dd.max()), 2), cells=int(len(np.unique(np.floor(Q[:, :2]), axis=0))),
               escapes=m["escapes"], states=m["planner_states"], n_door=F.n_door, n_unnarrow=F.n_unnarrow, h0=round(float(Q[0, 2]), 2))
    print(json.dumps(rec), flush=True)
    if not o.keep: shutil.rmtree(root, ignore_errors=True)
