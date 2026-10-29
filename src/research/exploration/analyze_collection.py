"""Evaluation-only comparison of collected datasets (e.g. the v4 10 h run vs the v5 rerun) on the same scene.

Replays the recorded poses (posture.txt) in the simulator at the agent resolution (256x256 depth, hfov 90) and measures every
dataset with the same pipeline, so differences come from the trajectories, not from bookkeeping:
  union surface-voxel coverage vs collected hours, (voxel, viewing-direction) multiplicity, per-trajectory coverage,
  navigable-area spread, altitude / heading spread, action-label mix vs the official IndoorUAV sample, frame redundancy.

  python analyze_collection.py replay --data DIR --scene S --out stats.npz [--stride 2] [--workers 4 --worker k]
  python analyze_collection.py summary --data DIR --replay stats.npz --out summary.json
"""
from __future__ import annotations

import argparse, glob, json, os, collections
import numpy as np

GRID = 0.25
OFFICIAL = {"turn": 0.448, "forward": 0.397, "up_down": 0.105, "lateral": 0.040, "backward": 0.0}  # official sample, 299 trajectories (docs/exploration/data_collection.md)


def all_trajs(data):
    """Trajectories in approximate collection-time order: parallel workers w0..w3 interleaved by trajectory index."""
    out = []
    for wi, w in enumerate(sorted(glob.glob(f"{data}/w*")) or [data]):
        out += [(int(p.rsplit("_", 1)[1]), wi, p) for p in glob.glob(f"{w}/*/*/traj_*") if os.path.exists(f"{p}/meta.json")]
    return [p for _, _, p in sorted(out)]


def replay(data, scene, out, stride=2, workers=1, worker=0):
    from uav_sim import UAVSim
    from scenes import paths, Geometry
    import mapping5 as M
    g, n = paths(scene); S = UAVSim(g); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds())
    gt = {v: i for i, v in enumerate(sorted(G.gt_vox))}; T = all_trajs(data)[worker::workers]
    f = 128 / np.tan(np.deg2rad(45)); u, v = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4))
    r4 = np.stack([(u - 128) / f, -(v - 128) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
    per = []; masks = []; seen_all = np.zeros(len(gt), np.int32); bits = np.zeros(len(gt), np.uint32)
    lo, hi = G.bounds
    for k, tp in enumerate(T):
        P = np.array(json.load(open(f"{tp}/posture.txt")), float); seen = set(); oob = 0
        for x, y, z, yaw in P[::stride]:
            base = np.array([x, z, y]); S.set_pose(base, yaw); d = S.obs()[1]; cam = base + [0, 1.5, 0]
            oob += int(((cam < lo) | (cam > hi)).any())
            d4 = d[::4, ::4].reshape(-1); m = (d4 > .05) & (d4 < 8); w = (r4[m] * d4[m, None]) @ M.yaw_R(yaw).T + cam
            vx = np.floor(w / 0.25).astype(np.int32); b = M.view_bins(cam[None] - w)
            idx = np.array([gt.get(tuple(q), -1) for q in vx[::2]], dtype=np.int64); bb = b[::2]; ok = idx >= 0
            np.bitwise_or.at(bits, idx[ok], (np.uint32(1) << bb[ok].astype(np.uint32)))
            seen |= set(idx[ok].tolist())
        s = np.fromiter(seen, int, len(seen)); seen_all[s] += 1; mask = np.zeros(len(gt), bool); mask[s] = True
        per.append(dict(path=tp, frames=len(P), coverage=len(s) / len(gt), oob_frac=oob / max(1, len(P[::stride])))); masks.append(np.packbits(mask))
        if k % 20 == 0: print(worker, k, len(T), f"union {(seen_all > 0).mean():.3f}", flush=True)
    np.savez_compressed(out, seen=seen_all, bits=bits, masks=np.array(masks), per=json.dumps(per), n_gt=len(gt))


def merge(files):
    seen = bits = None; per = []; masks = {}
    for f in files:
        z = np.load(f, allow_pickle=True); pp = json.loads(str(z["per"]))
        seen = z["seen"].copy() if seen is None else seen + z["seen"]; bits = z["bits"].copy() if bits is None else bits | z["bits"]; per += pp
        for p, m in zip(pp, z["masks"]): masks[p["path"]] = m
    return seen, bits, per, masks


def popcount(x):
    c = np.zeros(x.shape, np.int32)
    for k in range(24): c += ((x >> np.uint32(k)) & np.uint32(1)).astype(np.int32)
    return c


def action_mix(T):
    lab = collections.Counter(); fr = collections.Counter()
    for tp in T:
        ann = json.load(open(f"{tp}/annotation.json"))
        for k, v in ann.items(): lab[k] += len(v)
        for r in json.load(open(f"{tp}/real_action.json"))["frame"]:
            for k in r["actions"]: fr[k] += 1
    grp = lambda c: dict(turn=sum(v for k, v in c.items() if k.startswith("turn")), forward=sum(v for k, v in c.items() if k.startswith("fly_forward")), up_down=c["fly_up"] + c["fly_down"],
                         lateral=c["fly_left"] + c["fly_right"], backward=c["fly_backward"])
    gl = grp(lab); n = sum(gl.values()); gf = grp(fr); nf = sum(gf.values())
    lab_share = {k: v / n for k, v in gl.items()}
    return dict(labels=lab_share, frames={k: v / nf for k, v in gf.items()}, l1_to_official=float(sum(abs(lab_share[k] - OFFICIAL[k]) for k in OFFICIAL)), raw=dict(lab))


def summary(data, replay_files, out, nav_cells=None):
    T = all_trajs(data); metas = [json.load(open(f"{t}/meta.json")) for t in T]
    seen, bits, per, masks = merge(replay_files); pc = popcount(bits); n_gt = len(seen)
    order = {p["path"]: p for p in per}; P = [order[t] for t in T if t in order]
    # union coverage vs hours, in collection order (workers interleaved by trajectory index)
    frames = np.array([p["frames"] for p in P]); u = np.zeros(n_gt, bool); newv = []
    for p in P:
        m = np.unpackbits(masks[p["path"]])[:n_gt].astype(bool); newv.append(int((m & ~u).sum())); u |= m
    newv = np.array(newv)
    pos = []; zr = []; net_dz = []; yaw_h = np.zeros(12); starts = []
    for t in T:
        A = np.array(json.load(open(f"{t}/posture.txt")), float); pos.append(A[:, :3]); zr.append(np.percentile(A[:, 2], 95) - np.percentile(A[:, 2], 5)); net_dz.append(abs(A[-1, 2] - A[0, 2]))
        yaw_h += np.bincount((A[:, 3] // 30).astype(int) % 12, minlength=12); starts.append(A[0, :3])
    allp = np.concatenate(pos); cells = {tuple(c) for c in np.floor(allp[:, :2]).astype(int)}
    per_traj_cells = [len({tuple(c) for c in np.floor(p[:, :2]).astype(int)}) for p in pos]
    p_ = yaw_h / yaw_h.sum(); yaw_ent = float(-(p_[p_ > 0] * np.log2(p_[p_ > 0])).sum() / np.log2(12))
    res = dict(n_traj=len(T), frames=int(sum(m["frames"] for m in metas)), hours=sum(m["frames"] for m in metas) / 36000.0,
               union_coverage=float((seen > 0).mean()), views_ge1=float((pc >= 1).mean()), views_ge3=float((pc >= 3).mean()), views_ge6=float((pc >= 6).mean()), mean_views=float(pc.mean()),
               traj_coverage_mean=float(np.mean([p["coverage"] for p in P])), traj_coverage_median=float(np.median([p["coverage"] for p in P])),
               flight_m_mean=float(np.mean([m["flight_m"] for m in metas])), flight_m_median=float(np.median([m["flight_m"] for m in metas])),
               collisions_mean=float(np.mean([m["collisions"] for m in metas])), traj_with_collision=float(np.mean([m["collisions"] > 0 for m in metas])),
               oob_frame_frac=float(np.mean([p["oob_frac"] for p in P])), traj_with_oob=float(np.mean([p["oob_frac"] > 0 for p in P])),
               xz_cells_1m=len(cells), xz_cells_per_traj=float(np.mean(per_traj_cells)), z_range_mean=float(np.mean(zr)), frac_net_dz_gt1=float(np.mean(np.array(net_dz) > 1.0)),
               yaw_entropy=yaw_ent, start_cells_1m=len({tuple(c) for c in np.floor(np.array(starts)[:, :2]).astype(int)}),
               vertical_frame_frac=float(np.mean([m["vertical_frame_frac"] for m in metas])), actions=action_mix(T),
               curve=dict(hours=(np.cumsum(frames) / 36000.0).tolist(), union=(np.cumsum(newv) / n_gt).tolist()))
    for k in ("final_pose_err_m", "fence_blocks", "backward", "diversity_targets", "frontier_targets"):
        v = [m[k] for m in metas if k in m]
        if v: res[k + "_mean"] = float(np.mean(v))
    json.dump(res, open(out, "w"), indent=1); return res


def dry(root, out_json, out_png, variants=None, n_max=None):
    """Collection-policy comparison from dry runs (in-loop ground-truth bookkeeping, identical code path for v4 and v5)."""
    import re
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    curves = {}
    for v in variants or sorted(os.path.basename(d) for d in glob.glob(f"{root}/*") if os.path.isdir(d)):
        rows = []
        if os.path.exists(f"{root}/{v}/stats.jsonl"):
            rows = [json.loads(l) for l in open(f"{root}/{v}/stats.jsonl")]
            rows = [dict(frames=r["cum_frames"], union=r["union_coverage"], v1=r["views_ge1"], v3=r["views_ge3"], v6=r["views_ge6"]) for r in rows]
        elif os.path.exists(f"{root}/{v}.log"):
            for l in open(f"{root}/{v}.log"):
                m = re.search(r"UNION episode (\d+) cum_frames (\d+) .*union_coverage ([\d.]+).*views>=1/3/6: ([\d.]+)/([\d.]+)/([\d.]+)", l)
                if m: rows.append(dict(frames=int(m.group(2)), union=float(m.group(3)), v1=float(m.group(4)), v3=float(m.group(5)), v6=float(m.group(6))))
        if rows: curves[v] = rows
    n = min(len(r) for r in curves.values()); n = min(n, n_max) if n_max else n
    res = {}
    for v, rows in curves.items():
        T = all_trajs(f"{root}/{v}")
        T = sorted(T, key=lambda p: int(p.rsplit("_", 1)[1]))[:n]; M = [json.load(open(f"{t}/meta.json")) for t in T]
        net = [abs(np.array(json.load(open(f"{t}/posture.txt")))[-1, 2] - np.array(json.load(open(f"{t}/posture.txt")))[0, 2]) for t in T]
        am = action_mix(T); last = rows[n - 1]
        res[v] = dict(episodes=n, hours=last["frames"] / 36000.0, union=last["union"], views_ge3=last["v3"], views_ge6=last["v6"], ep_cov=float(np.mean([m["coverage"] for m in M])),
                      flight=float(np.median([m["flight_m"] for m in M])), coll=float(np.mean([m["collisions"] for m in M])), net_dz1=float(np.mean(np.array(net) > 1.0)), l1=am["l1_to_official"],
                      actions=am["labels"], backward=am["labels"]["backward"])
    json.dump(res, open(out_json, "w"), indent=1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for v, rows in curves.items():
        h = [r["frames"] / 36000 for r in rows[:n]]
        for a, k in zip(ax, ("union", "v3", "v6")): a.plot(h, [r[k] for r in rows[:n]], label=v)
    for a, t in zip(ax, ("union surface coverage", "voxels seen from >=3 directions", "voxels seen from >=6 directions")): a.set_title(t, fontsize=9); a.set_xlabel("collected hours"); a.grid(alpha=.25)
    ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(out_png, dpi=130); plt.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["replay", "summary", "dry"]); ap.add_argument("--data", required=True); ap.add_argument("--scene", default="gibson_Adrian")
    ap.add_argument("--out", required=True); ap.add_argument("--stride", type=int, default=2); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--replay", nargs="*")
    a = ap.parse_args()
    if a.cmd == "dry": print(json.dumps(dry(a.data, a.out, a.out.replace(".json", ".png")), indent=1)); raise SystemExit
    if a.cmd == "replay": replay(a.data, a.scene, a.out, a.stride, a.workers, a.worker)
    else: print(json.dumps({k: v for k, v in summary(a.data, a.replay, a.out).items() if k != "curve"}, indent=1))
