"""3D heatmap data over all collected trajectories of one scene (evaluation only; uses simulator poses and GT geometry).

  python heatmap_v5.py --tag v5 --data ../../../data/collected_10h_v5 --replay ../exploration_v5/replay/v5_*.npz --gallery ../exploration_v5/gallery
Every collection version is archived under docs/exploration/heatmaps/<tag>/ (heatmap3d.json, meta.json, heatmap_topdown.png) and an
existing tag is never overwritten (use a new tag, or --force for a deliberate rebuild). With --gallery the JSON is also copied
next to the gallery page. heatmap3d.json holds: GT surface voxels (0.25 m) with the number of trajectories that saw each one and its distinct
viewing directions (of 24), 0.25 m cells of camera positions with frames spent and trajectories passing, and the paths of the
gallery trajectories. Coordinates are habitat metres (y up).
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess
import numpy as np
from analyze_collection import all_trajs, merge, popcount

RES, CAM_H = 0.25, 1.5


def cam_xyz(Q): return np.stack([Q[:, 0], Q[:, 2] + CAM_H, Q[:, 1]], 1)  # posture = (x, habitat z, base height, yaw)


def compute(data, replay, gallery, scene="gibson_Adrian"):
    from uav_sim import UAVSim
    from scenes import paths, Geometry
    g, n = paths(scene); S = UAVSim(g); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds())
    V = np.array(sorted(G.gt_vox), np.int64); seen, bits, _, _ = merge(replay); assert len(seen) == len(V); dirs = popcount(bits)
    T = all_trajs(data); P = {os.path.relpath(t, data): np.array(json.load(open(f"{t}/posture.txt")), float) for t in T}
    cells = {}
    for Q in P.values():
        c = np.floor(cam_xyz(Q) / RES).astype(int); u, k = np.unique(c, axis=0, return_counts=True)
        for q, m in zip(map(tuple, u), k): f, t = cells.get(q, (0, 0)); cells[q] = (f + int(m), t + 1)
    C = np.array([[*q, f, t] for q, (f, t) in cells.items()], np.int64)
    paths_g = {}
    man = json.load(open(f"{gallery}/manifest.json")) if gallery and os.path.exists(f"{gallery}/manifest.json") else {"items": []}
    for it in man["items"]:
        w, tr = it["id"].split("/"); Q = P[f"{w}/gibson/Adrian/{tr}"]; X = cam_xyz(Q)[::3]
        paths_g[it["name"]] = np.round(X, 2).reshape(-1).tolist()
    return dict(source=f"heatmap_v5.py from {os.path.relpath(data)} + pose replay ({len(T)} trajectories); evaluation-side view, simulator poses",
                scene=scene, res=RES, n_traj=len(T), n_frames=int(sum(len(q) for q in P.values())), n_vox=int(len(V)),
                union=round(float((seen > 0).mean()), 4), ge3=round(float((dirs >= 3).mean()), 4),
                vox=np.c_[V, seen, dirs].reshape(-1).tolist(), cells=C.reshape(-1).tolist(), paths=paths_g)


def topdown_png(D, fn):
    """Static top-down summary of one version: per (x, z) column the mean trajectories-that-saw, mean viewing directions, and camera frames."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; from matplotlib.colors import LinearSegmentedColormap
    blue = LinearSegmentedColormap.from_list("b", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"]); orange = LinearSegmentedColormap.from_list("o", ["#fde4d6", "#f4a682", "#eb6834", "#ad4214", "#652509"])
    v = np.array(D["vox"]).reshape(-1, 5); c = np.array(D["cells"]).reshape(-1, 5); lo = np.minimum(v[:, [0, 2]].min(0), c[:, [0, 2]].min(0)); hi = np.maximum(v[:, [0, 2]].max(0), c[:, [0, 2]].max(0)) + 1
    def col(x, z, w, mean=True):
        S = np.zeros(hi - lo); N = np.zeros(hi - lo); np.add.at(S, (x - lo[0], z - lo[1]), w); np.add.at(N, (x - lo[0], z - lo[1]), 1)
        return np.where(N > 0, S / np.maximum(N, 1) if mean else S, np.nan).T
    panels = [(col(v[:, 0], v[:, 2], v[:, 3]), blue, "surface: trajectories that saw it (column mean)"), (col(v[:, 0], v[:, 2], v[:, 4]), blue, "surface: viewing directions of 24 (column mean)"),
              (np.log10(col(c[:, 0], c[:, 2], c[:, 3], False)), orange, "camera: log10 frames spent (column sum)")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6)); ext = [lo[0] * D["res"], hi[0] * D["res"], lo[1] * D["res"], hi[1] * D["res"]]
    for a, (m, cm, t) in zip(ax, panels):
        im = a.imshow(m, origin="lower", cmap=cm, extent=ext, aspect="equal"); a.set_title(t, fontsize=9); plt.colorbar(im, ax=a, fraction=.035); a.set_xlabel("x (m)"); a.set_ylabel("z (m)")
    fig.suptitle(f"{D['tag']}: {D['n_traj']} trajectories, {D['n_frames']} frames, union {100 * D['union']:.1f}%, >=3 directions {100 * D['ge3']:.1f}%  ({D['scene']})", fontsize=10)
    plt.tight_layout(); plt.savefig(fn, dpi=110); plt.close()


def sha(fn): return hashlib.sha256(open(fn, "rb").read()).hexdigest()


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True, help="collection version, e.g. v4, v5"); ap.add_argument("--data", required=True)
    ap.add_argument("--replay", nargs="+", required=True); ap.add_argument("--gallery", default=None); ap.add_argument("--scene", default="gibson_Adrian")
    ap.add_argument("--archive", default=os.path.join(here, "../../../docs/exploration/heatmaps")); ap.add_argument("--force", action="store_true")
    a = ap.parse_args(); out = os.path.join(a.archive, a.tag)
    if os.path.exists(f"{out}/heatmap3d.json") and not a.force: raise SystemExit(f"{out} already archived; pick a new --tag or pass --force")
    os.makedirs(out, exist_ok=True); D = compute(a.data, a.replay, a.gallery, a.scene); D["tag"] = a.tag
    json.dump(D, open(f"{out}/heatmap3d.json", "w"), separators=(",", ":")); topdown_png(D, f"{out}/heatmap_topdown.png")
    try: commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=here).stdout.strip()
    except Exception: commit = ""
    sample = next((os.path.join(r, "meta.json") for r, _, f in os.walk(a.data) if "meta.json" in f), None)
    meta = dict(tag=a.tag, scene=a.scene, generated_by="src/research/exploration/heatmap_v5.py", created=datetime.datetime.now().isoformat(timespec="seconds"), git_head=commit,
                data=os.path.relpath(os.path.abspath(a.data), os.path.abspath(os.path.join(here, "../../.."))), replay={os.path.basename(f): sha(f) for f in a.replay},
                policy=json.load(open(sample)).get("policy") if sample else None, n_traj=D["n_traj"], n_frames=D["n_frames"], union=D["union"], ge3=D["ge3"],
                n_unseen=int((np.array(D["vox"]).reshape(-1, 5)[:, 3] == 0).sum()), heatmap_sha256=sha(f"{out}/heatmap3d.json"))
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1, ensure_ascii=False)
    if a.gallery: shutil.copy(f"{out}/heatmap3d.json", f"{a.gallery}/heatmap3d.json")
    print(out, meta["n_traj"], "traj", "union", meta["union"], "ge3", meta["ge3"], "unseen", meta["n_unseen"])
