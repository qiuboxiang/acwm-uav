"""Qualitative figures comparing two collected datasets on the same scene (evaluation only).

  python viz_collection.py --a ../../../data/collected_10h --la "v4" --b ../../../data/collected_10h_v5 --lb "v5" \
      --ra ../exploration_v5/replay/v4_*.npz --rb ../exploration_v5/replay/v5_*.npz --out ../../../docs/exploration/v5
"""
import argparse, glob, json, os
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_collection import all_trajs, merge, popcount


def frames_of(tp, idx, size=(224, 126)):
    cap = cv2.VideoCapture(f"{tp}/video.mp4"); out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        out.append(cv2.resize(f, size, interpolation=cv2.INTER_AREA) if ok else np.zeros((size[1], size[0], 3), np.uint8))
    cap.release(); return out


def contact_sheet(T, fn, label, rows=8, cols=6, seed=0):
    rng = np.random.default_rng(seed); pick = rng.choice(len(T), rows, replace=False); tiles = []
    for k in pick:
        n = len(json.load(open(f"{T[k]}/posture.txt"))); fr = frames_of(T[k], np.linspace(0, n - 1, cols).astype(int))
        row = np.concatenate(fr, 1); cv2.putText(row, os.path.basename(T[k]), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1); tiles.append(row)
    img = np.concatenate(tiles, 0); cv2.putText(img, label, (4, img.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imwrite(fn, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def poses(T): return [np.array(json.load(open(f"{t}/posture.txt")), float) for t in T]


def topdown(geo, PA, PB, la, lb, fn, n_show=6, seed=1):
    pts = geo["pts"]; lo = geo["bounds"][0][[0, 2]]; hi = geo["bounds"][1][[0, 2]]
    fig, ax = plt.subplots(2, 3, figsize=(15, 9.5)); rng = np.random.default_rng(seed)
    for r, (P, lab) in enumerate(((PA, la), (PB, lb))):
        a = ax[r, 0]; a.scatter(pts[::3, 0], pts[::3, 2], s=.05, c="#c9cdd3", rasterized=True)
        for k in rng.choice(len(P), n_show, replace=False):
            Q = P[k]; a.plot(Q[:, 0], Q[:, 1], lw=.8); a.scatter(Q[0, 0], Q[0, 1], marker="*", s=40, c="k", zorder=5)
        a.set_title(f"{lab}: {n_show} random trajectories (star = start)", fontsize=9)
        allp = np.concatenate(P); H, xe, ye = np.histogram2d(allp[:, 0], allp[:, 1], bins=[np.arange(lo[0], hi[0] + .25, .25), np.arange(lo[1], hi[1] + .25, .25)])
        a = ax[r, 1]; a.scatter(pts[::3, 0], pts[::3, 2], s=.05, c="#c9cdd3", rasterized=True)
        im = a.imshow(np.log10(1 + H.T), origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="magma", alpha=.85, aspect="equal"); plt.colorbar(im, ax=a, fraction=.04, label="log10(1 + frames)")
        a.set_title(f"{lab}: frame density over all trajectories ({(H > 0).sum()} cells of 0.25 m visited)", fontsize=9)
        a = ax[r, 2]; S0 = np.array([q[0] for q in P]); a.scatter(pts[::3, 0], pts[::3, 2], s=.05, c="#c9cdd3", rasterized=True)
        a.scatter(S0[:, 0], S0[:, 1], s=6, c="#2b6cb0"); a.set_title(f"{lab}: start positions ({len(S0)} trajectories)", fontsize=9)
    for a in ax.ravel(): a.set_xlim(lo[0] - .5, hi[0] + .5); a.set_ylim(lo[1] - .5, hi[1] + .5); a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(fn, dpi=110); plt.close()


def view_map(geo_vox, RA, RB, la, lb, fn):
    """Top-down: for every ground-truth surface voxel column, the mean number of distinct viewing directions (of 24) observed."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2)); V = np.array(geo_vox)
    lo = V[:, [0, 2]].min(0); hi = V[:, [0, 2]].max(0) + 1; maps = []
    for R in (RA, RB):
        _, bits, _, _ = merge(R); pc = popcount(bits).astype(float)
        S = np.zeros(hi - lo); N = np.zeros(hi - lo); np.add.at(S, (V[:, 0] - lo[0], V[:, 2] - lo[1]), pc); np.add.at(N, (V[:, 0] - lo[0], V[:, 2] - lo[1]), 1)
        maps.append(np.where(N > 0, S / np.maximum(N, 1), np.nan))
    vmax = np.nanpercentile(np.concatenate([m[np.isfinite(m)] for m in maps]), 98)
    for a, m, lab in zip(ax[:2], maps, (la, lb)):
        im = a.imshow(m.T, origin="lower", cmap="viridis", vmin=0, vmax=vmax); a.set_title(f"{lab}: mean # viewing directions per surface voxel", fontsize=9); plt.colorbar(im, ax=a, fraction=.04)
    d = maps[1] - maps[0]; lim = np.nanpercentile(np.abs(d[np.isfinite(d)]), 98)
    im = ax[2].imshow(d.T, origin="lower", cmap="RdBu", vmin=-lim, vmax=lim); ax[2].set_title(f"{lb} minus {la}", fontsize=9); plt.colorbar(im, ax=ax[2], fraction=.04)
    for a in ax: a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(fn, dpi=110); plt.close()


def hists(PA, PB, la, lb, fn):
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8)); A = np.concatenate(PA); B = np.concatenate(PB)
    bins = np.linspace(min(A[:, 2].min(), B[:, 2].min()), max(A[:, 2].max(), B[:, 2].max()), 40)
    ax[0].hist(A[:, 2], bins, alpha=.6, density=True, label=la); ax[0].hist(B[:, 2], bins, alpha=.6, density=True, label=lb); ax[0].set_xlabel("base height z (m)"); ax[0].legend()
    yb = np.arange(0, 361, 15); ax[1].hist(A[:, 3] % 360, yb, alpha=.6, density=True, label=la); ax[1].hist(B[:, 3] % 360, yb, alpha=.6, density=True, label=lb); ax[1].set_xlabel("heading (deg)")
    za = [np.percentile(p[:, 2], 95) - np.percentile(p[:, 2], 5) for p in PA]; zb = [np.percentile(p[:, 2], 95) - np.percentile(p[:, 2], 5) for p in PB]
    b2 = np.linspace(0, max(max(za), max(zb)) + 1e-3, 30); ax[2].hist(za, b2, alpha=.6, label=la); ax[2].hist(zb, b2, alpha=.6, label=lb); ax[2].set_xlabel("per-trajectory altitude range p5-p95 (m)")
    for a in ax: a.grid(alpha=.25)
    plt.tight_layout(); plt.savefig(fn, dpi=120); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--la", default="v4"); ap.add_argument("--lb", default="v5")
    ap.add_argument("--ra", nargs="*"); ap.add_argument("--rb", nargs="*"); ap.add_argument("--out", required=True); ap.add_argument("--scene", default="gibson_Adrian")
    ap.add_argument("--geo", default="results_v4/geo/gibson_Adrian.npz"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    TA, TB = all_trajs(a.a), all_trajs(a.b); PA, PB = poses(TA), poses(TB); geo = np.load(a.geo)
    contact_sheet(TA, f"{a.out}/contact_{a.la}.jpg", a.la); contact_sheet(TB, f"{a.out}/contact_{a.lb}.jpg", a.lb)
    topdown(geo, PA, PB, a.la, a.lb, f"{a.out}/topdown.png"); hists(PA, PB, a.la, a.lb, f"{a.out}/pose_hists.png")
    if a.ra and a.rb:
        from uav_sim import UAVSim; from scenes import paths, Geometry
        g, n = paths(a.scene); S = UAVSim(g); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds())
        view_map(sorted(G.gt_vox), a.ra, a.rb, a.la, a.lb, f"{a.out}/view_map.png")
    print("ok")
