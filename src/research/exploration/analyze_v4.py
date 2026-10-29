import json, glob, os, sys, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../docs/exploration/v4"); os.makedirs(OUT, exist_ok=True)
R = [r for f in glob.glob("results_v4/*.json") for r in json.load(open(f))]
def label(r):
    p, pose = r["policy"], r["pose"]
    base = {"random": "random", "frontier1": "frontier v1", "frontier1+corr": "frontier v1 + pose-corr", "frontier2nopan+corr": "frontier v2 (no pan) + corr", "frontier2": "frontier v2", "frontier2+corr": "frontier v2 + pose-corr",
            "frontiernet+corr": "FrontierNet (orig. code) + corr"}.get(p, p)
    return base + (" [GT pose]" if pose == "gt" else "")
ORDER = ["random", "frontier v1", "frontier v1 + pose-corr", "frontier v2 (no pan) + corr", "frontier v2", "frontier v2 + pose-corr", "frontier v2 + pose-corr [GT pose]", "frontier v2 [GT pose]", "FrontierNet (orig. code) + corr"]
COL = dict(zip(ORDER, ["#8a8f98", "#d98c1f", "#c0541a", "#4aa3a2", "#2b6cb0", "#1b3f8f", "#7a4fa3", "#7a4fa3", "#2f8f4e"]))
def groups(sensor):
    g = collections.defaultdict(list)
    for r in R:
        if r["sensor"] == sensor: g[label(r)].append(r)
    return {k: g[k] for k in ORDER if k in g}
def auc_dist(rs_by_scene):
    out = collections.defaultdict(list)
    for s, rs in rs_by_scene.items():
        dmax = max(r["dcurve"][-1] for rs_ in [rs] for r in rs_)
        grid = np.linspace(0, dmax, 300)
        for r in rs:
            d = np.array(r["dcurve"]); c = np.array(r["curve"]); ci = np.interp(grid, d, c)  # hold final coverage after terminal distance (OPAL protocol)
            out[label(r)].append(np.trapz(ci, grid) / dmax)
    return out
def stats(sensor):
    G = groups(sensor); S = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in R:
        if r["sensor"] == sensor: S[r["scene"]]["all"].append(r)
    A = auc_dist({s: v["all"] for s, v in S.items()})
    T = {}
    for k, rs in G.items():
        f = lambda fn: float(np.mean([fn(r) for r in rs])); dv = lambda key: f(lambda r: r["div"][key])
        det = {x: sum(r["det"][x] for r in rs) for x in ("tp", "fp", "fn", "tn")}
        T[k] = dict(n=len(rs), cov=f(lambda r: r["coverage"]), cov_sd=float(np.std([r["coverage"] for r in rs])), auc=float(np.mean(A[k])), coll=f(lambda r: r["collisions"]), drift=f(lambda r: r["final_drift_m"]),
                    dist=f(lambda r: r["dist"]), actions=f(lambda r: r["actions"]), pans=f(lambda r: r["pans"]), nav_cov=dv("nav_cov"), revisit=dv("revisit"), yaw_ent=dv("yaw_entropy"), views=dv("views_per_100"), frame_nn=dv("frame_nn"),
                    ms=f(lambda r: r["ms_per_step"]), det=det)
    return T
def curves(sensor, fn):
    G = groups(sensor); fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, rs in G.items():
        xs = np.linspace(0, 1000, 101); ys = np.array([np.interp(xs, r["acurve"], r["curve"]) for r in rs]); ax[0].plot(xs, ys.mean(0), label=k, color=COL[k], lw=1.8)
        # coverage vs distance, hold final value
        dm = 80; g = np.linspace(0, dm, 161); ys = np.array([np.interp(g, r["dcurve"], r["curve"]) for r in rs]); ax[1].plot(g, ys.mean(0), color=COL[k], lw=1.8)
    ax[0].set_xlabel("executed actions"); ax[0].set_ylabel("surface-voxel coverage"); ax[1].set_xlabel("distance flown (m)"); ax[1].set_ylabel("coverage")
    ax[0].set_title(f"Coverage vs actions ({sensor})"); ax[1].set_title("Coverage vs distance"); ax[0].legend(fontsize=7, loc="lower right"); [a.grid(alpha=.25) for a in ax]
    plt.tight_layout(); plt.savefig(f"{OUT}/{fn}", dpi=140); plt.close()
def bars(T, fn, keys, titles, sensor):
    n = len(keys); fig, ax = plt.subplots(1, n, figsize=(3.1 * n, 3.6)); ax = np.atleast_1d(ax)
    for a, k, t in zip(ax, keys, titles):
        names = list(T); v = [T[m][k] for m in names]; a.barh(range(len(names)), v, color=[COL[m] for m in names]); a.set_title(t, fontsize=9)
        a.set_yticks(range(len(names))); a.set_yticklabels(names if a is ax[0] else [], fontsize=6.5); a.invert_yaxis(); a.grid(axis="x", alpha=.25)
        for i, x in enumerate(v): a.text(x, i, f" {x:.2f}" if x < 100 else f" {x:.0f}", va="center", fontsize=6.5)
    plt.tight_layout(); plt.savefig(f"{OUT}/{fn}", dpi=140); plt.close()
def traj_fig(scene, methods, fn, seed=0):
    geo = np.load(f"results_v4/geo/{scene}.npz"); pts = geo["pts"]; lo = geo["bounds"][0][[0, 2]]; hi = geo["bounds"][1][[0, 2]]
    fig, ax = plt.subplots(1, len(methods), figsize=(3.3 * len(methods), 3.6))
    for a, m in zip(np.atleast_1d(ax), methods):
        rs = [r for r in R if r["scene"] == scene and r["sensor"] == "rgbd" and label(r) == m and r["seed"] == seed]
        a.scatter(pts[:, 0], pts[:, 2], s=.15, c="#c9cdd3", rasterized=True)
        if rs:
            T = np.array(rs[0]["traj"]); a.scatter(T[:, 0], T[:, 2], c=np.arange(len(T)), s=3, cmap="plasma"); a.scatter(T[0, 0], T[0, 2], c="k", s=25, marker="*", zorder=5)
            a.set_title(f"{m}\ncov {rs[0]['coverage']:.2f}, {rs[0]['dist']:.0f} m", fontsize=8)
        a.set_xlim(lo[0] - 1, hi[0] + 1); a.set_ylim(hi[1] + 1, lo[1] - 1); a.set_aspect("equal"); a.axis("off")
    fig.suptitle(scene, fontsize=9); plt.tight_layout(); plt.savefig(f"{OUT}/{fn}", dpi=140); plt.close()
if __name__ == "__main__":
    res = {}
    for sensor in ("rgbd", "rgb"):
        if not any(r["sensor"] == sensor for r in R): continue
        T = stats(sensor); res[sensor] = T; curves(sensor, f"curves_{sensor}.png")
        bars(T, f"summary_{sensor}.png", ["cov", "auc", "coll", "drift"], ["final coverage", "coverage-vs-distance AUC", "collisions", "final pose error (m)"], sensor)
        bars(T, f"diversity_{sensor}.png", ["nav_cov", "revisit", "yaw_ent", "views", "frame_nn"], ["navigable-area visited", "revisit ratio (lower=less redundant)", "yaw entropy (norm.)", "distinct views /100 steps", "frame novelty (NN dist.)"], sensor)
    if "rgbd" in res:
        top = [m for m in ("random", "frontier v1", "frontier v2 + pose-corr", "FrontierNet (orig. code) + corr") if m in res["rgbd"]]
        for s in ("replica_apartment_0", "mp3d_17DRP5sb8fy", "hm3d_12e6joG2B2T", "gibson_Anaheim"): traj_fig(s, top, f"traj_{s}.png")
    json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
    for sensor, T in res.items():
        print("==", sensor)
        for k, v in T.items(): print(f"{k:34s} n={v['n']:2d} cov={v['cov']:.3f} auc={v['auc']:.3f} coll={v['coll']:.1f} drift={v['drift']:.2f} dist={v['dist']:.0f} act={v['actions']:.0f} pans={v['pans']:.1f} nav={v['nav_cov']:.3f} rev={v['revisit']:.3f} yawH={v['yaw_ent']:.3f} views={v['views']:.1f} nn={v['frame_nn']:.4f} det={v['det']}")

def paired(sensor="rgbd", pairs=None, metrics=("coverage", "auc", "final_drift_m", "collisions", "nav_cov", "views")):
    idx = collections.defaultdict(dict)
    G = collections.defaultdict(list)
    for r in R:
        if r["sensor"] == sensor: G[r["scene"]].append(r)
    # per-run auc with per-scene dmax
    auc = {}
    for s, rs in G.items():
        dmax = max(r["dcurve"][-1] for r in rs); grid = np.linspace(0, dmax, 300)
        for r in rs: auc[(s, label(r), r["seed"])] = float(np.trapz(np.interp(grid, r["dcurve"], r["curve"]), grid) / dmax)
    def val(r, m):
        if m == "auc": return auc[(r["scene"], label(r), r["seed"])]
        if m in ("nav_cov",): return r["div"]["nav_cov"]
        if m == "views": return r["div"]["views_per_100"]
        return r[m]
    runs = {(r["scene"], r["seed"], label(r)): r for r in R if r["sensor"] == sensor}
    out = {}; rng = np.random.default_rng(0)
    for name, (a, b) in pairs.items():
        keys = [(s, sd) for (s, sd, l) in runs if l == a and (s, sd, b) in runs]
        row = {}
        for m in metrics:
            d = np.array([val(runs[(s, sd, a)], m) - val(runs[(s, sd, b)], m) for s, sd in keys])
            bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
            row[m] = (float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
        out[name] = dict(n=len(keys), **row)
    return out

PAIRS = {"v2+corr  vs  v1 (full upgrade)": ("frontier v2 + pose-corr", "frontier v1"),
         "info-gain scoring only (v2 no-pan+corr vs v1+corr)": ("frontier v2 (no pan) + corr", "frontier v1 + pose-corr"),
         "junction pan (v2+corr vs v2 no-pan+corr)": ("frontier v2 + pose-corr", "frontier v2 (no pan) + corr"),
         "pose-corr on v1": ("frontier v1 + pose-corr", "frontier v1"),
         "pose-corr on v2": ("frontier v2 + pose-corr", "frontier v2"),
         "GT pose upper bound (v2 GT vs v2+corr)": ("frontier v2 [GT pose]", "frontier v2 + pose-corr"),
         "v2+corr vs random": ("frontier v2 + pose-corr", "random")}
if __name__ == "__main__":
    P = paired("rgbd", PAIRS); json.dump(P, open(f"{OUT}/paired_rgbd.json", "w"), indent=1)
    for k, v in P.items(): print(k, "n=", v["n"], {m: f"{v[m][0]:+.3f} [{v[m][1]:+.3f},{v[m][2]:+.3f}]" for m in ("coverage", "auc", "final_drift_m", "collisions", "nav_cov")})
