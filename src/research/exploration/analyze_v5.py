"""Summaries, paired bootstrap comparisons and figures for the v5 exploration ablation (explore5.py output).

  python analyze_v5.py --res ../exploration_v5/ablation --out ../../../docs/exploration/v5 --tag ablation
"""
import argparse, collections, glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt


def load(ds):
    ds = [ds] if isinstance(ds, str) else ds
    return [r for d in ds for f in sorted(glob.glob(f"{d}/*.json")) for r in json.load(open(f))]


def at(r, xs, x): return float(np.interp(x, r[xs], r["curve"]))  # holds the final value past the run's end


def auc_dist(r, dmax=100.0):
    g = np.linspace(0, dmax, 201); return float(np.trapz(np.interp(g, r["dcurve"], r["curve"]), g) / dmax)


def flat_last_1000(r):  # v4 stress-test deadlock definition
    a = np.array(r["acurve"]); c = np.array(r["curve"]); prev = np.interp(max(0, r["actions"] - 1000), a, c)
    return r["actions"] >= 1500 and c[-1] - prev < 0.01 and c[-1] < 0.9


def stalled(r):  # protocol-independent: < 1 % coverage gained over the last third of the run while below 90 %
    a = np.array(r["acurve"]); c = np.array(r["curve"]); prev = np.interp(2 * r["actions"] / 3, a, c)
    return c[-1] - prev < 0.01 and c[-1] < 0.9 and r["dist"] < 99


METRICS = {
    "cov_final": lambda r: r["coverage"], "cov@1000a": lambda r: at(r, "acurve", 1000), "cov@20m": lambda r: at(r, "dcurve", 20), "cov@50m": lambda r: at(r, "dcurve", 50),
    "auc_0_100m": auc_dist, "flat1000": lambda r: float(flat_last_1000(r)), "stalled": lambda r: float(stalled(r)), "reach90": lambda r: float(r["coverage"] >= 0.9),
    "collisions": lambda r: r["collisions"], "coll_real": lambda r: r["collisions"] - r.get("n_fence", 0), "fence": lambda r: r.get("n_fence", 0), "drift_m": lambda r: r["final_drift_m"], "yaw_err": lambda r: r.get("final_yaw_err", np.nan), "dist_m": lambda r: r["dist"],
    "actions": lambda r: r["actions"], "views_ge3": lambda r: r["views"].get("views_ge3", np.nan), "views_ge1": lambda r: r["views"].get("views_ge1", np.nan),
    "nav_cov": lambda r: r["div"]["nav_cov"], "turn_frac": lambda r: r["kcount"]["yaw"] / max(1, sum(r["kcount"].values())), "escapes": lambda r: r["escapes"],
    "plan_ms": lambda r: r.get("plan_ms", np.nan), "wall_ms_per_act": lambda r: r["ms_per_step"], "z_range": lambda r: r["div"]["z_range"],
}


def table(R, order=None):
    G = collections.defaultdict(list)
    for r in R: G[r["name"]].append(r)
    names = order or list(G)
    out = {}
    for k in names:
        if k not in G: continue
        rs = G[k]; out[k] = dict(n=len(rs), **{m: (float(np.nanmean([f(r) for r in rs])), float(np.nanstd([f(r) for r in rs]) / np.sqrt(len(rs)))) for m, f in METRICS.items()})
        for m in ("drift_m", "coll_real", "fence", "yaw_err"): out[k][m + "_median"] = float(np.nanmedian([METRICS[m](r) for r in rs]))
    return out, G


def paired(G, a, b, metric, n_boot=4000, seed=0):
    ka = {(r["scene"], r["seed"]): METRICS[metric](r) for r in G[a]}; kb = {(r["scene"], r["seed"]): METRICS[metric](r) for r in G[b]}
    ks = sorted(k for k in ka if k in kb); d = np.array([ka[k] - kb[k] for k in ks])
    if len(d) == 0: return None
    rng = np.random.default_rng(seed); bs = rng.choice(d, (n_boot, len(d))).mean(1)
    return dict(n=len(d), mean=float(d.mean()), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)), wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def per_scene(G, names, metric="cov_final"):
    sc = sorted({r["scene"] for rs in G.values() for r in rs}); out = {}
    for k in names:
        if k not in G: continue
        out[k] = {s: float(np.mean([METRICS[metric](r) for r in G[k] if r["scene"] == s])) for s in sc}
    return out


def curves(G, names, fn, title, dmax=100, amax=3000, colors=None):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for i, k in enumerate(names):
        if k not in G: continue
        rs = G[k]; c = (colors or {}).get(k)
        g = np.linspace(0, dmax, 201); ax[0].plot(g, np.mean([np.interp(g, r["dcurve"], r["curve"]) for r in rs], 0), label=k, color=c, lw=1.8)
        ga = np.linspace(0, amax, 151); ax[1].plot(ga, np.mean([np.interp(ga, r["acurve"], r["curve"]) for r in rs], 0), color=c, lw=1.8)
    ax[0].set_xlabel("distance flown (m)"); ax[1].set_xlabel("executed actions"); [a.set_ylabel("surface-voxel coverage") for a in ax]; [a.grid(alpha=.25) for a in ax]
    ax[0].legend(fontsize=7, loc="lower right"); fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.savefig(fn, dpi=130); plt.close()


def scene_bars(G, names, fn, metric="cov_final"):
    sc = sorted({r["scene"] for k in names for r in G[k]}); w = 0.8 / max(1, len(names)); fig, ax = plt.subplots(figsize=(13, 4))
    for i, k in enumerate(names):
        v = [np.mean([METRICS[metric](r) for r in G[k] if r["scene"] == s] or [np.nan]) for s in sc]; ax.bar(np.arange(len(sc)) + i * w, v, w, label=k)
    ax.set_xticks(np.arange(len(sc)) + 0.4 - w / 2); ax.set_xticklabels([s.replace("_apartment", "").replace("replica_frl", "replica_frl") for s in sc], fontsize=8, rotation=15)
    ax.set_ylabel("final coverage (3 seeds)"); ax.legend(fontsize=7, ncol=4); ax.grid(axis="y", alpha=.25); plt.tight_layout(); plt.savefig(fn, dpi=130); plt.close()


def drift_curves(G, names, fn, amax=3000):
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8)); ga = np.linspace(0, amax, 61)
    for k in names:
        rs = [r for r in G[k] if r.get("drift")]
        if not rs: continue
        ax[0].plot(ga, np.median([np.interp(ga, r["acurve"][:len(r["drift"])], r["drift"]) for r in rs], 0), label=k)
        ys = [r for r in rs if r.get("yaw_err")]
        if ys: ax[1].plot(ga, np.median([np.interp(ga, r["acurve"][:len(r["yaw_err"])], r["yaw_err"]) for r in ys], 0), label=k)
    ax[0].set_ylabel("median position error (m)"); ax[1].set_ylabel("median |heading error| (deg)"); [a.set_xlabel("executed actions") for a in ax]; [a.grid(alpha=.25) for a in ax]
    ax[0].legend(fontsize=7); ax[0].set_yscale("symlog", linthresh=1); plt.tight_layout(); plt.savefig(fn, dpi=130); plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True, nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--base", default="v1"); ap.add_argument("--pairs", default=""); ap.add_argument("--curves", default=""); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = load(a.res); T, G = table(R)
    pairs = [p.split(">") for p in a.pairs.split(";") if p] or [(k, a.base) for k in G if k != a.base]
    P = {f"{x} vs {y}": {m: paired(G, x, y, m) for m in ("cov_final", "cov@1000a", "auc_0_100m", "stalled", "coll_real", "drift_m", "views_ge3")} for x, y in pairs if x in G and y in G}
    S = per_scene(G, list(G))
    json.dump(dict(table=T, paired=P, per_scene=S), open(f"{a.out}/{a.tag}.json", "w"), indent=1)
    keep = [k for k in (a.curves.split(";") if a.curves else list(G)) if k in G]
    curves(G, keep, f"{a.out}/{a.tag}_curves.png", a.tag)
    scene_bars(G, keep, f"{a.out}/{a.tag}_scenes.png"); drift_curves(G, [k for k in keep if k not in ("random",)], f"{a.out}/{a.tag}_drift.png")
    for k, v in T.items():
        print(f"{k:28s} n={v['n']:2d} " + " ".join(f"{m}={v[m][0]:.3f}" for m in ("cov_final", "cov@1000a", "auc_0_100m", "stalled", "flat1000", "coll_real", "fence", "drift_m", "yaw_err", "views_ge3", "dist_m", "turn_frac", "plan_ms")))
    for k, v in P.items(): print(k, {m: f"{x['mean']:+.3f} [{x['lo']:+.3f},{x['hi']:+.3f}] {x['wins']}/{x['losses']}" for m, x in v.items() if x})
