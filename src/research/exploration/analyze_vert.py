import json, glob, collections, numpy as np
R = [r for f in glob.glob("results_vert/*.json") for r in json.load(open(f))]
def key(r):
    c = r["cfg"].split(":"); return (c[0].replace("frontier2nopan", "frontier v2 (no pan)").replace("frontier1", "frontier v1"), "vert" if "vert=1" in r["cfg"] else "flat")
G = collections.defaultdict(list)
for r in R: G[key(r)].append(r)
def m(rs, f): return float(np.mean([f(r) for r in rs]))
print("| policy | vertical | n | coverage | AUC-dist | coll | drift | dist | up/down % of actions | z range p5-p95 | z std | alt bins(0.5m) | nav_cov | views/100 |")
rows = {}
for k in sorted(G):
    rs = G[k]; kc = lambda r: r["kcount"]["up"] / max(1, sum(r["kcount"].values())) * 100
    def auc(r): g = np.linspace(0, 60, 121); return float(np.trapz(np.interp(g, r["dcurve"], r["curve"]), g) / 60)
    rows[k] = dict(n=len(rs), cov=m(rs, lambda r: r["coverage"]), coll=m(rs, lambda r: r["collisions"]), drift=m(rs, lambda r: r["final_drift_m"]), dist=m(rs, lambda r: r["dist"]), up=m(rs, kc), zr=m(rs, lambda r: r["div"]["z_range"]), zs=m(rs, lambda r: r["div"]["z_std"]), ab=m(rs, lambda r: r["div"]["alt_bins"]), nav=m(rs, lambda r: r["div"]["nav_cov"]), views=m(rs, lambda r: r["div"]["views_per_100"]), auc=m(rs, auc))
    v = rows[k]; print(f"| {k[0]} | {k[1]} | {v['n']} | {v['cov']:.3f} | {v['auc']:.3f} | {v['coll']:.1f} | {v['drift']:.2f} | {v['dist']:.0f} | {v['up']:.1f} | {v['zr']:.2f} | {v['zs']:.2f} | {v['ab']:.1f} | {v['nav']:.3f} | {v['views']:.1f} |")
print("paired (vert - flat), 95% bootstrap CI")
rng = np.random.default_rng(0)
for pol in sorted({k[0] for k in G}):
    a = {(r["scene"], r["seed"]): r for r in G[(pol, "vert")]}; b = {(r["scene"], r["seed"]): r for r in G[(pol, "flat")]}; ks = [k for k in a if k in b]
    out = []
    for name, f in (("coverage", lambda r: r["coverage"]), ("collisions", lambda r: r["collisions"]), ("drift", lambda r: r["final_drift_m"]), ("z_range", lambda r: r["div"]["z_range"]), ("alt_bins", lambda r: r["div"]["alt_bins"])):
        d = np.array([f(a[k]) - f(b[k]) for k in ks]); bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]; out.append(f"{name} {d.mean():+.3f} [{np.percentile(bs,2.5):+.3f},{np.percentile(bs,97.5):+.3f}]")
    print(pol, len(ks), " | ".join(out))
json.dump({f"{k[0]}|{k[1]}": v for k, v in rows.items()}, open("results_vert/summary.json", "w"), indent=1)
