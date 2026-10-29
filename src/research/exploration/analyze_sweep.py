import json, glob, os, collections, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../docs/exploration/sweep"); os.makedirs(OUT, exist_ok=True)
R = [r for f in glob.glob("results_sweep/*.json") for r in json.load(open(f))]
def name(r):
    c = r["cfg"].split(":"); pol = c[0]; kw = dict(x.split("=") for x in c[3:])
    if pol == "frontier1": return "v1 (nearest)"
    s = f"λ={kw.get('lam', '0.04')}"
    if "mu" in kw: s += f", turn-cost {kw['mu']}"
    return s + (f", pan≥{kw['pan']} m" if pol == "frontier2" else ", no pan")
def at(r, xs, ys, x): return float(np.interp(x, r[xs], r[ys]))  # holds the last value past the run's end
def at_dist(r, d): return at(r, "dcurve", "curve", d)
def at_act(r, a): return at(r, "acurve", "curve", a)
def at_time(r, t): return at(r, "tcurve", "curve", t)
def auc60(r): g = np.linspace(0, 60, 241); return float(np.trapz(np.interp(g, r["dcurve"], r["curve"]), g) / 60)
G = collections.defaultdict(list)
for r in R: G[name(r)].append(r)
order = ["v1 (nearest)", "λ=0, no pan", "λ=0.02, no pan", "λ=0.04, no pan", "λ=0.1, no pan", "λ=0.3, no pan", "λ=0.04, turn-cost 0.3, no pan", "λ=0.04, pan≥3 m", "λ=0.04, pan≥6 m", "λ=0.04, pan≥10 m", "λ=0.04, turn-cost 0.3, pan≥6 m"]
order = [k for k in order if k in G]
def stat(rs, f): v = np.array([f(r) for r in rs]); return float(v.mean()), float(v.std() / np.sqrt(len(v)))
rows = {}
for k in order:
    rs = G[k]; yaw = np.mean([r["kcount"]["yaw"] / max(1, sum(r["kcount"].values())) for r in rs])
    rows[k] = dict(n=len(rs), cov20=stat(rs, lambda r: at_dist(r, 20)), cov40=stat(rs, lambda r: at_dist(r, 40)), cov60=stat(rs, lambda r: at_dist(r, 60)), auc=stat(rs, auc60),
                   cov1000a=stat(rs, lambda r: at_act(r, 1000)), cov300s=stat(rs, lambda r: at_time(r, 300)), actions=float(np.mean([r["actions"] for r in rs])), dist=float(np.mean([r["dist"] for r in rs])),
                   t_flight=float(np.mean([r["t_flight"] for r in rs])), yaw_frac=float(yaw), m_per_act=float(np.mean([r["dist"] / max(1, r["actions"]) for r in rs])), nplan100=float(np.mean([100 * r["nplan"] / max(1, r["actions"]) for r in rs])),
                   pans=float(np.mean([r["pans"] for r in rs])), coll=float(np.mean([r["collisions"] for r in rs])), reach60=float(np.mean([r["dist"] >= 59.5 for r in rs])))
def paired(a, b, f, seed=0):
    ka = {(r["scene"], r["seed"]): f(r) for r in G[a]}; kb = {(r["scene"], r["seed"]): f(r) for r in G[b]}; ks = [k for k in ka if k in kb]; d = np.array([ka[k] - kb[k] for k in ks]); rng = np.random.default_rng(seed)
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]; return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(ks)
ref = "v1 (nearest)"; base = "λ=0.04, no pan"
P = {}
for k in order:
    if k == ref: continue
    P[k] = dict(vs_v1_dist40=paired(k, ref, lambda r: at_dist(r, 40)), vs_v1_auc=paired(k, ref, auc60), vs_v1_act1000=paired(k, ref, lambda r: at_act(r, 1000)), vs_v1_time300=paired(k, ref, lambda r: at_time(r, 300)))
    if k != base and base in G: P[k]["vs_nopan_dist40"] = paired(k, base, lambda r: at_dist(r, 40)); P[k]["vs_nopan_act1000"] = paired(k, base, lambda r: at_act(r, 1000))
json.dump(dict(rows=rows, paired=P), open(f"{OUT}/sweep.json", "w"), indent=1)
col = {k: c for k, c in zip(order, ["#d98c1f", "#9aa8b8", "#7f9bc4", "#2b6cb0", "#1b3f8f", "#0f2a63", "#4aa3a2", "#c9625d", "#a33a3a", "#6e1f1f", "#7a4fa3"])}
# fig1: coverage at 40 m vs lambda (+ pan variants)
lam = [(float(k.split("λ=")[1].split(",")[0]), k) for k in order if k.endswith("no pan") and "turn-cost" not in k]
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for a, key, ttl in zip(ax, ("cov40", "cov1000a", "cov300s"), ("coverage at 40 m flown", "coverage at 1000 actions", "coverage at 300 s flight time")):
    xs = [l for l, _ in lam]; m = [rows[k][key][0] for _, k in lam]; e = [rows[k][key][1] for _, k in lam]
    a.errorbar(range(len(xs)), m, yerr=e, marker="o", color="#2b6cb0", label="no pan, vary λ", capsize=3)
    a.set_xticks(range(len(xs))); a.set_xticklabels([f"{x:g}" for x in xs]); a.set_xlabel("distance weight λ"); a.set_title(ttl, fontsize=10); a.grid(alpha=.25)
    if ref in rows: a.axhline(rows[ref][key][0], color="#d98c1f", ls="--", label="v1 (nearest)")
    for k in order:
        if "pan≥" in k and "turn-cost" not in k: a.errorbar([len(xs) - 1 + 0.5 + 0.25 * ["3", "6", "10"].index(k.split("≥")[1].split(" ")[0])], [rows[k][key][0]], yerr=[rows[k][key][1]], marker="s", color=col[k], ls="", capsize=3, label=k.split(", ")[1])
    a.legend(fontsize=6.5)
plt.tight_layout(); plt.savefig(f"{OUT}/lambda_pan.png", dpi=140); plt.close()
# fig2: curves vs distance and vs time
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9)); pick = [k for k in (ref, base, "λ=0.1, no pan", "λ=0.04, pan≥6 m", "λ=0.04, turn-cost 0.3, no pan") if k in G]
for k in pick:
    g = np.linspace(0, 60, 121); ax[0].plot(g, np.mean([np.interp(g, r["dcurve"], r["curve"]) for r in G[k]], 0), label=k, color=col[k])
    gt = np.linspace(0, 600, 121); ax[1].plot(gt, np.mean([np.interp(gt, r["tcurve"], r["curve"]) for r in G[k]], 0), color=col[k])
ax[0].set_xlabel("distance flown (m)"); ax[1].set_xlabel("flight time (s)  [1 m/s forward, 90°/s yaw]"); [a.set_ylabel("coverage") for a in ax]; [a.grid(alpha=.25) for a in ax]; ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{OUT}/curves.png", dpi=140); plt.close()
# fig3: action composition
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9)); ks = order
comp = np.array([[np.mean([r["kcount"][x] / max(1, sum(r["kcount"].values())) for r in G[k]]) for x in ("yaw", "fwd", "up")] for k in ks])
b = np.zeros(len(ks))
for i, (x, c) in enumerate(zip(("turn (yaw)", "forward", "vertical"), ("#c9625d", "#2b6cb0", "#9aa8b8"))): ax[0].barh(range(len(ks)), comp[:, i], left=b, color=c, label=x); b += comp[:, i]
ax[0].set_yticks(range(len(ks))); ax[0].set_yticklabels(ks, fontsize=6.5); ax[0].invert_yaxis(); ax[0].set_title("share of executed actions", fontsize=10); ax[0].legend(fontsize=7)
tt = np.array([[np.mean([r["kcount"]["yaw"] * 15 / 90 for r in G[k]]), np.mean([(r["dist"]) / 1.0 for r in G[k]])] for k in ks])
ax[1].barh(range(len(ks)), tt[:, 0], color="#c9625d", label="time turning"); ax[1].barh(range(len(ks)), tt[:, 1], left=tt[:, 0], color="#2b6cb0", label="time translating"); ax[1].set_yticks([]); ax[1].invert_yaxis(); ax[1].set_title("flight time (s)", fontsize=10); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/composition.png", dpi=140); plt.close()
for k in order:
    v = rows[k]; print(f"{k:34s} n={v['n']:2d} cov@20m={v['cov20'][0]:.3f} @40m={v['cov40'][0]:.3f}±{v['cov40'][1]:.3f} @60m={v['cov60'][0]:.3f} auc={v['auc'][0]:.3f} @1000a={v['cov1000a'][0]:.3f} @300s={v['cov300s'][0]:.3f} act={v['actions']:.0f} dist={v['dist']:.0f} t={v['t_flight']:.0f}s yaw%={100*v['yaw_frac']:.0f} m/act={v['m_per_act']:.3f} plan/100a={v['nplan100']:.0f} pans={v['pans']:.1f} coll={v['coll']:.1f} reach60={v['reach60']:.2f}")
for k, v in P.items(): print(k, {m: f"{x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]" for m, x in v.items()})
