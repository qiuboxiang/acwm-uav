import json, glob, numpy as np, collections, imageio, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = [r for f in sorted(glob.glob("results/*.json")) for r in json.load(open(f))]
pols = ["random", "nbv", "frontier"]; scenes = sorted({r["scene"] for r in R})
agg = collections.defaultdict(list)
for r in R: agg[(r["scene"], r["policy"])].append(r)
L = ["| scene | policy | coverage | collisions | path (m) | rollout wall (s) | ms/step |", "|---|---|---|---|---|---|---|"]
for s in scenes:
    for p in pols:
        a = agg[(s, p)]; f = lambda k: np.mean([x[k] for x in a])
        L.append(f"| {s} | {p} | {f('coverage'):.3f} ± {np.std([x['coverage'] for x in a]):.3f} | {f('collisions'):.1f} | {f('dist'):.0f} | {f('wall_s'):.1f} | {f('ms_per_step'):.0f} |")
L += ["", "**All scenes (mean over scenes x seeds):**", "", "| policy | coverage | collisions | path (m) | rollout wall (s) | ms/step |", "|---|---|---|---|---|---|"]
for p in pols:
    a = [r for r in R if r["policy"] == p]; f = lambda k: np.mean([x[k] for x in a])
    L.append(f"| {p} | {f('coverage'):.3f} | {f('collisions'):.1f} | {f('dist'):.0f} | {f('wall_s'):.1f} | {f('ms_per_step'):.0f} |")
open("results/summary.md", "w").write("\n".join(L)); print("\n".join(L))
fig, ax = plt.subplots(1, 2, figsize=(11, 4)); col = dict(random="#888", nbv="#e69f00", frontier="#0072b2")
for p in pols:
    C = np.mean([r["curve"] for r in R if r["policy"] == p], 0); ax[0].plot(np.arange(len(C)) * 10, C, label=p, color=col[p])
ax[0].set_xlabel("action steps"); ax[0].set_ylabel("GT surface coverage"); ax[0].legend(); ax[0].set_title("Coverage vs steps (mean, 8 scenes x 3 seeds)")
w = 0.27
for i, p in enumerate(pols): ax[1].bar(np.arange(len(scenes)) + i * w, [np.mean([x["coverage"] for x in agg[(s, p)]]) for s in scenes], w, label=p, color=col[p])
ax[1].set_xticks(np.arange(len(scenes)) + w); ax[1].set_xticklabels([s.replace("_", "\n", 1) for s in scenes], fontsize=6); ax[1].set_title("Final coverage per scene")
plt.tight_layout(); plt.savefig("results/summary.png", dpi=130)
# per-scene side-by-side video of the three baselines (seed 0), left->right: random | nbv | frontier
for s in scenes:
    vs = [imageio.mimread(f"results/{s}__{p}__0.mp4", memtest=False) for p in pols]; n = min(map(len, vs))
    imageio.mimsave(f"results/{s}__compare.mp4", [np.concatenate([v[i] for v in vs], 0) for i in range(n)], fps=15)
