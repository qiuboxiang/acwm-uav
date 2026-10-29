"""docs/exploration/v5/final_curves.png: v4 vs v5 10 h datasets - union coverage vs hours, per-trajectory coverage, action-label mix."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from analyze_collection import merge, OFFICIAL
H = os.path.dirname(os.path.abspath(__file__)); AN = f"{H}/../exploration_v5/analysis"; RP = f"{H}/../exploration_v5/replay"; OUT = f"{H}/../../../docs/exploration/v5/final_curves.png"
S = {k: json.load(open(f"{AN}/collect_{k}_summary.json")) for k in ("v4", "v5")}
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for k, c in zip(S, ("#8a8f98", "#1f5fbf")):
    ax[0].plot(S[k]["curve"]["hours"], S[k]["curve"]["union"], color=c, label=k)
    _, _, per, _ = merge([f"{RP}/{k}_{i}.npz" for i in range(3)]); cov = [p["coverage"] for p in per]
    ax[1].hist(cov, np.linspace(0, 0.8, 33), alpha=.55, color=c, label=f"{k} (mean {np.mean(cov):.3f})")
ax[0].set_xscale("log"); ax[0].set_xlabel("collected hours (log)"); ax[0].set_ylabel("union surface coverage"); ax[0].legend(); ax[0].grid(alpha=.25)
ax[1].set_xlabel("per-trajectory surface coverage"); ax[1].legend(); ax[1].grid(alpha=.25)
cats = list(OFFICIAL); x = np.arange(len(cats)); w = .27
for i, (lab, vals, c) in enumerate((("v4", S["v4"]["actions"]["labels"], "#8a8f98"), ("v5", S["v5"]["actions"]["labels"], "#1f5fbf"), ("official", OFFICIAL, "#2f8f4e"))):
    ax[2].bar(x + (i - 1) * w, [vals[k] for k in cats], w, label=lab, color=c)
ax[2].set_xticks(x); ax[2].set_xticklabels(cats); ax[2].set_ylabel("share of action labels"); ax[2].legend(); ax[2].grid(axis="y", alpha=.25)
plt.tight_layout(); plt.savefig(OUT, dpi=130); print(OUT)
