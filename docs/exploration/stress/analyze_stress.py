import json, glob, sys, collections, numpy as np
def load(d): return [r for f in glob.glob(f"{d}/*.json") for r in json.load(open(f))]
def summarize(d):
    R = load(d); G = collections.defaultdict(list)
    for r in R: G[r["cfg"].split(":")[0]].append(r)
    out = {}
    for k, rs in sorted(G.items()):
        def flat(r):  # deadlock: coverage gained < 0.01 over the last 1000 actions while far from complete
            a = np.array(r["acurve"]); c = np.array(r["curve"]); prev = np.interp(max(0, r["actions"] - 1000), a, c); return r["actions"] >= 1500 and c[-1] - prev < 0.01 and c[-1] < 0.9
        stuck = [r for r in rs if flat(r)]
        out[k] = dict(n=len(rs), cov=np.mean([r["coverage"] for r in rs]), stuck=len(stuck), dist=np.mean([r["dist"] for r in rs]), coll=np.mean([r["collisions"] for r in rs]),
                      drift=np.mean([r["final_drift_m"] for r in rs]), yaw=np.mean([r["kcount"]["yaw"] / max(1, sum(r["kcount"].values())) for r in rs]), acts=np.mean([r["actions"] for r in rs]),
                      nof=np.mean([r["stc"].get("no_frontier", 0) / max(1, sum(r["stc"].values())) for r in rs]), oob=np.mean([r["oob"] for r in rs]), reach97=np.mean([r["coverage"] >= 0.97 for r in rs]),
                      stuck_runs=[(r["scene"].replace("_apartment", "").replace("_0", ""), r["seed"], round(r["dist"]), round(r["coverage"], 2)) for r in stuck])
    return out
for d in sys.argv[1:]:
    print("==", d)
    for k, v in summarize(d).items():
        print(f"{k:16s} n={v['n']:2d} cov={v['cov']:.3f} flat-last-1000-actions={v['stuck']}/{v['n']} dist={v['dist']:.0f} actions={v['acts']:.0f} coll={v['coll']:.1f} drift={v['drift']:.1f} yaw%={100*v['yaw']:.0f} no_frontier%={100*v['nof']:.0f} oob={v['oob']:.0f} reach97={v['reach97']:.2f}")
        if v["stuck_runs"]: print("   stuck:", v["stuck_runs"])
