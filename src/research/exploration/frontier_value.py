"""Proposals 3a / 3b: learned frontier scoring (privileged information only at training time).

  3a  value regression: predict the ground-truth surface gain of each candidate viewpoint; the planner then picks the best
      predicted information rate value / (1 + kappa * cost).
  3b  oracle imitation (DARE-style, instead of full RL): a listwise ranking model trained to reproduce the choice of a
      privileged oracle that knows the true gain of every candidate; the planner picks the top-scored candidate.
  oracle  the privileged selector itself (renders the simulator per candidate) - an upper bound for selection only, not deployable.

At every planning event the hierarchical planner proposes one viewpoint per frontier cluster. For training we render the
simulator from each viewpoint and count how many ground-truth surface voxels it would newly reveal (the label). The model
only sees features computed from the agent's own map, so at test time (and on a real drone) it needs nothing privileged.

  python frontier_value.py gen   --scene S --seeds 3 --seed0 100 --out DIR     # rollouts with GT pose + render labels
  python frontier_value.py train --data DIR --scenes a,b,c --out model.pt [--kind rank]  # MLP on agent-side features
  python frontier_value.py eval  --data DIR --scenes d,e --model model.pt      # rank correlation vs heuristics
"""
from __future__ import annotations

import argparse, glob, json, os
import numpy as np

import mapping5 as M

GRID = 0.25
N_FEAT = 22


# --------------------------------------------------------------------------------------------------------------------- features
def _frustum(O, vp, yaw_deg, max_cells=16):
    """Ray-cast the agent map from a viewpoint over a 9 x 5 ray fan (90 x 60 deg): unknown / free cells passed before the first
    occupied hit, hit fraction and mean hit range. Classic next-best-view information gain, on the agent's own map."""
    az = np.deg2rad(yaw_deg) + np.deg2rad(np.linspace(-40, 40, 9)); el = np.deg2rad(np.linspace(-25, 25, 5))
    A, E = np.meshgrid(az, el); d = np.stack([-np.sin(A) * np.cos(E), np.sin(E), -np.cos(A) * np.cos(E)], -1).reshape(-1, 3)
    t = np.arange(1, max_cells + 1, 0.5)
    P = (np.asarray(vp, float) + 0.5)[None, None] + d[:, None] * t[None, :, None]
    I = np.floor(P).astype(int); inb = ((I >= 0) & (I < np.array(O.shape))).all(-1)
    I = np.clip(I, 0, np.array(O.shape) - 1); v = O[I[..., 0], I[..., 1], I[..., 2]]; v = np.where(inb, v, M.UNKNOWN)
    hit = v == M.OCC; first = np.where(hit.any(1), hit.argmax(1), len(t)); before = np.arange(len(t))[None] < first[:, None]
    unk = ((v == M.UNKNOWN) & before).sum(1) / 2.0; fre = ((v == M.FREE) & before).sum(1) / 2.0
    return np.array([unk.sum() / len(d), fre.sum() / len(d), hit.any(1).mean(), (np.where(hit.any(1), t[np.minimum(first, len(t) - 1)], max_cells)).mean() / max_cells])


def _box(O, c, r):
    lo = np.maximum(np.floor(c - r).astype(int), 0); hi = np.minimum(np.ceil(c + r).astype(int) + 1, np.array(O.shape))
    b = O[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; n = max(1, b.size)
    return [(b == M.UNKNOWN).sum() / n, (b == M.FREE).sum() / n, (b == M.OCC).sum() / n]


def features(A, O, lo, cand):
    """(n, N_FEAT) agent-side features for the planner's candidate viewpoints (crop-local map O, crop offset lo)."""
    known = float((O != M.UNKNOWN).mean()); costs = np.array([c["cost"] for c in cand]); rank = costs.argsort().argsort() / max(1, len(cand) - 1)
    out = []
    for c, rk in zip(cand, rank):
        vp = np.array(c["vp_crop"]); cen = np.array(c["centroid"]) - lo
        f = [np.log1p(c["size"]), np.log1p(c["gain"]), c["unk"], c["cost"] / 10.0, c["dyaw"] / 180.0, c["dz"], (A.origin[1] + (c["vp"][1] + 0.5) * GRID - A.z0),
             np.linalg.norm((cen - vp) * GRID)]
        f += list(_frustum(O, vp, c["yaw"])); f += _box(O, cen, 4) + _box(O, cen, 8)
        f += [known, np.log1p(len(cand)), float(rk), np.log1p(c["cost"])]
        out.append(f)
    return np.asarray(out, np.float32)


# --------------------------------------------------------------------------------------------------------------------- model
class Model:
    def __init__(self, state):
        import torch
        self.torch = torch; self.mu = np.array(state["mu"], np.float32); self.sd = np.array(state["sd"], np.float32); self.kind = state.get("kind", "value")
        self.net = make_net(); self.net.load_state_dict(state["net"]); self.net.eval()

    def __call__(self, A, O, lo, cand):
        X = (features(A, O, lo, cand) - self.mu) / self.sd
        with self.torch.no_grad(): y = self.net(self.torch.from_numpy(X)).numpy().reshape(-1)
        return y if self.kind == "rank" else np.expm1(np.maximum(y, 0))


def frustum_gain(A, O, lo, cand):
    """Classic next-best-view gain on the agent's own map: unknown cells passed by a 9x5 ray fan before the first obstacle."""
    return np.array([_frustum(O, np.array(c["vp_crop"]), c["yaw"])[0] for c in cand], float)


def oracle(A, O, lo, cand):
    """Privileged upper bound: true 3-view gain of every candidate, rendered in the simulator (evaluation only)."""
    return np.array([render_gain(A.env, A, c)[1] for c in cand], float)


def make_net():
    import torch.nn as nn
    return nn.Sequential(nn.Linear(N_FEAT, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))


def load(path):
    import torch
    return Model(torch.load(path, map_location="cpu", weights_only=False))


# --------------------------------------------------------------------------------------------------------------------- data generation
def render_gain(env, A, c, CAM_H=1.5):
    """Privileged label: new ground-truth surface voxels seen from the candidate viewpoint (training data only)."""
    S = env.S; p0, y0 = S.pose()
    cam = A.origin + (np.array(c["vp"]) + 0.5) * GRID; base = cam - np.array([0, CAM_H, 0]); out = []
    for dy in (0.0, -40.0, 40.0):
        S.set_pose(base, (c["yaw"] + dy) % 360); d = S.obs()[1]
        R = M.yaw_R((c["yaw"] + dy) % 360); f = 128.0
        u, v = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4)); r4 = np.stack([(u - 128) / f, -(v - 128) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        d4 = d[::4, ::4].reshape(-1); m = (d4 > .05) & (d4 < 8); w = (r4[m] * d4[m, None]) @ R.T + cam
        out.append(set(map(tuple, np.floor(w / 0.25).astype(np.int32))))
    S.set_pose(p0, y0)
    gt = env.G.gt_vox
    single = len((out[0] - env.seen) & gt); sweep = len(((out[0] | out[1] | out[2]) - env.seen) & gt)
    return single, sweep


def gen(scene, seeds, seed0, outdir, steps=1500):
    from uav_sim import UAVSim
    from scenes import paths, Geometry
    import explore5 as X5
    g, n = paths(scene); S = UAVSim(g); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds()); nav = X5.nav_cell_set(S, 500)
    os.makedirs(outdir, exist_ok=True)
    for sd in range(seed0, seed0 + seeds):
        rows = []
        cfg = dict(plan="hier", pose="gt", map="logodds", sel="tsp", div=False, translate=False, vert=False)
        X5.rollout5(cfg, S, G, steps, sd, nav, scene, feat_log=_Collector(rows), dist_budget=60.0)
        np.save(f"{outdir}/{scene}__{sd}.npy", np.array(rows, np.float32)); print(scene, sd, len(rows), flush=True)


class _Collector:
    """Passed as HierPlanner.feat_log; HierPlanner calls .append(dict(...)) with the crop and candidates at each planning event."""
    def __init__(self, rows): self.rows = rows; self.event = 0

    def append(self, item):
        A = item["A"]; cand = item["cand"]
        X = features(A, item["O"], item["lo"], cand)
        for c, x in zip(cand, X):
            s1, s3 = render_gain(A.env, A, c); self.rows.append(np.r_[x, s1, s3, self.event])
        self.event += 1


# --------------------------------------------------------------------------------------------------------------------- train / eval
KAPPA = 0.4


def cost0(X): return 10.0 * X[:, 3] + 2.0 * X[:, 4]  # path cost (m) + yaw cost, recovered from features 3 and 4


def _load(data, scenes, events=False):
    arrs = []; off = 0
    for s in scenes:
        for f in sorted(glob.glob(f"{data}/{s}__*.npy")):
            a = np.load(f)
            if len(a): a = a.copy(); a[:, N_FEAT + 2] += off; off = a[:, N_FEAT + 2].max() + 1; arrs.append(a)
    D = np.concatenate(arrs)
    return (D[:, :N_FEAT], D[:, N_FEAT + 1], D[:, N_FEAT + 2].astype(int)) if events else (D[:, :N_FEAT], D[:, N_FEAT + 1])  # label: 3-view sweep gain


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def train_rank(data, scenes, out, epochs=60, seed=0):
    """3b: listwise imitation of the privileged oracle choice argmax gain / (1 + kappa * cost) within each planning event."""
    import torch
    torch.manual_seed(seed); X, y, ev = _load(data, scenes, True); mu = X.mean(0); sd = X.std(0) + 1e-6
    rate = y / (1.0 + KAPPA * cost0(X)); evs = np.unique(ev); groups = [np.where(ev == e)[0] for e in evs]; groups = [g for g in groups if len(g) > 1]
    tgt = [int(g[np.argmax(rate[g])]) for g in groups]
    Xt = torch.from_numpy(((X - mu) / sd).astype(np.float32)); net = make_net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        order = np.random.default_rng(seed + ep).permutation(len(groups))
        for i in range(0, len(order), 32):
            loss = 0.0
            for k in order[i:i + 32]:
                g = groups[k]; s = net(Xt[g]).reshape(-1); loss = loss + (torch.logsumexp(s, 0) - s[list(g).index(tgt[k])])
            opt.zero_grad(); (loss / len(order[i:i + 32])).backward(); opt.step()
    with torch.no_grad(): sc = net(Xt).numpy().reshape(-1)
    top1 = float(np.mean([g[np.argmax(sc[g])] == t for g, t in zip(groups, tgt)]))
    torch.save(dict(net=net.state_dict(), mu=mu.tolist(), sd=sd.tolist(), scenes=scenes, n=int(len(X)), kind="rank"), out)
    print(json.dumps(dict(kind="rank", train_scenes=scenes, events=len(groups), train_top1=top1)))


def train(data, scenes, out, epochs=60, seed=0):
    import torch
    torch.manual_seed(seed); X, y = _load(data, scenes); mu = X.mean(0); sd = X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mu) / sd); yt = torch.from_numpy(np.log1p(y).astype(np.float32))[:, None]
    net = make_net(); opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4); n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            b = perm[i:i + 256]; loss = ((net(Xt[b]) - yt[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): pred = net(Xt).numpy().reshape(-1)
    torch.save(dict(net=net.state_dict(), mu=mu.tolist(), sd=sd.tolist(), scenes=scenes, n=int(n)), out)
    print(json.dumps(dict(train_scenes=scenes, n=int(n), train_spearman=spearman(pred, y))))


def evaluate(data, scenes, model, rank_model=None):
    """Held-out scenes: rank correlation with the true gain, and per planning event how good the selected candidate is
    (top-1 agreement with the oracle and the selected / best information-rate ratio) for each selection rule."""
    X, y, ev = _load(data, scenes, True); m = load(model)
    with m.torch.no_grad(): p = np.expm1(np.maximum(m.net(m.torch.from_numpy(((X - m.mu) / m.sd).astype(np.float32))).numpy().reshape(-1), 0))
    rate = y / (1.0 + KAPPA * cost0(X)); groups = [g for g in (np.where(ev == e)[0] for e in np.unique(ev)) if len(g) > 1]
    rules = dict(nearest=-cost0(X), heuristic_rate=np.expm1(X[:, 1]) / (1 + KAPPA * cost0(X)), frustum_rate=X[:, 8] / (1 + KAPPA * cost0(X)), learned_rate=p / (1 + KAPPA * cost0(X)))
    if rank_model:
        r = load(rank_model)
        with r.torch.no_grad(): rules["rank_3b"] = r.net(r.torch.from_numpy(((X - r.mu) / r.sd).astype(np.float32))).numpy().reshape(-1)
    sel = {k: dict(top1=float(np.mean([g[np.argmax(v[g])] == g[np.argmax(rate[g])] for g in groups])),
                   rate_ratio=float(np.mean([rate[g][np.argmax(v[g])] / max(rate[g].max(), 1e-9) for g in groups]))) for k, v in rules.items()}
    res = dict(test_scenes=scenes, n=int(len(y)), events=len(groups), spearman=dict(learned=spearman(p, y), heuristic_visible_frontier=spearman(X[:, 1], y), frustum_unknown=spearman(X[:, 8], y),
               cluster_size=spearman(X[:, 0], y), neg_cost=spearman(-X[:, 3], y)), selection=sel)
    print(json.dumps(res)); return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "train", "eval"]); ap.add_argument("--scene"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=100); ap.add_argument("--out"); ap.add_argument("--data"); ap.add_argument("--scenes"); ap.add_argument("--model"); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--kind", default="value", choices=["value", "rank"]); ap.add_argument("--rank_model", default=None)
    a = ap.parse_args()
    if a.cmd == "gen": gen(a.scene, a.seeds, a.seed0, a.out, a.steps)
    elif a.cmd == "train": (train_rank if a.kind == "rank" else train)(a.data, a.scenes.split(","), a.out)
    else: evaluate(a.data, a.scenes.split(","), a.model, a.rank_model)
