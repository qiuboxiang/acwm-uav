"""Sensor-side building blocks for the v5 exploration agent (pure numpy / scipy, no simulator imports).

Everything here consumes only what a real drone has on board: depth images (sensor or monocular estimate), the commanded
motion, and the agent's own map. Nothing reads the scene mesh, ground-truth pose or collision flags.

  LogOddsGrid     probabilistic voxel map (OctoMap-style clamped log-odds); free-space updates can erase stale / drifted obstacles
  DepthOdometry   4-DoF (x, y, z, yaw) point-to-plane ICP against a keyframe, with the commanded motion as a weak prior and a
                  "blocked" hypothesis (no translation) so undetected collisions do not corrupt the pose
  ViewBits        per-voxel bitmask of observed viewing directions (8 azimuth x 3 elevation bins) for view-diversity scoring
  grid_graph / shortest paths / frontier clustering / open-TSP ordering used by the hierarchical planner
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation, label, uniform_filter
from scipy.sparse import csgraph
from scipy.spatial import cKDTree

UNKNOWN, FREE, OCC = 0, 1, 2


def yaw_R(yaw_deg: float) -> np.ndarray:
    """Rotation about +y (habitat up) used throughout the agent: world = R @ camera."""
    r = np.deg2rad(yaw_deg); c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# --------------------------------------------------------------------------------------------------------------------- mapping
class LogOddsGrid:
    """Clamped log-odds occupancy. ``occ`` mirrors the three-state map the planners read (0 unknown / 1 free / 2 occupied):
    a cell is occupied when its log-odds reaches ``t_occ``; any other observed cell is free; never-observed cells are unknown.
    One frame contributes at most one update per cell; a cell that is both hit and traversed in a frame counts as hit."""

    def __init__(self, shape, l_free=-0.4, l_occ=0.85, l_min=-2.0, l_max=3.5, t_occ=0.4):
        self.shape = tuple(int(s) for s in shape)
        self.L = np.zeros(self.shape, np.float32); self.occ = np.zeros(self.shape, np.uint8); self.seen = np.zeros(self.shape, bool)
        self.l_free, self.l_occ, self.l_min, self.l_max, self.t_occ = l_free, l_occ, l_min, l_max, t_occ
        self.n_cleared = 0  # occupied -> free transitions (stale or drifted obstacles erased)

    def _lin(self, idx):
        idx = np.asarray(idx, np.int64).reshape(-1, 3)
        ok = ((idx >= 0) & (idx < np.array(self.shape))).all(1)
        return np.unique(np.ravel_multi_index(idx[ok].T, self.shape)) if ok.any() else np.zeros(0, np.int64)

    def update(self, free_idx, hit_idx):
        hit = self._lin(hit_idx); free = np.setdiff1d(self._lin(free_idx), hit, assume_unique=True)
        L = self.L.reshape(-1); O = self.occ.reshape(-1); S = self.seen.reshape(-1)
        was_occ = O[free] == OCC
        L[free] = np.maximum(L[free] + self.l_free, self.l_min); L[hit] = np.minimum(L[hit] + self.l_occ, self.l_max)
        S[free] = True; S[hit] = True
        touched = np.concatenate([free, hit])
        O[touched] = np.where(L[touched] >= self.t_occ, OCC, FREE).astype(np.uint8)
        self.n_cleared += int((was_occ & (O[free] == FREE)).sum())
        return touched


def ray_cells(origin, dirs, dd, hit, grid, map_origin, free_margin=0.25, max_range=6.0, step=None):
    """Voxel indices traversed by depth rays (free, up to ``free_margin`` before the return) and the hit voxels."""
    step = step or grid / 2
    frees = []
    for dist in np.arange(0.0, max_range, step):
        m = dist < dd - free_margin
        if not m.any(): break
        frees.append(np.floor((origin + dirs[m] * dist - map_origin) / grid).astype(np.int64))
    free = np.concatenate(frees) if frees else np.zeros((0, 3), np.int64)
    hp = origin + dirs[hit] * dd[hit, None]
    return free, np.floor((hp - map_origin) / grid).astype(np.int64), hp


# --------------------------------------------------------------------------------------------------------------------- odometry
class DepthOdometry:
    """Frame-to-keyframe 4-DoF point-to-plane ICP on the depth image. Roll/pitch are taken as known (a real drone gets them
    from its IMU). The commanded motion is a weak Gaussian prior; for translational commands a second hypothesis
    "the move was blocked" is also tried and the lower robust cost wins."""

    def __init__(self, f, res=256, stride=4, max_range=8.0, sigma_t=0.5, sigma_yaw=5.0, iters=12, kf_dist=0.75, kf_yaw=40.0, min_inliers=150, block_penalty=3e-4, n_kf=200):
        self.f, self.res, self.stride, self.max_range = f, res, stride, max_range
        self.lam_t = 1.0 / sigma_t ** 2; self.lam_y = 1.0 / np.deg2rad(sigma_yaw) ** 2
        self.iters, self.kf_dist, self.kf_yaw, self.min_inliers, self.block_penalty = iters, kf_dist, kf_yaw, min_inliers, block_penalty
        u, v = np.meshgrid(np.arange(0, res, stride) + 0.5 * stride - 0.5, np.arange(0, res, stride) + 0.5 * stride - 0.5)
        self.rx = (u - res / 2) / f; self.ry = -(v - res / 2) / f
        self.kf = None; self.kfs = []; self.n_kf = n_kf; self.stats = dict(tracked=0, fallback=0, blocked_hyp=0, keyframes=0, revisits=0)

    def cloud(self, depth):
        s = self.stride; d = depth[s // 2::s, s // 2::s].astype(np.float64)
        P = np.stack([self.rx * d, self.ry * d, -d], -1)
        ok = (d > 0.1) & (d < self.max_range)
        dx = np.zeros_like(P); dy = np.zeros_like(P)
        dx[:, 1:-1] = P[:, 2:] - P[:, :-2]; dy[1:-1] = P[2:] - P[:-2]
        n = np.cross(dx, dy); nn = np.linalg.norm(n, axis=-1)
        ok[:, [0, -1]] = False; ok[[0, -1]] = False
        # reject depth discontinuities: neighbours must be close relative to range
        jump = np.zeros_like(d); jump[:, 1:-1] = np.abs(d[:, 2:] - d[:, :-2]); jump[1:-1] = np.maximum(jump[1:-1], np.abs(d[2:] - d[:-2]))
        ok &= (nn > 1e-9) & (jump < 0.08 * d + 0.03)
        n = n / np.maximum(nn, 1e-12)[..., None]
        flip = (n * P).sum(-1) > 0; n[flip] *= -1  # face the camera
        return P[ok], n[ok]

    def set_keyframe(self, depth, pos, yaw):
        P, N = self.cloud(depth); R = yaw_R(yaw)
        if len(P) < self.min_inliers: return
        with np.errstate(all="ignore"): P = P @ R.T + pos; N = N @ R.T
        self.kf = dict(P=P, N=N, tree=cKDTree(P), pos=np.array(pos, float), yaw=float(yaw), id=self.stats["keyframes"]); self.stats["keyframes"] += 1
        self.kfs.append(self.kf); self.kfs = self.kfs[-self.n_kf:]

    def _candidates(self, pos, yaw):
        """Keyframes to align against: the one closest in pose (re-observing an old keyframe bounds drift, a light loop
        closure) and the most recent one."""
        if not self.kfs: return []
        d = [np.linalg.norm(k["pos"] - pos) + 0.02 * abs((k["yaw"] - yaw + 180) % 360 - 180) for k in self.kfs]
        near = [self.kfs[i] for i in np.argsort(d) if abs((self.kfs[i]["yaw"] - yaw + 180) % 360 - 180) < 45 and np.linalg.norm(self.kfs[i]["pos"] - pos) < 2.0][:1]
        out = near + ([self.kfs[-1]] if not near or near[0] is not self.kfs[-1] else [])
        return out

    def _score(self, P, pos, yaw, pos_pred, yaw_pred, tau=0.05):
        """Truncated point-to-plane cost over *all* points (missing correspondences pay tau^2) plus the motion prior, and the
        inlier fraction. Normalising by all points (not inliers) stops a bad alignment with few matches from looking cheap."""
        kf = self.kf; Q = P @ yaw_R(yaw).T + pos
        dist, j = kf["tree"].query(Q, distance_upper_bound=0.25); ok = np.isfinite(dist)
        r = np.full(len(P), tau); r[ok] = np.minimum(np.abs((kf["N"][j[ok]] * (Q[ok] - kf["P"][j[ok]])).sum(1)), tau)
        dy = np.deg2rad((yaw - yaw_pred + 180) % 360 - 180)
        prior = self.lam_t * float(((pos - pos_pred) ** 2).sum()) + self.lam_y * dy * dy
        return float((r * r).mean() + prior / len(P)), float((r < tau).mean())

    def _solve(self, P, N, pos0, yaw0, pos_pred, yaw_pred):
        kf = self.kf; pos = pos0.astype(float).copy(); yaw = np.deg2rad(yaw0); yp = np.deg2rad(yaw_pred)
        lam = np.r_[self.lam_t, self.lam_t, self.lam_t, self.lam_y]; Ht = np.zeros((3, 3))
        with np.errstate(all="ignore"):  # numpy>=2 + Accelerate emits spurious matmul warnings
            for it in range(self.iters):  # Gauss-Newton with re-association
                R = yaw_R(np.rad2deg(yaw)); V = P @ R.T; Q = V + pos
                dist, j = kf["tree"].query(Q, distance_upper_bound=0.6 if it < 3 else 0.25)
                ok = np.isfinite(dist)
                if ok.sum() < self.min_inliers: break
                nt = kf["N"][j[ok]]; ok2 = np.abs(((N[ok] @ R.T) * nt).sum(1)) > 0.7
                nt = nt[ok2]; v = V[ok][ok2]; q = Q[ok][ok2]; pt = kf["P"][j[ok]][ok2]
                if len(q) < self.min_inliers: break
                r = (nt * (q - pt)).sum(1); a = np.abs(r); w = np.where(a < 0.03, 1.0, 0.03 / np.maximum(a, 1e-9))
                J = np.concatenate([nt, (nt[:, 0] * v[:, 2] - nt[:, 2] * v[:, 0])[:, None]], 1)
                ep = np.r_[pos - pos_pred, yaw - yp]
                H = (J * w[:, None]).T @ J + np.diag(lam); g = (J * w[:, None]).T @ r + lam * ep
                Ht = (nt * w[:, None]).T @ nt / max(w.sum(), 1e-9)  # translational observability (normalised)
                dx = -np.linalg.solve(H, g); pos += dx[:3]; yaw += dx[3]
                if np.abs(dx).max() < 1e-4: break
        y = float(np.rad2deg(yaw)); c, frac = self._score(P, pos, y, pos_pred, yaw_pred); self._Ht = Ht
        return pos, y, c, frac

    def track(self, depth, pos_pred, yaw_pred, pos_prev=None, yaw_prev=None, translated=False):
        """Returns (pos, yaw, info). ``translated`` enables the blocked hypothesis (start from the previous position)."""
        pos_pred = np.asarray(pos_pred, float)
        if not self.kfs:
            self.set_keyframe(depth, pos_pred, yaw_pred); self.stats["fallback"] += 1; return pos_pred, yaw_pred, dict(ok=False)
        P, N = self.cloud(depth)
        if len(P) < self.min_inliers:
            self.stats["fallback"] += 1; return pos_pred, yaw_pred, dict(ok=False)
        if len(P) > 1800:
            sel = np.linspace(0, len(P) - 1, 1800).astype(int); P, N = P[sel], N[sel]
        # hypothesis 0: the commanded motion happened (prior centred on it); hypothesis 1: the move was blocked (prior centred
        # on the previous position, small a-priori penalty). Each hypothesis is scored against its own prior.
        hyps = [(pos_pred, 0.0)]
        if translated and pos_prev is not None: hyps.append((np.asarray(pos_prev, float), self.block_penalty))
        u = None
        if translated and pos_prev is not None:
            mv = pos_pred - np.asarray(pos_prev, float); u = mv / np.linalg.norm(mv) if np.linalg.norm(mv) > 1e-6 else None

        def solve_on(kf):
            self.kf = kf; res = []
            for k, (pm, pen) in enumerate(hyps):
                p, y, c, frac = self._solve(P, N, pm, yaw_pred, pm, yaw_pred); res.append((p, y, c + pen, frac, k, kf, self._Ht))
            cand = res[0]
            if len(res) > 1 and u is not None:
                obs = float(u @ res[1][6] @ u)  # can the depth constrain motion along the commanded direction at all?
                if obs >= 0.1 and res[1][2] < min(res[0][2] - self.block_penalty, 0.85 * res[0][2]) and res[1][3] >= 0.4: cand = res[1]
            return cand if cand[3] >= 0.3 else None
        # chain on the most recent keyframe; an older, nearer keyframe (revisit) may only nudge that solution (<= 0.15 m, <= 2 deg),
        # which bounds drift when re-observing a place without letting a wrong association make the pose jump
        best = solve_on(self.kfs[-1])
        old = [k for k in self._candidates(pos_pred, yaw_pred) if k is not self.kfs[-1]]
        if old:
            alt = solve_on(old[0]) if best is not None else None
            if alt is not None and np.linalg.norm(alt[0] - best[0]) <= 0.15 and abs((alt[1] - best[1] + 180) % 360 - 180) <= 2.0 and alt[3] >= 0.5: best = alt
        if best is None:
            self.stats["fallback"] += 1; self.set_keyframe(depth, pos_pred, yaw_pred); return pos_pred, yaw_pred, dict(ok=False)
        p, y, c, frac, k, kf = best[:6]; self.kf = kf
        self.stats["tracked"] += 1; self.stats["blocked_hyp"] += int(k == 1); self.stats["revisits"] += int(kf is not self.kfs[-1])
        y = float(y % 360)
        if np.linalg.norm(p - kf["pos"]) > self.kf_dist or abs((y - kf["yaw"] + 180) % 360 - 180) > self.kf_yaw or frac < 0.5:
            self.set_keyframe(depth, p, y)
        return p, y, dict(ok=True, cost=c, inlier_frac=frac, hyp=k)


# --------------------------------------------------------------------------------------------------------------------- views
N_AZ, N_EL = 8, 3


def view_bins(vec_cell_to_cam):
    """Bin index (0..23) of the direction from a surface cell towards the camera: 8 azimuth x 3 elevation bins."""
    v = vec_cell_to_cam / (np.linalg.norm(vec_cell_to_cam, axis=-1, keepdims=True) + 1e-9)
    az = (((np.arctan2(v[..., 0], v[..., 2]) + np.pi) / (2 * np.pi) * N_AZ).astype(int)) % N_AZ
    el = np.digitize(np.arcsin(np.clip(v[..., 1], -1, 1)), [-0.35, 0.35])
    return az * N_EL + el


class ViewBits:
    """For every map voxel, which of the 24 viewing-direction bins it has been observed from (agent-side, from its own map)."""

    def __init__(self, shape): self.B = np.zeros(shape, np.uint32)

    def add(self, hit_idx, hit_pts, cam):
        ok = ((hit_idx >= 0) & (hit_idx < np.array(self.B.shape))).all(1)
        if not ok.any(): return
        b = view_bins(cam[None] - hit_pts[ok]); i = hit_idx[ok]
        np.bitwise_or.at(self.B, (i[:, 0], i[:, 1], i[:, 2]), (np.uint32(1) << b.astype(np.uint32)))

    @staticmethod
    def popcount(x):
        x = x.astype(np.uint32); c = np.zeros(x.shape, np.int32)
        for k in range(N_AZ * N_EL): c += ((x >> np.uint32(k)) & np.uint32(1)).astype(np.int32)
        return c


# --------------------------------------------------------------------------------------------------------------------- planning
H_OFF = [(1, 0, 0, 1.0), (-1, 0, 0, 1.0), (0, 0, 1, 1.0), (0, 0, -1, 1.0)]
D_OFF = [(1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1)]


def _shift_pairs(mask, off):
    """Linear-index pairs (a, a+off) where both cells are True."""
    dx, dy, dz = off; X, Y, Z = mask.shape
    sa = (slice(max(0, -dx), X - max(0, dx)), slice(max(0, -dy), Y - max(0, dy)), slice(max(0, -dz), Z - max(0, dz)))
    sb = tuple(slice(a.start + o, a.stop + o) for a, o in zip(sa, off))
    both = mask[sa] & mask[sb]
    ia = np.argwhere(both) + np.array([s.start for s in sa]); ib = ia + np.array(off)
    return ia, ib


def grid_graph(trav, near_obst=None, grid=0.25, vert_w=2.0, obst_w=1.0, vmask=None):
    """Sparse undirected graph over traversable cells: 4 horizontal + 4 horizontal-diagonal (no corner cutting) + 2 vertical
    neighbours. Edge length in metres, inflated near obstacles (``near_obst``) so paths keep clearance when they can."""
    idx = -np.ones(trav.shape, np.int64); cells = np.argwhere(trav); idx[tuple(cells.T)] = np.arange(len(cells))
    rows, cols, w = [], [], []
    pen = (1.0 + obst_w * near_obst.astype(float)) if near_obst is not None else np.ones(trav.shape)  # near_obst may be a graded cost

    def add(ia, ib, base):
        if len(ia) == 0: return
        rows.append(idx[tuple(ia.T)]); cols.append(idx[tuple(ib.T)])
        w.append(base * grid * 0.5 * (pen[tuple(ia.T)] + pen[tuple(ib.T)]))

    for dx, dy, dz, b in H_OFF[:1] + H_OFF[2:3]:  # +x and +z (undirected graph)
        ia, ib = _shift_pairs(trav, (dx, dy, dz)); add(ia, ib, b)
    for dx, dy, dz in [(1, 0, 1), (1, 0, -1)]:
        ia, ib = _shift_pairs(trav, (dx, dy, dz))
        if len(ia):
            ok = trav[ia[:, 0] + dx, ia[:, 1], ia[:, 2]] & trav[ia[:, 0], ia[:, 1], ia[:, 2] + dz]
            add(ia[ok], ib[ok], np.sqrt(2))
    ia, ib = _shift_pairs(trav, (0, 1, 0))
    if vmask is not None and len(ia): ok = vmask[tuple(ia.T)]; ia, ib = ia[ok], ib[ok]  # vertical edge allowed at its lower cell
    add(ia, ib, vert_w)
    n = len(cells)
    if rows:
        r = np.concatenate(rows); c = np.concatenate(cols); ww = np.concatenate(w)
        G = sparse.coo_matrix((ww, (r, c)), shape=(n, n)).tocsr()
    else:
        G = sparse.csr_matrix((n, n))
    return G, idx, cells


def shortest(G, sources, limit=np.inf):
    D, P = csgraph.dijkstra(G, directed=False, indices=sources, return_predecessors=True, limit=limit)
    return D, P


def path_from(pred_row, target, cells):
    """Cell path (start excluded) from a dijkstra predecessor row."""
    out = []; k = int(target)
    while k >= 0:
        out.append(tuple(int(v) for v in cells[k])); k = int(pred_row[k])
    return out[::-1][1:]


def frontier_clusters(occ, reach, band, tile=8, min_size=4, min_unknown=4):
    """Frontier = reachable free cell with an unknown 6-neighbour (vertical neighbours only inside the flight band) and at least
    ``min_unknown`` unknown cells in its 3x3x3 neighbourhood (rejects single-voxel speckle). Clusters = 26-connected components
    split into ``tile`` x ``tile`` horizontal tiles (FUEL-style splitting of large frontiers)."""
    unk = occ == UNKNOWN
    nb = np.zeros_like(unk)
    nb[1:] |= unk[:-1]; nb[:-1] |= unk[1:]; nb[:, :, 1:] |= unk[:, :, :-1]; nb[:, :, :-1] |= unk[:, :, 1:]
    ub = unk & band
    nb[:, 1:] |= ub[:, :-1]; nb[:, :-1] |= ub[:, 1:]
    dense = uniform_filter(unk.astype(np.float32), size=3, mode="constant", cval=0.0) * 27 >= min_unknown - 0.5
    F = reach & nb & dense
    lab, nl = label(F, structure=np.ones((3, 3, 3), int))
    if nl == 0: return [], F
    c = np.argwhere(F); l = lab[tuple(c.T)]
    key = l.astype(np.int64) * 1_000_000 + (c[:, 0] // tile) * 1000 + (c[:, 2] // tile)
    uk, inv = np.unique(key, return_inverse=True)
    out = []
    for k in range(len(uk)):
        cc = c[inv == k]
        if len(cc) >= min_size: out.append(cc)
    return out, F


def los_free(occ, a, b, step=0.5):
    """Line of sight in cell units between cell centres a (K,3) and b (K,3): no occupied cell sampled along the segment."""
    a = np.asarray(a, float) + 0.5; b = np.asarray(b, float) + 0.5
    n = int(np.ceil(np.linalg.norm(b - a, axis=1).max() / step)) + 1 if len(a) else 1
    t = np.linspace(0, 1, max(2, n))[1:-1]
    if len(t) == 0: return np.ones(len(a), bool)
    P = (a[:, None] + (b - a)[:, None] * t[None, :, None]).astype(int)
    P = np.clip(P, 0, np.array(occ.shape) - 1)
    return ~(occ[P[..., 0], P[..., 1], P[..., 2]] == OCC).any(1)


def open_tsp(C):
    """Open path from node 0 through all nodes, nearest neighbour + 2-opt. C: (n, n) costs (asymmetric allowed)."""
    n = len(C)
    if n <= 2: return list(range(n))
    order = [0]; left = set(range(1, n))
    while left:
        j = min(left, key=lambda k: C[order[-1], k]); order.append(j); left.remove(j)

    def cost(o): return sum(C[o[i], o[i + 1]] for i in range(len(o) - 1))
    best = cost(order); improved = True; it = 0
    while improved and it < 50:
        improved = False; it += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                o = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                c = cost(o)
                if c < best - 1e-9: order, best, improved = o, c, True
    return order


def depth_gap(depth: np.ndarray, f: float, band: float = 0.3, reach: float = 2.5, open_extra: float = 0.5,
              half: float = 1.2, bin_w: float = 0.05, stride: int = 2):
    """Opening in front of the camera from one depth image (camera frame: x right, y up, z-depth forward).

    Points within ``band`` of the camera height and ``reach`` in depth are binned laterally. A lateral bin is blocked when its
    nearest point lies within ``open_extra`` of the nearest point overall (the door frame / wall the camera faces). Returns
    ``(x_left, x_right, d_front, hard_left, hard_right)`` for the open run that contains the optical axis or lies nearest to it, or
    ``None`` when no point is close (nothing to pass through) or every bin is blocked. ``hard_*`` is True when that edge is a seen
    obstacle; False when the run ends at the edge of the field of view (the true opening may be wider).
    """
    h, w = depth.shape; d = depth[::stride, ::stride].astype(float)
    u, v = np.meshgrid(np.arange(0, w, stride) + 0.5, np.arange(0, h, stride) + 0.5)
    x = (u - w / 2) / f * d; y = -(v - h / 2) / f * d
    sel = (d > 0.05) & (np.abs(y) < band) & (np.abs(x) < half) & (d < reach)
    if not sel.any(): return None
    d_front = float(d[sel].min()); blk = sel & (d < d_front + open_extra)
    edges = np.arange(-half, half + bin_w / 2, bin_w); nb = len(edges) - 1
    obs = np.zeros(nb, bool); obs[np.clip(((x[blk] + half) / bin_w).astype(int), 0, nb - 1)] = True
    seen = d_front * (w / 2) / f  # lateral half-extent the camera sees at the blocking depth: beyond it is unknown, not open
    occ = obs | (edges[:-1] < -seen) | (edges[1:] > seen)
    free = ~occ
    if not free.any(): return None
    runs, i = [], 0
    while i < nb:
        if free[i]:
            j = i
            while j + 1 < nb and free[j + 1]: j += 1
            runs.append((i, j)); i = j + 1
        else: i += 1
    def dist0(r): a, b = edges[r[0]], edges[r[1] + 1]; return 0.0 if a <= 0 <= b else min(abs(a), abs(b))
    i0, i1 = min(runs, key=lambda r: (dist0(r), -(r[1] - r[0])))
    return float(edges[i0]), float(edges[i1 + 1]), d_front, bool(i0 > 0 and obs[i0 - 1]), bool(i1 < nb - 1 and obs[i1 + 1])


def near_obstacle(occ, it=1):
    return binary_dilation(occ == OCC, structure=np.ones((3, 1, 3), bool), iterations=it)


def clearance_cost(occ):
    """Graded extra traversal cost: +3 for cells touching an obstacle (26-neighbourhood), +1 at two cells. Keeps paths centred
    in doorways and corridors without closing them (hard inflation closes ~3-cell doors)."""
    o = occ == OCC; d1 = binary_dilation(o, structure=np.ones((3, 3, 3), bool)); d2 = binary_dilation(d1, structure=np.ones((3, 1, 3), bool))
    return 3.0 * d1 + 1.0 * (d2 & ~d1)


# --------------------------------------------------------------------------------------------------------------------- model uncertainty (4a)
class ForwardEnsemble:
    """Proposal 4a stand-in until the world model exists: K online ridge-regression forward models predict the next frame's
    embedding from (current embedding, action). Their disagreement on the transition just experienced is a Plan2Explore-style
    novelty signal - high where the agent's own predictive model is uncertain. Embedding: fixed random projection of a 32x32
    grey image + 32x32 depth thumbnail (no learning, cheap enough to run onboard). ``novelty`` is what a real world model's
    ensemble disagreement would replace."""

    KINDS = ("fwd", "yaw", "up", "lat")

    def __init__(self, dim=48, k=5, lam=1.0, seed=0):
        rng = np.random.default_rng(seed); self.W = rng.normal(0, 1 / np.sqrt(2048), (2048, dim)); self.k = k; self.dim = dim
        d_in = dim + len(self.KINDS) + 1
        self.P = [np.eye(d_in) / lam for _ in range(k)]  # recursive least squares inverse covariance per member
        self.B = [rng.normal(0, 0.3, (d_in, dim)) for _ in range(k)]  # random prior means -> initial disagreement
        self.boot = np.random.default_rng(seed + 1); self.n = 0; self.mean_nov = None

    def embed(self, rgb, depth):
        import cv2
        g = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        dd = cv2.resize(np.clip(depth, 0, 8).astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA) / 8.0
        x = np.r_[g.reshape(-1) - 0.5, dd.reshape(-1) - 0.5]; return np.tanh(x @ self.W * 4)

    def _x(self, e, kind, mag):
        a = np.zeros(len(self.KINDS)); a[self.KINDS.index(kind)] = mag / (15.0 if kind == "yaw" else 0.9); return np.r_[e, a, 1.0]

    def step(self, e0, kind, mag, e1):
        """Novelty of the transition (before learning from it), then one RLS update per member on a bootstrap draw."""
        x = self._x(e0, kind, mag); pred = np.stack([x @ B for B in self.B]); nov = float(pred.var(0).mean())
        for i in range(self.k):
            if self.boot.random() < 0.8:
                P = self.P[i]; Px = P @ x; g = Px / (1.0 + x @ Px); self.B[i] = self.B[i] + np.outer(g, e1 - x @ self.B[i]); self.P[i] = P - np.outer(g, Px)
        self.n += 1; self.mean_nov = nov if self.mean_nov is None else 0.99 * self.mean_nov + 0.01 * nov
        return nov
