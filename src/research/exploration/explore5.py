"""Exploration v5: sensor-only agent with the improvements proposed in docs/exploration/improvement_proposals.md.

  map    sticky   : v4 three-state map (an occupied voxel is never cleared)
         logodds  : clamped log-odds map, free-space updates erase stale / drifted obstacles                       (proposal 1)
  pose   odom     : dead reckoning of commanded motion          corr : + v4 depth-consistency bump correction
         vo       : 4-DoF depth ICP against keyframes, commanded motion as prior, blocked hypothesis              (proposal 1)
         gt       : ground-truth pose (upper bound only)
  plan   v1 / f3  : v4 planners (nearest 3-D frontier / 2-D band frontier)
         hier     : FUEL-style hierarchical planner: frontier clusters -> viewpoints (visible-frontier gain) -> open-TSP
                    ordering over exact grid shortest paths -> path following; hysteresis + change-triggered replanning (proposal 2)
  sel    tsp | greedy | learned  : how the next viewpoint is chosen (learned = proposal 3a, see frontier_value.py)
  div    0 / 1    : when no frontier is left, fly to the reachable pose with the most unseen (voxel, viewing-direction) pairs
                    in the agent's own map, weighted by a Go-Explore-style visit-count archive (proposal 4b/4c)

The agent only ever reads: its RGB(-D) image, its own commands, its own map. The Env (simulator side) executes motion with noise,
blocks motion on real collisions without telling the agent, and scores coverage / view diversity for evaluation."""
from __future__ import annotations

import argparse, collections, json, os, time
import numpy as np
from scipy.ndimage import uniform_filter

from uav_sim import UAVSim
from scenes import paths, Geometry
import explore2 as E2
from explore2 import frontier_step, Frontier, seg_free_map, CAM_H, RADIUS, GRID, A_FWD_L, A_FWD_S, A_TURN_L, A_UP
from explore3 import Env, DepthNet, pol_random, pol_wander, A_LAT
from explore4 import Frontier3, SensorAgent2, crop_plan, diversity, nav_cell_set, pick_alt
import mapping5 as M


# ===================================================================================================================== env (privileged)
class Env5(Env):
    """Simulator side. Adds lateral moves, renders once per sense, and records ground-truth view diversity (evaluation only)."""

    def __init__(self, S, G, rng, depth_net=None, gt_index=None):
        super().__init__(S, G, rng, depth_net=depth_net)
        self.n_act = 0; self.kcount = dict(yaw=0, fwd=0, up=0, lat=0); self.t_flight = 0.0; self.last_blocked = False
        self.gt_index = gt_index; self.viewset = set(); self.n_back = 0; self.n_fence = 0

    def step(self, kind, mag):
        p, y = self.S.pose(); cp = lambda q: q + np.array([0, CAM_H, 0]); self.traj.append(cp(p).copy()); self.last_blocked = False
        if kind == "yaw": self.S.set_pose(p, (y + mag + self.rng.normal(0, self.yn)) % 360)
        else:
            m = mag * (1 + self.rng.normal(0, self.fn)); r = np.deg2rad(y)
            vec = {"fwd": np.array([-np.sin(r), 0, -np.cos(r)]), "lat": np.array([-np.cos(r), 0, np.sin(r)]), "up": np.array([0, 1.0, 0])}[kind]
            q = p + vec * m; c = cp(q)
            inside = bool(((c >= self.G.bounds[0] + 0.3) & (c <= self.G.bounds[1] - 0.3)).all())  # geofence: scans have holes to the void outside
            if inside and self.G.segment_free(cp(p), c, RADIUS): self.S.set_pose(q, y); self.dist += abs(m)
            else: self.coll += 1; self.last_blocked = True; self.n_fence += int(not inside)
            self.n_back += int(kind == "fwd" and mag < 0)
        self.n_act += 1; self.kcount[kind] += 1
        self.t_flight += abs(mag) / (1.0 if kind in ("fwd", "lat") else 90.0 if kind == "yaw" else 0.5)

    def sense(self):
        o = self.S.sim.get_sensor_observations(); rgb = o["rgb"][..., :3]; dtrue = o["depth"]; d = dtrue.copy(); v = d > 0.05
        if self.depth_net is not None:
            dp = self.depth_net(rgb); self.absrel.append(float(np.median(np.abs(dp[v] - d[v]) / d[v]))) if v.any() else None; d = dp
        else: d[v] *= 1 + self.rng.normal(0, self.dn, d.shape)[v]
        p, y = self.S.pose(); R = M.yaw_R(y)
        f = 128 / np.tan(np.deg2rad(getattr(self.S, "hfov", 90)) / 2); u, vv = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4))
        r4 = np.stack([(u - 128) / f, -(vv - 128) / f, -np.ones_like(u, float)], -1).reshape(-1, 3)
        d4 = dtrue[::4, ::4].reshape(-1); m = (d4 > .05) & (d4 < 8); w = (r4[m] * d4[m, None]) @ R.T + p + [0, CAM_H, 0]
        self.seen |= set(map(tuple, np.floor(w / 0.25).astype(np.int32)))
        if self.gt_index:
            ws = w[::8]; b = M.view_bins(p + [0, CAM_H, 0] - ws)
            for vx, bb in zip(map(tuple, np.floor(ws / 0.25).astype(np.int32)), b):
                gi = self.gt_index.get(vx)
                if gi is not None: self.viewset.add(gi * 24 + int(bb))
        return rgb, d

    def reset(self, seed=None):
        """Start pose. Seeding habitat's pathfinder makes the start depend only on the seed (paired comparisons across configs)."""
        self.G.ensure_main(self.S.sim.pathfinder)
        if seed is not None: self.S.sim.pathfinder.seed(int(seed) + 12345)
        return super().reset()

    def fence(self):
        """Flight-area box handed to the agent (scene bounds shrunk by 0.3 m); the env blocks motion outside it."""
        return (self.G.bounds[0] + 0.3, self.G.bounds[1] - 0.3)

    def view_stats(self):
        if not self.gt_index: return {}
        vc = np.bincount(np.fromiter((k // 24 for k in self.viewset), int, len(self.viewset)), minlength=len(self.gt_index))
        return {f"views_ge{q}": float((vc >= q).mean()) for q in (1, 3, 6)}


# ===================================================================================================================== agent
class SensorAgent5(SensorAgent2):
    def __init__(self, env, seed, pose_mode="odom", correct=False, mapper="logodds", vo_kw=None):
        super().__init__(env, seed, pose_mode, correct)  # mode "vo": SensorAgent2 does not integrate; act() below does
        self.pose_mode = pose_mode; self.mapper = mapper
        self.vo = M.DepthOdometry(self.f, **(vo_kw or {})) if pose_mode == "vo" else None
        self._vo_pending = None; self.visits = collections.Counter(); self.n_obs = 0; self.n_bump = 0; self.safe_lat = 0.30; self.safe_extra = 0.25; self.noret_unknown = True
        self.fwm = None; self.novelty = {}; self._prev_e = None; self._last_act = None; self.blocked_est = False; self.n_blocked_est = 0; self._gt_pending = None

    def init(self, p, y, fence=None):
        """``fence``: operator-provided flight-area box (world frame, as on a real indoor deployment). Cells outside it are
        permanent obstacles in the agent's map and are never carved free."""
        super().init(p, y); self.z0 = float(self.pos[1])
        if self.mapper == "logodds": self.grid = M.LogOddsGrid(self.shape); self.occ = self.grid.occ
        self.vb = M.ViewBits(self.shape); self.inside = None; self.keep = np.zeros(self.shape, bool)
        if fence is not None:
            ax = [self.origin[i] + (np.arange(self.shape[i]) + 0.5) * GRID for i in range(3)]
            ins = [(a >= fence[0][i]) & (a <= fence[1][i]) for i, a in enumerate(ax)]
            self.inside = ins[0][:, None, None] & ins[1][None, :, None] & ins[2][None, None, :]
            self.occ[~self.inside] = M.OCC
            if self.mapper == "logodds": self.grid.L[~self.inside] = self.grid.l_max; self.grid.seen[~self.inside] = True

    # ---- control ----
    def act(self, kind, mag):
        pre = self.pos.copy()
        ok = super().act(kind, mag)
        if ok: self._last_act = (kind, mag)
        if ok and self.mode == "gt" and kind != "yaw": self._gt_pending = (pre, kind, mag)
        if ok and self.pose_mode == "vo": self._integrate(kind, mag, pre)
        return ok

    def _integrate(self, kind, mag, pre):
        """Commanded motion as the odometry prior; ICP corrects it in observe()."""
        if kind == "yaw": self.yaw = (self.yaw + mag) % 360
        elif kind == "fwd": self.pos = self.pos + self.fwd() * mag
        elif kind == "lat": self.pos = self.pos + self.left() * mag
        else: self.pos = self.pos + np.array([0, mag, 0])
        self._vo_pending = (pre, kind != "yaw", kind, mag)

    def _after_move(self, kind, mag):
        """Pose bookkeeping for a move issued directly to the env (nudge), mirroring act()."""
        pre = self.pos.copy()
        if self.pose_mode == "vo": self._integrate(kind, mag, pre)
        elif self.mode == "odom":
            if kind == "fwd": self.pos = self.pos + self.fwd() * mag
            elif kind == "lat": self.pos = self.pos + self.left() * mag
            elif kind == "up": self.pos = self.pos + np.array([0, mag, 0])
            else: self.yaw = (self.yaw + mag) % 360

    # ---- perception ----
    def observe(self, rgb, d):
        self.n_obs += 1
        if self.pose_mode == "vo":
            if self.vo.kf is None: self.vo.set_keyframe(d, self.pos, self.yaw)
            elif self._vo_pending is not None:
                pre, tr, kind, mag = self._vo_pending
                self.pos, self.yaw, info = self.vo.track(d, self.pos, self.yaw, pre, translated=tr)
                if tr:  # progress watchdog: the estimated motion along the command is < 30 % of it -> the move was blocked
                    vec = {"fwd": self.fwd() * np.sign(mag), "lat": self.left() * np.sign(mag), "up": np.array([0, np.sign(mag), 0])}[kind]
                    moved = float((self.pos - pre) @ vec)
                    if info.get("hyp") == 1 or (info.get("ok") and moved < 0.3 * abs(mag)):
                        self.blocked_est = True; self.n_blocked_est += 1
                        if self.mapper == "logodds": self._bump(np.floor((self.pos + vec * 0.35 - self.origin) / GRID).astype(int))
            self._vo_pending = None
        elif self.pending is not None and self.mode == "odom":  # v4 bump correction (pose "corr")
            pre, prev, mag, fwd = self.pending; self.pending = None
            a, b = prev[88:168, 88:168], d[88:168, 88:168]; m = (a > 0.3) & (a < 4.5) & (b > 0.05)
            reach = np.percentile(a[a > 0.05], 5) < mag + 0.6 if (a > 0.05).any() else False
            if m.sum() > 200:
                t_hat = float(np.median((a - b)[m])); det = reach and t_hat < 0.4 * mag
                self.det[("tp" if det else "fn") if self.env.last_blocked else ("fp" if det else "tn")] += 1
                if det and self.correct:
                    self.pos = pre; idx = np.clip(np.floor((pre + fwd * 0.5 - self.origin) / GRID).astype(int), 0, self.shape - 1)
                    if self.mapper == "logodds": self._bump(idx)
                    else: self.occ[tuple(idx)] = 2
        if self.mode == "gt":
            p, y = self.env.S.pose(); self.pos = p + np.array([0, CAM_H, 0]); self.yaw = y
            if self._gt_pending is not None:  # same progress watchdog as with odometry (here the pose is exact)
                pre, kind, mag = self._gt_pending; self._gt_pending = None
                vec = {"fwd": self.fwd() * np.sign(mag), "lat": self.left() * np.sign(mag), "up": np.array([0, np.sign(mag), 0])}[kind]
                if float((self.pos - pre) @ vec) < 0.3 * abs(mag):
                    self.blocked_est = True; self.n_blocked_est += 1
                    if self.mapper == "logodds": self._bump(np.floor((self.pos + vec * 0.35 - self.origin) / GRID).astype(int))
        if self.fwm is not None:  # 4a: model-uncertainty novelty of the transition just experienced, stored in the visit archive
            e = self.fwm.embed(rgb, d)
            if self._prev_e is not None and self._last_act is not None:
                nov = self.fwm.step(self._prev_e, *self._last_act, e); c = np.floor((self.pos - self.origin) / GRID).astype(int)
                key = self.visit_key(c, self.yaw)[:3]; self.novelty[key] = 0.5 * self.novelty.get(key, nov) + 0.5 * nov  # per 1 m cell x 0.5 m altitude
            self._prev_e = e; self._last_act = None
        self._map(d)

    def _bump(self, idx):
        """Obstacle inferred from a blocked move (odometry), often outside the camera frustum (below / beside the body): it is
        made permanent, so later free-space rays that pass next to it cannot erase it."""
        idx = np.clip(np.asarray(idx, int), 0, self.shape - 1)
        self.grid.update(np.zeros((0, 3), int), idx[None]); self.grid.L[tuple(idx)] = self.grid.l_max; self.keep[tuple(idx)] = True; self.n_bump += 1

    def _map(self, d):
        R = M.yaw_R(self.yaw); dd = d[::8, ::8].reshape(-1); ok = dd > 0.05; dd = np.where(ok, dd, 6.0); hit = ok & (dd < 6.0); dirs = self.rays @ R.T
        shp = np.array(self.shape)
        if self.mapper == "logodds" and self.noret_unknown: dd = np.where(ok, dd, 0.0)  # no return (glass, holes, out of range noise): unknown, never carved free
        free, hidx, hp = M.ray_cells(self.pos, dirs, dd, hit, GRID, self.origin, free_margin=0.1)
        if self.inside is not None:
            ok = ((free >= 0) & (free < shp)).all(1); free = free[ok]; free = free[self.inside[tuple(free.T)]]
        if self.n_bump:
            ok = ((free >= 0) & (free < shp)).all(1); free = free[ok]; free = free[~self.keep[tuple(free.T)]]
        if self.mapper == "logodds":
            touched = self.grid.update(free, hidx)
            if len(touched):
                t = np.stack(np.unravel_index(touched, self.shape), 1); self.bb_lo = np.minimum(self.bb_lo, t.min(0)); self.bb_hi = np.maximum(self.bb_hi, t.max(0))
        else:  # v4 sticky map
            for idx in (free,):
                idx = idx[((idx >= 0) & (idx < shp)).all(1)]
                if len(idx): self.occ[tuple(idx.T)] = np.where(self.occ[tuple(idx.T)] == 2, 2, 1); self.bb_lo = np.minimum(self.bb_lo, idx.min(0)); self.bb_hi = np.maximum(self.bb_hi, idx.max(0))
            h = hidx[((hidx >= 0) & (hidx < shp)).all(1)]
            if len(h): self.occ[tuple(h.T)] = 2; self.bb_lo = np.minimum(self.bb_lo, h.min(0)); self.bb_hi = np.maximum(self.bb_hi, h.max(0))
        self.vb.add(hidx, hp, self.pos)
        c = np.floor((self.pos - self.origin) / GRID).astype(int); self.oob += int(((c < 0) | (c >= shp)).any())
        self.visits[self.visit_key(c, self.yaw)] += 1
        self._depth = d; self.est_traj.append(self.pos.copy())

    def corridor_clear(self, dist, axis="fwd"):
        """Depth-only safety check (as v4) with configurable margins; RGB-only runs widen them for monocular depth error."""
        if axis != "fwd": return super().corridor_clear(dist, axis)
        d = self._depth[::4, ::4]; u, v = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4)); ok = d > 0.05
        x = (u - 128) / self.f * d; y = -(v - 128) / self.f * d
        box = ok & (np.abs(x) < self.safe_lat) & (np.abs(y) < self.safe_lat) & (d < dist + self.safe_extra)
        return not box.any()

    @staticmethod
    def visit_key(c, yaw): return (int(c[0]) // 4, int(c[2]) // 4, int(c[1]) // 2, int(yaw // 45) % 8)


# ===================================================================================================================== hierarchical planner
class HierPlanner:
    """Frontier clusters -> one viewpoint each (max visible frontier cells, FUEL-style) -> open-TSP order over exact shortest
    paths on the agent's map -> follow the path to the first viewpoint. Optional view-diversity mode when no frontier is left."""

    def __init__(self, sel="tsp", lam=0.04, div=False, div_beta=1.0, yaw_w=2.0, k_max=12, replan_every=30, value_fn=None, band=(-1.5, 2.0), seed=0, infl=1, kappa=0.4):
        self.sel, self.lam, self.div, self.div_beta, self.yaw_w, self.k_max, self.replan_every = sel, lam, div, div_beta, yaw_w, k_max, replan_every
        self.value_fn = value_fn; self.band = band; self.rng = np.random.default_rng(seed + 77)
        self.path = []; self.goal = None; self.target_yaw = None; self.bad = set(); self.stuck = 0; self.replan_in = 0
        self.cluster = None; self.mode = None; self.nplan = 0; self.plan_ms = 0.0; self.n_div = 0; self.n_front = 0; self.n_fail = 0; self.n_opt = 0
        self.known_at_clear = 0; self.feat_log = None; self.done_vp = set(); self.fail_wait = 0; self.infl = infl; self.n_noinfl = 0; self.n_vp = 40; self.vbad = set(); self.kappa = kappa; self.dead = set(); self.check_dead = None; self.n_dead = 0; self.hyst = True; self.n_hyst = 0; self.wm_gamma = 1.0; self.narrow = {}; self.now = 0
        self.door = False; self.door_run = 0; self.n_door = 0; self.narrow_seen = set(); self.narrow_ttl = 400; self.narrow_reset = False; self.n_unnarrow = 0  # set from cfg by Explorer

    # ---- helpers ----
    def _crop(self, A):
        lo = np.maximum(A.bb_lo - 6, 0); hi = np.minimum(A.bb_hi + 7, np.array(A.shape)); O = A.occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        start = tuple(np.clip(np.floor((A.cam() - A.origin) / GRID).astype(int) - lo, 0, np.array(O.shape) - 1))
        ys = (A.origin[1] + (np.arange(O.shape[1]) + lo[1] + 0.5) * GRID) - A.z0
        band = np.zeros(O.shape, bool); band[:, (ys >= self.band[0]) & (ys <= self.band[1]), :] = True
        return lo, O, start, band

    def _graph(self, O, start, band, optimistic=False, lo=None, infl=0):
        trav = ((O == M.FREE) | ((O == M.UNKNOWN) & optimistic)) & band
        if infl: trav &= ~M.binary_dilation(O == M.OCC, structure=np.ones((3, 3, 3), bool), iterations=infl)  # keep >= 1 cell clearance
        self.narrow = {k: v for k, v in self.narrow.items() if v > self.now}
        for b in list(self.bad) + list(self.narrow):
            bb = np.array(b) - lo
            if ((bb >= 0) & (bb < np.array(O.shape))).all(): trav[tuple(bb)] = False
        s = start; nr = tuple(slice(max(0, v - 1), v + 2) for v in s); trav[nr] |= (O[nr] == M.FREE)
        trav[s] = True
        near = M.clearance_cost(O)
        if optimistic: near = near + (O == M.UNKNOWN)  # unknown cells cost double
        # vertical edges (y -> y+1) need the agent's vertical safety margin: no occupied cell within one cell laterally from y-2 to y+3
        oh = M.binary_dilation(O == M.OCC, structure=np.ones((3, 1, 3), bool)); vm = np.ones(O.shape, bool)
        for k in (-2, -1, 2, 3):
            sh = np.zeros_like(oh)
            if k > 0: sh[:, :-k] = oh[:, k:]
            else: sh[:, -k:] = oh[:, :k]
            vm &= ~sh
        for (x, z) in self.vbad:
            xx, zz = x - lo[0], z - lo[2]
            if 0 <= xx < O.shape[0] and 0 <= zz < O.shape[2]: vm[xx, :, zz] = False
        G, idx, cells = M.grid_graph(trav, near, GRID, vmask=vm)
        return trav, G, idx, cells

    @staticmethod
    def _yaw_to(a, b):
        v = np.asarray(b, float) - np.asarray(a, float); return float(np.rad2deg(np.arctan2(-v[0], -v[2])) % 360) if np.hypot(v[0], v[2]) > 1e-6 else None

    def _viewpoint(self, O, RC, Dr, cl):
        """Best viewpoint for a frontier cluster among reachable cells 0.5-2.0 m from its centroid: most cluster cells inside the
        camera frustum (+-40 deg) within 4 m with line of sight; ties broken by path cost from the agent."""
        c = cl.mean(0); dh = np.hypot(RC[:, 0] - c[0], RC[:, 2] - c[2]); dv = np.abs(RC[:, 1] - c[1])
        cand = np.where((dh >= 2) & (dh <= 8) & (dv <= 3))[0]
        if len(cand) == 0: cand = np.where((dh <= 3) & (dv <= 3))[0]
        if len(cand) == 0: return None
        if len(cand) > 40: cand = cand[np.linspace(0, len(cand) - 1, 40).astype(int)]
        S = cl[np.linspace(0, len(cl) - 1, min(16, len(cl))).astype(int)]
        P = RC[cand]; V = S[None].astype(float) - P[:, None].astype(float)  # (k, s, 3)
        yaw = np.arctan2(-(c[0] - P[:, 0]), -(c[2] - P[:, 2]))
        az = np.arctan2(-V[..., 0], -V[..., 2]); da = np.abs((az - yaw[:, None] + np.pi) % (2 * np.pi) - np.pi)
        hz = np.hypot(V[..., 0], V[..., 2]); el = np.abs(np.arctan2(V[..., 1], hz + 1e-9)); rng_ = np.linalg.norm(V, axis=-1)
        inf = (da < np.deg2rad(40)) & (el < np.deg2rad(40)) & (rng_ <= 16)
        los = M.los_free(O, np.repeat(P, len(S), 0), np.tile(S, (len(P), 1))).reshape(len(P), len(S))
        vis = (inf & los).sum(1)
        if vis.max() == 0: k = int(np.argmin(Dr[cand]))
        else:
            good = np.where(vis >= 0.8 * vis.max())[0]; k = int(good[np.argmin(Dr[cand[good]])])
        return int(cand[k]), float(vis[k]) / len(S) * len(cl), float(np.rad2deg(yaw[k]) % 360)

    # ---- planning ----
    def plan(self, A):
        t0 = time.time(); self.nplan += 1; self.now = A.n_obs
        try:
            ok = self._plan_frontier(A, optimistic=False, infl=self.infl)
            if not ok and self.infl: ok = self._plan_frontier(A, optimistic=False, infl=0); self.n_noinfl += int(ok)
            if not ok: ok = self._plan_frontier(A, optimistic=True); self.n_opt += int(ok)
            if not ok and self.narrow_reset and self.narrow:  # the only frontiers may lie behind a passage marked narrow: retry it
                self.narrow = {}; self.n_unnarrow += 1; ok = self._plan_frontier(A, optimistic=False, infl=0) or self._plan_frontier(A, optimistic=True)
            if not ok and self.div: ok = self._plan_diversity(A)
            if not ok: self.n_fail += 1
            return ok
        finally: self.plan_ms += 1000 * (time.time() - t0)

    def _plan_frontier(self, A, optimistic, infl=0):
        lo, O, start, band = self._crop(A)
        trav, G, idx, cells = self._graph(O, start, band, optimistic, lo, infl)
        s_id = idx[start]; D0, P0 = M.shortest(G, [s_id]); D0 = D0[0]
        rmask = np.isfinite(D0); RC = cells[rmask]; Dr = D0[rmask]
        reach = np.zeros(O.shape, bool); reach[tuple(RC.T)] = True
        if optimistic: reach &= (O == M.FREE)  # frontier cells themselves must be known free
        if self.dead:
            dl = np.array(list(self.dead)); dl = dl - lo; ok = ((dl >= 0) & (dl < np.array(O.shape))).all(1); dl = dl[ok]
            if len(dl): reach = reach.copy(); reach[tuple(dl.T)] = False  # frontier cells that stayed unresolved after being looked at
        clusters, Fm = M.frontier_clusters(O, reach, band)
        if not clusters: return False
        unk = (O == M.UNKNOWN).astype(np.float32); ucnt = uniform_filter(unk, size=(5, 3, 5), mode="constant", cval=0.0)
        rid = -np.ones(O.shape, np.int64); rid[tuple(RC.T)] = np.arange(len(RC))
        if len(clusters) > self.n_vp:  # viewpoints only for the most promising clusters: size / (1 + path distance to the cluster)
            dmin = np.array([float(Dr[rid[tuple(cl.T)][rid[tuple(cl.T)] >= 0]].min()) if (rid[tuple(cl.T)] >= 0).any() else 1e3 for cl in clusters])
            score = np.array([min(len(cl), 60) for cl in clusters]) / (1.0 + dmin)
            clusters = [clusters[i] for i in np.argsort(-score)[: self.n_vp]]
        cand = []
        for cl in clusters:
            vp = self._viewpoint(O, RC, Dr, cl)
            if vp is None: continue
            k, gain, yaw = vp; g_cell = tuple(int(v) for v in RC[k] + lo)
            if g_cell in self.bad or any(max(abs(g_cell[0] - d[0]), abs(g_cell[1] - d[1]), abs(g_cell[2] - d[2])) <= 1 for d in self.done_vp): continue
            cand.append(dict(cells=cl + lo, vp=g_cell, vp_crop=tuple(RC[k]), gain=gain, size=len(cl), unk=float(ucnt[tuple(cl.T)].mean()), yaw=yaw, cost=float(Dr[k]),
                             dz=float((RC[k][1] - start[1]) * GRID), centroid=cl.mean(0) + lo))
        if not cand: return False
        dyaw = np.array([abs(((self._yaw_to(A.cam(), A.origin + (np.array(c["vp"]) + 0.5) * GRID) or A.yaw) - A.yaw + 180) % 360 - 180) for c in cand])
        for c, dy in zip(cand, dyaw): c["dyaw"] = float(dy)
        if self.value_fn is not None:
            pv = self.value_fn(A, O, lo, cand)
            for c, v in zip(cand, pv): c["value"] = float(v)
        if self.feat_log is not None: self.feat_log.append(dict(A=A, O=O, lo=lo, cand=cand))
        cost0 = np.array([c["cost"] + self.yaw_w * c["dyaw"] / 180.0 for c in cand])
        if self.sel == "greedy":
            sc = np.array([min(c["gain"], 40) / 40.0 for c in cand]) - self.lam * cost0; j = int(np.argmax(sc))
        elif self.sel in ("ratio", "learned", "oracle", "frustum"):  # information rate: (heuristic | learned 3a | privileged oracle) gain / (1 + kappa * cost)
            g = np.array([c["value"] if self.sel != "ratio" else c["gain"] for c in cand]); sc = g / (1.0 + self.kappa * cost0); j = int(np.argmax(sc))
        elif self.sel == "rank":  # 3b: oracle-imitation score already accounts for cost (cost is an input feature)
            j = int(np.argmax([c["value"] for c in cand]))
        else:  # tsp over the k_max most promising clusters
            g = np.array([c["value"] if self.sel == "learned_tsp" else c["gain"] for c in cand])
            keep = np.argsort(-(g / (1.0 + cost0)))[: self.k_max]
            sub = [cand[i] for i in keep]
            src = [s_id] + [idx[c["vp_crop"]] for c in sub]
            D, _ = M.shortest(G, src, limit=float(cost0[keep].max()) * 2 + 5.0); C = np.zeros((len(src), len(src)))
            for a_ in range(len(src)):
                for b_ in range(1, len(src)): C[a_, b_] = D[a_, src[b_]] if a_ != b_ else 0.0
            C[0, 1:] = cost0[keep]; C[~np.isfinite(C)] = 1e6
            order = M.open_tsp(C); j = int(keep[order[1] - 1])
        # hysteresis: keep the current target while its cluster is unresolved and it is not much worse than the new choice
        if self.hyst and self.mode == "frontier" and self.cluster is not None and not optimistic:
            # same target = the candidate whose frontier cells overlap most with the current target's (its best viewpoint may move)
            prev = set(np.ravel_multi_index(np.asarray(self.cluster).T, A.shape).tolist())
            ov = [len(prev & set(np.ravel_multi_index(np.asarray(c["cells"]).T, A.shape).tolist())) / max(1, len(prev)) for c in cand]
            pj = int(np.argmax(ov))
            if ov[pj] >= 0.3 and pj != j and cost0[pj] <= 1.5 * cost0[j] + 1.0: j = pj; self.n_hyst += 1
        c = cand[j]; P0r = P0[0]
        self.path = [tuple(int(v) for v in np.array(q) + lo) for q in M.path_from(P0r, idx[c["vp_crop"]], cells)]
        self.goal = c["vp"]; self.target_yaw = c["yaw"]; self.cluster = c["cells"]; self.mode = "frontier"; self.n_front += 1
        self.replan_in = self.replan_every; self.cur = c
        return True

    def _plan_diversity(self, A):
        """No frontier left: go to the reachable pose that would observe the most (voxel, view-direction) pairs not yet in the
        agent's own map, discounted by path cost and boosted for rarely visited (cell, altitude, heading) archive entries."""
        lo, O, start, band = self._crop(A)
        trav, G, idx, cells = self._graph(O, start, band, False, lo, self.infl)
        D0, P0 = M.shortest(G, [idx[start]]); D0 = D0[0]; rm = np.isfinite(D0); RC = cells[rm]; Dr = D0[rm]
        if len(RC) < 2: return False
        OC = np.argwhere(O == M.OCC)
        if len(OC) == 0: return False
        if len(OC) > 3000: OC = OC[self.rng.choice(len(OC), 3000, replace=False)]
        bits = A.vb.B[tuple((OC + lo).T)]
        k = min(64, len(RC)); w = np.clip(Dr, 0.5, None); cand = self.rng.choice(len(RC), k, replace=False, p=w / w.sum())
        P = RC[cand].astype(float) + 0.5; V = OC[None].astype(float) + 0.5 - P[:, None]  # cell -> point vectors (k, n, 3)
        rng_ = np.linalg.norm(V, axis=-1); hz = np.hypot(V[..., 0], V[..., 2]); el = np.abs(np.arctan2(V[..., 1], hz + 1e-9))
        az = np.arctan2(-V[..., 0], -V[..., 2]); vbin = M.view_bins(-V); new = ((bits[None] >> vbin.astype(np.uint32)) & 1) == 0
        base = new & (rng_ <= 20) & (rng_ >= 2) & (el < np.deg2rad(45))
        best = None
        for yb in range(8):
            yaw = np.deg2rad(yb * 45.0); da = np.abs((az - yaw + np.pi) % (2 * np.pi) - np.pi)
            nov = (base & (da < np.deg2rad(45))).sum(1).astype(float)
            keys = [SensorAgent5.visit_key(RC[cand[i]] + lo, yb * 45.0) for i in range(k)]
            vis = np.array([A.visits.get(q, 0) for q in keys])
            sc = nov * (1 + self.div_beta / np.sqrt(1 + vis)) / (1 + 0.1 * Dr[cand])
            if A.fwm is not None and A.fwm.mean_nov:  # 4a: revisit places where the forward model is still uncertain
                mn = A.fwm.mean_nov; sc = sc * (1 + self.wm_gamma * np.array([min(A.novelty.get(q[:3], mn) / mn, 3.0) for q in keys]))
            i = int(np.argmax(sc))
            if best is None or sc[i] > best[0]: best = (sc[i], i, yb * 45.0, nov[i])
        if best is None or best[3] < 5: return False
        _, i, yaw, nov = best
        # verify line of sight on a sample of the counted cells; reject if mostly occluded
        tgt = RC[cand[i]]
        self.path = [tuple(int(v) for v in np.array(q) + lo) for q in M.path_from(P0[0], idx[tuple(tgt)], cells)]
        self.goal = tuple(int(v) for v in tgt + lo); self.target_yaw = float(yaw); self.cluster = None; self.mode = "diversity"; self.n_div += 1
        self.replan_in = self.replan_every; self.cur = dict(nov=float(nov))
        return True

    def target_resolved(self, A, thr=0.3):
        if self.mode != "frontier" or self.cluster is None: return False
        c = self.cluster; shp = np.array(A.shape); cnt = 0
        for d in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)):
            q = np.clip(c + np.array(d), 0, shp - 1); cnt = cnt + (A.occ[tuple(q.T)] == M.UNKNOWN)
        still = (cnt > 0) & (A.occ[tuple(c.T)] != M.OCC)
        return still.mean() < thr


DOOR_HALF = RADIUS + 0.03  # door mode tube half-width: true body radius plus a depth-noise allowance (normal mode uses 0.30 + 0.25 m ahead)


def tube_clear(A, dist, half):
    """Depth-only check of a straight square tube of half-width ``half`` reaching ``dist`` + 0.1 m ahead of the camera."""
    d = A._depth[::4, ::4]; u, v = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4)); ok = d > 0.05
    x = (u - 128) / A.f * d; y = -(v - 128) / A.f * d
    return not (ok & (np.abs(x) < half) & (np.abs(y) < half) & (d < dist + 0.1)).any()


def ray_free_map(A, dist):
    """No occupied cell of the agent's own map on the straight forward ray."""
    for s_ in np.arange(0.1, dist + 1e-6, 0.1):
        i = tuple(np.clip(np.floor((A.pos + A.fwd() * s_ - A.origin) / GRID).astype(int), 0, A.shape - 1))
        if A.occ[i] == M.OCC: return False
    return True


def side_free_map(A, vec, dist, half=0.15):
    """Lenient map check for a short move along ``vec``: no occupied cell of the agent's map on the path or ``half`` m to either side."""
    side = np.array([-vec[2], 0, vec[0]])
    for s_ in np.arange(0.05, dist + 0.2 + 1e-6, 0.05):
        for o in (-half, 0.0, half):
            i = tuple(np.clip(np.floor((A.pos + vec * s_ + side * o - A.origin) / GRID).astype(int), 0, A.shape - 1))
            if A.occ[i] == M.OCC: return False
    return True


def door_step(A, F, v):
    """Door / narrow-passage traversal (proposal 1). Called when the normal corridor check (0.30 m half-width, 0.25 m extra reach)
    refuses to move. Steps forward when a body-sized tube (depth) and the agent's own map are clear; otherwise finds the opening in
    the depth image near the path direction and centres on it (sideways step when close to the frame, yaw when further away).
    Returns a status, or None when there is no passable opening (the caller then marks the passage as too narrow)."""
    if F.door_run >= 24: F.door_run = 0; return None  # give up this opening for now
    right = -A.left(); xp, zp = float(v @ right), float(v @ A.fwd())
    if zp <= 0: return None  # the path does not go ahead
    if F.door_run == 0 and F.door >= 2:  # door=2, recovery only: force a passage the agent already refused once (its narrow mark expired)
        here = {tuple(int(x) for x in F.path[0]), tuple(np.floor((A.pos + A.fwd() * 0.4 - A.origin) / GRID).astype(int).tolist())}
        if not (here & F.narrow_seen): return None
    if tube_clear(A, A_FWD_S, DOOR_HALF) and ray_free_map(A, A_FWD_S) and side_free_map(A, A.fwd(), A_FWD_S) and A.act("fwd", A_FWD_S):
        F.door_run += 1; F.n_door += 1; return "door"
    g = M.depth_gap(A._depth, A.f)
    if g is None: return None
    xl, xr, d_front, hard_l, hard_r = g; xc = 0.5 * (xl + xr)
    if hard_l and hard_r and xr - xl < 2 * DOOR_HALF + 0.02: return None  # both edges seen and the body does not fit
    if xr - xl < 0.2: return None
    if abs(np.arctan2(xc, max(d_front, 0.2)) - np.arctan2(xp, zp)) > np.deg2rad(50): return None  # opening is not where the path goes
    F.door_run += 1; F.n_door += 1
    if abs(xc) > 0.05 and d_front < 0.6:  # close to the frame: centre sideways
        m = float(np.clip(-xc, -A_LAT, A_LAT)); vec = A.left() * np.sign(m)
        if side_free_map(A, vec, abs(m)) and A.act("lat", m): return "door"
    ang = float(np.rad2deg(np.arctan2(xc, max(d_front, 0.3))))
    if abs(ang) > 3: A.act("yaw", float(np.clip(-ang, -A_TURN_L, A_TURN_L))); return "door"
    F.door_run -= 1; F.n_door -= 1
    return None


def follow(A, F, translate=False, align_tol=8.0):
    """One control action along F.path. Returns a status string (same vocabulary as explore2.frontier_step)."""
    cp = A.cam()
    for _ in range(8):
        if not F.path: return "arrived"
        if len(F.path) > 1:  # string pulling over the agent's own map
            for k in range(min(len(F.path), 16) - 1, 0, -1):
                if seg_free_map(A, cp, A.origin + (np.array(F.path[k]) + 0.5) * GRID): del F.path[:k]; break
        wp = A.origin + (np.array(F.path[0]) + 0.5) * GRID; v = wp - cp
        if np.linalg.norm(v) < 0.15 or tuple(np.floor((cp - A.origin) / GRID).astype(int)) == tuple(F.path[0]): F.path.pop(0); continue
        break
    else: return "wp"
    hd = float(np.hypot(v[0], v[2]))
    if abs(v[1]) > 0.15 and hd < 0.3:
        ok = A.act("up", float(np.sign(v[1]) * A_UP)); st = "vert"
    else:
        if getattr(F, "door", False) and F.door_run > 0:  # in the middle of a door traversal: do not let path alignment undo it
            st_ = door_step(A, F, v)
            if st_ is not None: return st_
        want = np.rad2deg(np.arctan2(-v[0], -v[2])) % 360 if hd > 0.1 else A.yaw; dy = (want - A.yaw + 180) % 360 - 180
        if translate and abs(dy) > align_tol and hd < 0.6:  # small sideways correction instead of a turn (never backwards)
            lat = float(np.array([v[0], 0, v[2]]) @ A.left()) / max(hd, 1e-6)
            if abs(lat) > 0.7 and A.act("lat", float(np.sign(lat) * min(A_LAT, hd))): F.stuck = 0; return "slide"
        if abs(dy) > align_tol: A.act("yaw", float(np.clip(dy, -A_TURN_L, A_TURN_L))); return "turn"
        step = next((s for s in (A_FWD_L, A_FWD_S) if s <= hd + 0.1 and A.map_clear(s)), None)
        if step is None and A.map_clear(A_FWD_S): step = A_FWD_S
        if step is None and getattr(F, "door", False):  # the normal check refuses: try the door traversal before giving up
            st_ = door_step(A, F, v)
            if st_ is not None: return st_
        if step is None and hasattr(F, "narrow"):  # depth / map refuse even a small step: remember the passage as too narrow for a while
            ttl = int(getattr(F, "narrow_ttl", 400))
            for q in (F.path[0], tuple(np.floor((cp + A.fwd() * 0.4 - A.origin) / GRID).astype(int))):
                F.narrow[tuple(int(v) for v in q)] = A.n_obs + ttl; F.narrow_seen.add(tuple(int(v) for v in q))
        ok = step is not None and A.act("fwd", step); st = "move"
        if ok and hasattr(F, "door_run"): F.door_run = 0
        if ok and abs(v[1]) > 0.15: A.act("up", float(np.sign(v[1]) * A_UP))
    if not ok:
        F.stuck += 1
        if st == "vert" and hasattr(F, "vbad"):  # vertical move refused here: no vertical edges in this column until the map grows
            c = np.floor((cp - A.origin) / GRID).astype(int); F.vbad.add((int(c[0]), int(c[2]))); F.vbad.add((int(F.path[0][0]), int(F.path[0][2])))
            F.path = []; F.stuck = 0; F.replan_in = 0
        elif st == "move" and nudge(A): pass  # depth says something is inside the flight corridor: side-step away from it
        elif F.stuck > 1: F.bad.add(tuple(F.path[0])); F.path = []; F.stuck = 0; F.replan_in = 0
    else: F.stuck = 0
    return st


def nudge(A, step=0.15):
    """Forward move refused by the depth corridor check: look at which side of the corridor the offending depth points are and
    translate 0.15 m away from them (lateral, else vertical), checked against the agent's own map. Depth-only, real-drone feasible."""
    d = A._depth[::4, ::4]; u, v = np.meshgrid(np.arange(0, 256, 4), np.arange(0, 256, 4)); ok = d > 0.05
    x = (u - 128) / A.f * d; y = -(v - 128) / A.f * d
    box = ok & (np.abs(x) < 0.30) & (np.abs(y) < 0.30) & (d < A_FWD_S + 0.25)
    if not box.any(): box = ok & (np.abs(x) < 0.30) & (np.abs(y) < 0.30) & (d < A_FWD_L + 0.25)
    if not box.any(): return False
    mx, my = float(x[box].mean()), float(y[box].mean())
    opts = []
    if abs(mx) > 0.03: opts.append(("lat", step * (1 if mx > 0 else -1) * 1.0))  # obstacle on the right (+x) -> move left (+lat)
    if abs(my) > 0.03: opts.append(("up", -A_UP if my > 0 else A_UP))
    if abs(mx) >= abs(my): opts = opts[:1] + opts[1:]
    else: opts = opts[::-1]
    for k, m in opts:
        if not (A.move_clear(A.left() * np.sign(m), abs(m)) if k == "lat" else A.vert_clear(m)): continue  # standard map-based checks for unseen directions
        A.env.step(k, m); A._after_move(k, m)
        A.env.__dict__.setdefault("n_nudge", 0); A.env.n_nudge += 1; return True
    return False


def pol_wander5(A):
    """Fallback when the planner has nothing: like explore3.pol_wander but backing off only when boxed in (the official
    IndoorUAV trajectories contain no backward flight); altitude change is preferred over reversing."""
    r = A.rng.random()
    if A.free_dist() >= 0.9 and r < 0.8: A.act("fwd", A_FWD_L); return
    if A.free_dist() >= 0.15 and r < 0.9: A.act("fwd", A_FWD_S); return
    opts = [("lat", sgn * A_LAT) for sgn in (1, -1) if A.move_clear(A.left() * sgn, A_LAT)]
    if opts and A.rng.random() < 0.7:
        k, m = opts[int(A.rng.integers(len(opts)))]
        if A.act(k, m): return
    if not opts and A.rng.random() < 0.5:
        dz = float(A.rng.choice([-1, 1]) * A_UP)
        if A.vert_clear(dz) and A.act("up", dz): return
        if A.move_clear(-A.fwd(), A_LAT) and A.rng.random() < 0.3 and A.act("fwd", -A_LAT): return
    A.act("yaw", float(A.rng.choice([-A_TURN_L, A_TURN_L])))


FALLBACK = dict(random=pol_random, wander=pol_wander, wander5=pol_wander5)


class Explorer:
    """Policy wrapper: one call = at most one executed action (queued yaw alignments are spread over iterations so the map sees them)."""

    def __init__(self, cfg, seed=0, value_fn=None):
        self.cfg = cfg; self.q = collections.deque(); self.stc = collections.Counter(); self.escapes = 0; self.n_watchdog = 0; self.esc_t = []; self.n_recover = 0
        pl = cfg["plan"]
        if pl == "hier": self.F = HierPlanner(cfg.get("sel", "tsp"), cfg.get("lam", 0.04), cfg.get("div", False), cfg.get("div_beta", 1.0), value_fn=value_fn, seed=seed, infl=int(cfg.get("infl", 1)), kappa=float(cfg.get("kappa", 0.4)), yaw_w=float(cfg.get("yaw_w", 2.0)))
        if pl == "hier":  # proposals 1-2: door traversal, narrow-passage blacklist lifetime / reset (defaults keep the published behaviour)
            self.F.door = int(cfg.get("door", 0)); self.F.narrow_ttl = int(cfg.get("narrow_ttl", 400)); self.F.narrow_reset = bool(cfg.get("narrow_reset", 0))
        elif pl == "f3": self.F = Frontier3(cfg.get("lam", 0.04)); self._wrap()
        elif pl == "v1": self.F = Frontier(); self._wrap()
        else: self.F = None
        if self.F is not None and pl != "hier":
            self.F.vfront = bool(cfg.get("vert")) and pl == "v1"
            if cfg.get("translate"): self.F.translate = True; self.F.fallback = pol_wander
        self.alt = None

    def _wrap(self):
        F = self.F; F.nplan = 0; raw = F.plan
        def _p(A_, F=F, raw=raw): F.nplan += 1; return crop_plan(F, A_, raw)
        F.plan = _p

    def step(self, A, idle):
        cfg = self.cfg; F = self.F
        if A.blocked_est:  # the agent's own odometry says the last move went nowhere: drop the manoeuvre / path and re-plan around the bump
            A.blocked_est = False; self.alt = None; self.q.clear(); self.n_watchdog += 1
            if F is not None:
                if getattr(F, "path", None): F.bad.add(tuple(F.path[0]))
                F.path = []; F.replan_in = 0
        if self.q:
            k, m = self.q.popleft(); A.act(k, m); return "queued"
        if self.alt is not None:  # altitude manoeuvre (collection behaviour / v4 vertical option)
            dz = self.alt - A.pos[1]
            if abs(dz) < 0.13 or not A.act("up", float(np.sign(dz) * A_UP)):
                self.alt = None
                if cfg["plan"] != "hier": F.path = []; F.replan_in = 0
            return "alt"
        if cfg.get("p_alt_step", 0) and A.rng.random() < cfg["p_alt_step"]: self.alt = pick_alt(A, A.z0)
        if idle >= 4 and F is not None:  # stall breaker (same as v4)
            A.act("yaw", float(A.rng.choice([-1, 1]) * A_TURN_L)); self.escapes += 1; F.path = []; F.replan_in = 0
            self.esc_t.append(A.env.n_act)
            if isinstance(F, HierPlanner) and sum(1 for e in self.esc_t[-3:] if A.env.n_act - e <= 40) >= 3:
                # recovery: repeated stalls around the same target -> give it up and wander away for a few actions (as v4 f3 does)
                if F.goal is not None: F.done_vp.add(F.goal)
                F.fail_wait = 12; self.n_recover += 1; self.esc_t = []
            if idle >= 12 and not isinstance(F, HierPlanner): F.bad = set()  # hier clears its blacklist when the map grows instead
            return "escape"
        if F is None: pol_random(A); return "random"
        if cfg["plan"] != "hier":
            st = frontier_step(A, F)
            if st == "arrived":
                if cfg.get("vert") and A.rng.random() < cfg.get("p_alt", 0.4): self.alt = pick_alt(A, A.z0)
                if F.target_yaw is not None: self._queue_yaw(A, F.target_yaw)
                F.path = []; F.replan_in = 0
            self.stc[st] += 1; return st
        # ---- hierarchical planner ----
        known = int((A.occ != 0).sum()) if A.n_obs % 25 == 0 else None
        if known is not None and known > F.known_at_clear * 1.05 + 50:  # map grew: retry old targets
            F.bad = set(); F.done_vp = set(); F.vbad = set(); F.known_at_clear = known
            if F.narrow_reset: F.narrow = {}
        if F.check_dead is not None:  # arrived and looked: a frontier that is still mostly unresolved cannot be observed (window, mirror, scan hole)
            cl = F.check_dead; F.check_dead = None; F.cluster = cl; F.mode = "frontier"
            if cfg.get("dead", 0) and not F.target_resolved(A, 0.5): F.dead.update(map(tuple, cl.tolist())); F.n_dead += 1
        F.replan_in -= 1
        if F.fail_wait > 0:  # planner found nothing recently: wander a few actions before trying again
            F.fail_wait -= 1; self.stc["no_frontier"] += 1
            FALLBACK[cfg.get("fallback", "wander5")](A)
            return "no_frontier"
        if F.replan_in <= 0 or not F.path or (A.n_obs % 3 == 0 and F.target_resolved(A)):
            if not F.plan(A):
                F.fail_wait = 8; self.stc["no_frontier"] += 1
                FALLBACK[cfg.get("fallback", "wander5")](A)
                return "no_frontier"
        st = follow(A, F, cfg.get("translate", False), float(cfg.get("align_tol", 8.0)))
        if st == "arrived":
            if F.target_yaw is not None: self._queue_yaw(A, F.target_yaw)
            if F.goal is not None: F.done_vp.add(F.goal)
            if F.mode == "frontier" and F.cluster is not None: F.check_dead = F.cluster  # evaluated once the queued look-around is done
            if cfg.get("p_alt", 0) and A.rng.random() < cfg.get("p_alt", 0): self.alt = pick_alt(A, A.z0)
            F.path = []; F.replan_in = 0
        self.stc[st] += 1
        return st

    def _queue_yaw(self, A, target):
        dy = (target - A.yaw + 180) % 360 - 180; n = 0
        if abs(dy) < float(self.cfg.get("arrive_tol", 8.0)): return  # target already inside the field of view
        while abs(dy) >= 8 and n < 12:
            s = float(np.clip(dy, -A_TURN_L, A_TURN_L)); self.q.append(("yaw", s)); dy -= s; n += 1


# ===================================================================================================================== rollout
def rollout5(cfg, S, G, steps, seed, nav_cells, name="", depth_net=None, dist_budget=1e9, cov_stop=2.0, gt_index=None, value_fn=None, feat_log=None):
    os.environ["PLAN_DIL"] = str(int(cfg.get("dil", 0)))  # v4 planners read their obstacle inflation from this variable (v4 report: 1)
    rng = np.random.default_rng(seed); env = Env5(S, G, rng, depth_net=depth_net, gt_index=gt_index)
    A = SensorAgent5(env, seed, cfg["pose"] if cfg["pose"] != "corr" else "odom", cfg["pose"] == "corr", cfg.get("map", "logodds"))
    if cfg.get("wm"): A.fwm = M.ForwardEnsemble(seed=seed)
    p, y = env.reset(None if cfg.get("unseeded") else seed); A.init(p, y, env.fence() if cfg.get("fence", 1) else None); A.relax = True; A.safe_lat = float(cfg.get("safe_lat", 0.30)); A.safe_extra = float(cfg.get("safe_extra", 0.25)); A.noret_unknown = bool(cfg.get("noret", 1))
    X = Explorer(cfg, seed, value_fn)
    if feat_log is not None and isinstance(X.F, HierPlanner): X.F.feat_log = feat_log
    curve, dcurve, acurve, tcurve, drift, P, Y, thumbs, yerr = [], [], [], [], [], [], [], [], []
    t = -1; nxt = 0; idle = 0; last = -1; t0 = time.time()
    import cv2
    while env.n_act < steps and env.dist < dist_budget and t < 3 * steps:
        t += 1; rgb, d = env.sense(); A.observe(rgb, d); tp = S.pose(); P.append(tp[0] + [0, CAM_H, 0]); Y.append(tp[1])
        if t % 5 == 0: thumbs.append(cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (16, 16), interpolation=cv2.INTER_AREA).reshape(-1) / 255.0)
        if env.n_act >= nxt:
            nxt += 10; curve.append(env.coverage()); dcurve.append(env.dist); acurve.append(env.n_act); tcurve.append(env.t_flight); drift.append(float(np.linalg.norm(A.pos - P[-1])))
            yerr.append(float(abs((A.yaw - tp[1] + 180) % 360 - 180)))
            if curve[-1] >= cov_stop: break
        idle = idle + 1 if env.n_act == last else 0; last = env.n_act
        X.step(A, idle)
    wall = time.time() - t0; F = X.F
    out = dict(scene=name, cfg=cfg, seed=seed, coverage=env.coverage(), curve=curve, dcurve=dcurve, acurve=acurve, tcurve=tcurve, collisions=env.coll, dist=env.dist,
               actions=env.n_act, kcount=dict(env.kcount), n_back=env.n_back, n_fence=env.n_fence, n_nudge=getattr(env, "n_nudge", 0), iters=t + 1, t_flight=env.t_flight, wall_s=wall, ms_per_step=1000 * wall / max(1, env.n_act),
               final_drift_m=drift[-1], mean_drift_m=float(np.mean(drift)), drift=drift, final_yaw_err=yerr[-1], yaw_err=yerr, escapes=X.escapes, stc=dict(X.stc), oob=int(A.oob),
               div=diversity(P, Y, thumbs, nav_cells), views=env.view_stats(), traj=[list(map(float, q)) for q in np.array(P)[::4]], yaws=[float(v) for v in np.array(Y)[::4]],
               vo=dict(A.vo.stats) if A.vo is not None else None, cleared=int(getattr(getattr(A, "grid", None), "n_cleared", 0)), watchdog=X.n_watchdog, bumps=A.n_bump, recover=X.n_recover)
    if isinstance(F, HierPlanner): out.update(nplan=F.nplan, plan_ms=F.plan_ms / max(1, F.nplan), n_div=F.n_div, n_front=F.n_front, n_fail=F.n_fail, n_opt=F.n_opt, n_noinfl=F.n_noinfl, n_dead=F.n_dead, n_hyst=F.n_hyst, n_door=F.n_door, n_unnarrow=F.n_unnarrow)
    elif F is not None: out.update(nplan=getattr(F, "nplan", 0))
    return out


def parse_cfg(s):
    """'name|plan=hier,pose=vo,map=logodds,div=1' -> (name, dict)."""
    name, _, rest = s.partition("|"); cfg = dict(plan="v1", pose="corr", map="logodds")
    for kv in filter(None, rest.split(",")):
        k, v = kv.split("=")
        cfg[k] = (float(v) if "." in v else int(v)) if v.replace(".", "").replace("-", "").isdigit() else v
    for b in ("div", "translate", "vert"): cfg[b] = bool(cfg.get(b, 0))
    return name, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--scene", required=True); ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed0", type=int, default=0); ap.add_argument("--sensor", default="rgbd", choices=["rgbd", "rgb"]); ap.add_argument("--out", default="results_v5")
    ap.add_argument("--configs", required=True, help="';'-separated 'name|key=val,...'"); ap.add_argument("--dist_budget", type=float, default=100.0); ap.add_argument("--cov_stop", type=float, default=0.97)
    ap.add_argument("--views", type=int, default=1); ap.add_argument("--value_model", default=None); ap.add_argument("--rank_model", default=None)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    g, n = paths(a.scene); S = UAVSim(g); S.sim.pathfinder.load_nav_mesh(n); G = Geometry(g, S.sim.pathfinder.get_bounds()); nav = nav_cell_set(S)
    gti = {v: i for i, v in enumerate(G.gt_vox)} if a.views else None
    dn = DepthNet() if a.sensor == "rgb" else None
    import frontier_value
    vfs = dict(learned=frontier_value.load(a.value_model) if a.value_model else None, learned_tsp=frontier_value.load(a.value_model) if a.value_model else None,
               rank=frontier_value.load(a.rank_model) if a.rank_model else None, oracle=frontier_value.oracle, frustum=frontier_value.frustum_gain)
    out = []; fn = f"{a.out}/{a.scene}__{a.sensor}.json"
    for spec in a.configs.split(";"):
        name, cfg = parse_cfg(spec)
        for sd in range(a.seed0, a.seed0 + a.seeds):
            r = rollout5(cfg, S, G, a.steps, sd, nav, a.scene, dn, a.dist_budget, a.cov_stop, gti, vfs.get(str(cfg.get("sel", ""))))
            r["name"] = name; r["sensor"] = a.sensor; out.append(r)
            print(f"{a.scene} {name} s{sd} cov={r['coverage']:.3f} coll={r['collisions']} dist={r['dist']:.0f}m act={r['actions']} drift={r['final_drift_m']:.2f} "
                  f"v3={r['views'].get('views_ge3', 0):.3f} stc={r['stc']} plan_ms={r.get('plan_ms', 0):.0f} vo={r['vo']} wall={r['wall_s']:.0f}s", flush=True)
            json.dump(out, open(fn, "w"))
