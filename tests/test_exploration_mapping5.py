"""Unit tests for the sensor-side mapping / odometry / planning blocks of the v5 exploration agent (no simulator needed)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "research" / "exploration"))
import mapping5 as M  # noqa: E402

RES = 256
F = (RES / 2) / np.tan(np.deg2rad(45))
BOXES = [((-4.0, 0.0, -3.0), (4.0, 2.8, 3.0), True), ((1.0, 0.0, -1.5), (1.8, 1.2, -0.7), False), ((-2.5, 0.0, 1.0), (-1.9, 2.0, 1.6), False)]


def render_depth(pos, yaw):
    """Analytic z-depth of a closed room (inside of the first box) with two solid boxes, for the agent's camera model."""
    u, v = np.meshgrid(np.arange(RES) + 0.5, np.arange(RES) + 0.5)
    rc = np.stack([(u - RES / 2) / F, -(v - RES / 2) / F, -np.ones_like(u)], -1)
    d = rc @ M.yaw_R(yaw).T
    t_best = np.full(u.shape, np.inf)
    for lo, hi, inside in BOXES:
        lo, hi = np.array(lo), np.array(hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (lo - pos) / d; t2 = (hi - pos) / d
        if inside:
            t = np.where(d > 0, t2, t1).min(-1)
        else:
            tn = np.minimum(t1, t2).max(-1); tf = np.maximum(t1, t2).min(-1)
            t = np.where((tn <= tf) & (tn > 0), tn, np.inf)
        t_best = np.minimum(t_best, t)
    return np.where(np.isfinite(t_best), t_best, 0.0).astype(np.float32)


def test_log_odds_clears_stale_obstacle_and_keeps_hits():
    g = M.LogOddsGrid((10, 10, 10))
    g.update(np.zeros((0, 3), int), [[5, 5, 5]])
    assert g.occ[5, 5, 5] == M.OCC
    for _ in range(3):
        g.update([[5, 5, 5], [4, 5, 5]], np.zeros((0, 3), int))
    assert g.occ[5, 5, 5] == M.FREE and g.n_cleared == 1 and g.occ[4, 5, 5] == M.FREE
    assert g.occ[0, 0, 0] == M.UNKNOWN
    # a cell hit and traversed in the same frame counts as hit
    g.update([[1, 1, 1]], [[1, 1, 1]])
    assert g.occ[1, 1, 1] == M.OCC
    # clamping bounds the log-odds
    for _ in range(20): g.update(np.zeros((0, 3), int), [[2, 2, 2]])
    assert g.L[2, 2, 2] == pytest.approx(g.l_max)


@pytest.mark.parametrize("motion", [("fwd", 0.9), ("yaw", 15.0), ("blocked", 0.9), ("up", 0.25)])
def test_depth_odometry_recovers_motion(motion):
    vo = M.DepthOdometry(F)
    p0 = np.array([-0.5, 1.5, 0.8]); y0 = 200.0
    vo.set_keyframe(render_depth(p0, y0), p0, y0)
    fwd = np.array([-np.sin(np.deg2rad(y0)), 0, -np.cos(np.deg2rad(y0))])
    kind, mag = motion
    if kind == "fwd": p1, y1, pred = p0 + fwd * mag * 0.96, y0, (p0 + fwd * mag, y0)
    elif kind == "yaw": p1, y1, pred = p0, y0 + mag + 0.8, (p0, y0 + mag)
    elif kind == "up": p1, y1, pred = p0 + [0, mag * 0.9, 0], y0, (p0 + [0, mag, 0], y0)
    else: p1, y1, pred = p0, y0, (p0 + fwd * mag, y0)  # commanded forward, physically blocked
    p, y, info = vo.track(render_depth(p1, y1), pred[0], pred[1], p0, y0, translated=kind in ("fwd", "blocked", "up"))
    assert info["ok"]
    assert np.linalg.norm(p - p1) < 0.05
    assert abs((y - y1 + 180) % 360 - 180) < 0.5


def test_grid_graph_shortest_path_avoids_wall_and_corner_cutting():
    trav = np.ones((9, 1, 9), bool); trav[4, 0, :8] = False  # wall with a gap at z=8
    G, idx, cells = M.grid_graph(trav)
    s = idx[0, 0, 0]; t = idx[8, 0, 0]
    D, P = M.shortest(G, [s])
    path = M.path_from(P[0], t, cells)
    assert path[-1] == (8, 0, 0)
    assert all(trav[c] for c in path)
    assert any(c[2] == 8 for c in path)  # had to go through the gap
    assert D[0, t] > 8 * 0.25 * 1.5


def test_frontier_clusters_and_open_tsp():
    occ = np.full((20, 5, 20), M.UNKNOWN, np.uint8); occ[:10] = M.FREE; occ[:10, :, 0] = M.OCC
    reach = occ == M.FREE; band = np.ones_like(reach)
    cl, Fm = M.frontier_clusters(occ, reach, band, tile=8)
    assert Fm[9].any() and not Fm[:8].any()
    assert sum(len(c) for c in cl) == int(Fm.sum()) and len(cl) >= 2
    C = np.array([[0, 1, 5, 9], [1, 0, 4, 8], [5, 4, 0, 4], [9, 8, 4, 0]], float)
    assert M.open_tsp(C) == [0, 1, 2, 3]


def test_view_bits_records_directions():
    vb = M.ViewBits((4, 4, 4))
    vb.add(np.array([[1, 1, 1], [1, 1, 1]]), np.array([[0.0, 0, 0], [0.0, 0, 0]]), np.array([0.0, 0, 2.0]))
    vb.add(np.array([[1, 1, 1]]), np.array([[0.0, 0, 0]]), np.array([2.0, 0, 0.0]))
    assert M.ViewBits.popcount(vb.B)[1, 1, 1] == 2


def test_forward_ensemble_novelty_drops_on_repeated_transitions():
    pytest.importorskip("cv2")
    fe = M.ForwardEnsemble(seed=0); rng = np.random.default_rng(0)
    rgb = (rng.random((64, 64, 3)) * 255).astype(np.uint8); depth = rng.random((64, 64)).astype(np.float32) * 4
    e0 = fe.embed(rgb, depth); e1 = fe.embed(np.roll(rgb, 3, 1), np.roll(depth, 3, 1))
    first = fe.step(e0, "yaw", 15.0, e1)
    for _ in range(30): last = fe.step(e0, "yaw", 15.0, e1)
    assert last < 0.2 * first
    novel = fe.step(fe.embed(255 - rgb, 4 - depth), "fwd", 0.9, e0)  # an unseen transition stays uncertain
    assert novel > last


def _door_depth(gap_lo, gap_hi, wall_d=0.8, far_d=4.0):
    """z-depth image of a wall at ``wall_d`` with a vertical opening spanning lateral offsets [gap_lo, gap_hi], open room behind."""
    u, v = np.meshgrid(np.arange(RES) + 0.5, np.arange(RES) + 0.5)
    xr = (u - RES / 2) / F  # lateral offset per metre of depth
    x_wall = xr * wall_d
    return np.where((x_wall > gap_lo) & (x_wall < gap_hi), far_d, wall_d)


def test_depth_gap_centred_door():
    g = M.depth_gap(_door_depth(-0.4, 0.4), F)
    assert g is not None
    xl, xr, d, hard_l, hard_r = g
    assert abs((xl + xr) / 2) < 0.06 and 0.7 <= xr - xl <= 0.9 and abs(d - 0.8) < 0.05 and hard_l and hard_r


def test_depth_gap_offset_door_and_closed_wall():
    xl, xr, _, _, _ = M.depth_gap(_door_depth(0.1, 0.9, wall_d=1.2), F)
    assert 0.4 < (xl + xr) / 2 < 0.6 and 0.7 <= xr - xl <= 0.9  # opening to the right: centre is on the +x side
    assert M.depth_gap(np.full((RES, RES), 0.8), F) is None  # flat wall: no opening
    assert M.depth_gap(np.full((RES, RES), 6.0), F) is None  # nothing near: nothing to pass through


def test_depth_gap_flags_field_of_view_edge():
    xl, xr, _, hard_l, hard_r = M.depth_gap(_door_depth(-0.1, 2.0, wall_d=0.3), F)  # camera next to the left frame, opening runs past the view
    assert hard_l and not hard_r and xr <= 0.31  # right edge is where the 90 deg view ends at 0.3 m, not a seen obstacle
