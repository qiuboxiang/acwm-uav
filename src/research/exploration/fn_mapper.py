"""Numpy stand-in for FrontierNet's pywavemap-based WaveMapper (same interface): dense log-odds voxel grid, ray-cast free space + hit voxels.
Only the occupancy backend is replaced (the C++ wavemap wheel does not build on this machine); the frontier network, manager and planner are the original code."""
import numpy as np

class NumpyMapper:
    ORIGIN = np.array([-20.0, -20.0, -5.0]); EXTENT = np.array([40.0, 40.0, 10.0])
    def __init__(self, params):
        self.p = params; self.res = params["resolution"]; self.shape = np.ceil(self.EXTENT / self.res).astype(int)
        self.L = np.zeros(self.shape, np.float32); self.buf = []
        s = 4; u, v = np.meshgrid(np.arange(0, params["width"], s), np.arange(0, params["height"], s))
        self.u = u.reshape(-1); self.v = v.reshape(-1); self.occ_pts = np.zeros((0, 3)); self.free_pts = np.zeros((0, 3))
        self.max_range = params["max_range"]; self.min_range = params["min_range"]
    def get_parameters(self): return self.p
    def insert_depth_to_buffer(self, depth, transform): self.buf.append((np.asarray(depth, np.float32), np.asarray(transform, float)))
    def _idx(self, pts):
        i = np.floor((pts - self.ORIGIN) / self.res).astype(np.int64); ok = ((i >= 0) & (i < self.shape)).all(1); return i[ok]
    def integrate_from_buffer(self):
        p = self.p
        for depth, T in self.buf:
            d = depth[self.v, self.u]; ok = np.isfinite(d) & (d > self.min_range)
            if not ok.any(): continue
            d = d[ok]; x = (self.u[ok] - p["cx"]) / p["fx"]; y = (self.v[ok] - p["cy"]) / p["fy"]; rays = np.stack([x, y, np.ones_like(x)], 1)
            R, t = T[:3, :3], T[:3, 3]; hit = d < self.max_range; dm = np.minimum(d, self.max_range)
            lin = []
            for s in np.arange(self.min_range, self.max_range, self.res / 2):
                m = s < dm - self.res
                if not m.any(): break
                i = self._idx((rays[m] * s) @ R.T + t); lin.append(np.ravel_multi_index(i.T, self.shape))
            if lin:
                u = np.unique(np.concatenate(lin)); self.L.reshape(-1)[u] = np.maximum(self.L.reshape(-1)[u] - 0.4, -4.0)
            i = self._idx((rays[hit] * d[hit, None]) @ R.T + t)
            if len(i):
                u = np.unique(np.ravel_multi_index(i.T, self.shape)); self.L.reshape(-1)[u] = np.minimum(self.L.reshape(-1)[u] + 0.85, 4.0)
        self.buf.clear()
    def interpolate_occupancy_grid(self):
        self.occ_pts = np.argwhere(self.L > 0.6) * self.res + self.ORIGIN; self.free_pts = np.argwhere(self.L < -1e-5) * self.res + self.ORIGIN
    def get_occupancy_grid(self): return {"occupied": self.occ_pts, "free": self.free_pts}
