import os, glob, numpy as np, trimesh
from scipy.spatial import cKDTree
ROOT = os.environ.get("UAV_SCENES_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "scenes", "scene_datasets")
SCENES = {  # name -> (glb/ply, navmesh)
    "gibson_Adrian": ("gibson/Adrian.glb", "gibson/Adrian.navmesh"),
    "gibson_Anaheim": ("gibson/Anaheim.glb", "gibson/Anaheim.navmesh"),
    "replica_apartment_0": ("replica/apartment_0/habitat/mesh_semantic.ply", "replica/apartment_0/habitat/mesh_semantic.navmesh"),
    "replica_frl_apartment_0": ("replica/frl_apartment_0/habitat/mesh_semantic.ply", "replica/frl_apartment_0/habitat/mesh_semantic.navmesh"),
    "mp3d_17DRP5sb8fy": ("mp3d/17DRP5sb8fy/17DRP5sb8fy.glb", "mp3d/17DRP5sb8fy/17DRP5sb8fy.navmesh"),
    "mp3d_1LXtFkjw3qL": ("mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb", "mp3d/1LXtFkjw3qL/1LXtFkjw3qL.navmesh"),
    "hm3d_16tymPtM7uS": ("hm3d/00083-16tymPtM7uS/16tymPtM7uS.basis.glb", "hm3d/00083-16tymPtM7uS/16tymPtM7uS.basis.navmesh"),
    "hm3d_12e6joG2B2T": ("hm3d/00542-12e6joG2B2T/12e6joG2B2T.basis.glb", "hm3d/00542-12e6joG2B2T/12e6joG2B2T.basis.navmesh"),
}
def paths(name): g, n = SCENES[name]; return os.path.join(ROOT, g), os.path.join(ROOT, n)

class Geometry:
    """Surface point cloud of the scene mesh in habitat world frame -> KD-tree for collision, voxel set as coverage ground truth."""
    def __init__(self, glb, nav_bounds, density=400, max_pts=1_500_000):
        m = trimesh.load(glb, force="mesh")
        lo, hi = np.array(nav_bounds[0]), np.array(nav_bounds[1])
        best = None
        for R in (np.eye(4), trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]), trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])):
            b = trimesh.transformations.transform_points(m.bounds, R); b = np.sort(b, 0)
            # navigable region (xz) must sit inside mesh bbox; score by how well it's contained
            err = np.maximum(lo[[0, 2]] - b[0][[0, 2]] * 0 - b[1][[0, 2]] * 0, 0).sum() * 0 + np.abs((b[0][[0, 2]] + b[1][[0, 2]]) / 2 - (lo[[0, 2]] + hi[[0, 2]]) / 2).sum() + max(0, lo[1] - b[1][1]) + max(0, b[0][1] - hi[1])
            if best is None or err < best[0]: best = (err, R)
        m.apply_transform(best[1]); self.R = best[1]
        n = int(min(max_pts, m.area * density))
        pts, _ = trimesh.sample.sample_surface(m, n)
        self.pts = pts.astype(np.float32); self.tree = cKDTree(self.pts)
        self.bounds = m.bounds
        self.gt_vox = set(map(tuple, np.floor(self.pts / 0.25).astype(np.int32)))
    def collides(self, p, r=0.25):
        return len(self.tree.query_ball_point(p, r, return_length=True)) > 0 if False else self.tree.query(p, k=1)[0] < r
    def segment_free(self, a, b, r=0.25, step=0.1):
        a, b = np.asarray(a, float), np.asarray(b, float); n = max(2, int(np.linalg.norm(b - a) / step) + 1)
        P = a[None] + (b - a)[None] * np.linspace(0, 1, n)[:, None]
        return bool((self.tree.query(P, k=1)[0] >= r).all())
