import numpy as np, habitat_sim, os
from scenes import SCENES, paths, Geometry
os.makedirs("results_v4/geo", exist_ok=True)
for s in SCENES:
    g, n = paths(s); pf = habitat_sim.nav.PathFinder(); pf.load_nav_mesh(n); G = Geometry(g, pf.get_bounds())
    np.savez_compressed(f"results_v4/geo/{s}.npz", pts=G.pts[::8].astype(np.float32), bounds=G.bounds); print(s, len(G.pts))
