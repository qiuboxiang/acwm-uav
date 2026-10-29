"""FrontierNet (cvg/FrontierNet) wrapped as an RPC server so it can run in its own conda env (py3.11) next to our habitat-sim env (py3.9).
Mirrors ExplorerApp.exploration() of demo_exploration.py; replaces the Open3D renderer with observations sent by the client."""
import sys, os, logging, argparse, numpy as np
from multiprocessing.connection import Listener
FN = os.environ.get("FRONTIERNET_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../third_party/FrontierNet"))
sys.path.insert(0, FN); os.chdir(FN)
import torch
from frontier.detector import FrontierDetector
from frontier.model.predict import load_model
from utils.frontier_utils import read_config_yaml
try:
    from mapping.wavemap import WaveMapper
except Exception:  # pywavemap wheel unavailable here -> numpy stand-in with the same interface
    from fn_mapper import NumpyMapper as WaveMapper
from frontier.manager import FrontierManager

def arr(d, key, dtype): return np.frombuffer(d[key + "_b"], dtype=dtype).reshape(d[key + "_s"]).copy()

class Server:
    def __init__(self, p):
        self.cfg = read_config_yaml(os.path.join(FN, "config/hm3d_exploration.yaml"))
        self.cfg["bounds"] = p["bounds"]; self.cfg["filter_bbox"] = p["bounds"]; self.cfg["min_dist2occ"] = p.get("min_dist2occ", 0.2)
        K = np.array(p["K"], float).reshape(3, 3); self.K = K; H, W = p["H"], p["W"]
        vox = self.cfg["voxel_size"]
        self.mapper = WaveMapper(params={"min_cell_width": vox / 2.0, "width": W, "height": H, "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
                                         "min_range": 0.05, "max_range": self.cfg["depth_range"], "resolution": vox})
        unet = load_model(path=os.path.join(FN, "model_weights/rgbd_11cls.pth"), num_classes=self.cfg["num_classes"], use_depth=True)
        self.det = FrontierDetector(model=unet, camera_intrinsic=K.copy(), use_depth=True, img_size_model=self.cfg["input_img_size"], device="cpu", log_level="WARNING")
        self.man = FrontierManager(params=self.cfg, log_level="WARNING")
        self.predict_interval = int(self.cfg.get("predict_interval", 5)); self.path = []; self.move_enough = True; self.last = None; self.started = False; self.n_detect = 0

    def iterate(self, d):
        rgb = arr(d, "rgb", np.uint8); depth = arr(d, "depth", np.float32); W_T_C = arr(d, "pose", np.float64); C_T_W = np.linalg.inv(W_T_C)
        self.mapper.insert_depth_to_buffer(depth=depth, transform=W_T_C)
        if not self.started:
            self.mapper.integrate_from_buffer(); self.mapper.interpolate_occupancy_grid(); og = self.mapper.get_occupancy_grid()
            self.man.update_map(free_map=og["free"], occ_map=og["occupied"]); self.started = True
        if self.last is None or np.linalg.norm(W_T_C[:3, 3] - self.last[:3, 3]) > 0.1 or np.arccos(np.clip((np.trace(self.last[:3, :3].T @ W_T_C[:3, :3]) - 1) / 2, -1, 1)) > 0.26:
            self.man.add_robot_poses([W_T_C]); self.last = W_T_C.copy(); self.move_enough = True
        n = len(self.man.robot_poses)
        no_more = len(self.man.valid_frontiers) == 0 and n > 10
        reach = len(self.path) == 0 or (n - 1) % self.predict_interval == 0
        if no_more or reach:
            self.det.detect(rgb=rgb, depth=depth, df_normalizer=self.cfg["df_normalizer"], df_thr=self.cfg["df_thr"])
            fts = self.det.anchor_fts(depth=depth, extrinsic=C_T_W); self.n_detect += 1
            if fts:
                ids = self.man.add_robot_poses([W_T_C]); self.man.add_frontiers(frontiers=fts, parent_ids=ids)
                self.man.filter_frontiers(); self.man.gain_adjustment(); self.man.filter_frontiers()
            self.last_nfts = len(fts) if fts else 0
            if len(self.man.valid_frontiers) == 0 and n > 10: return dict(done=True, nfront=0, nfts=self.last_nfts, n=n)
        self.mapper.integrate_from_buffer(); self.mapper.interpolate_occupancy_grid(); og = self.mapper.get_occupancy_grid()
        self.man.update_map(free_map=og["free"], occ_map=og["occupied"]); self.man.gain_adjustment(); self.man.filter_frontiers()
        self.man.merge_frontiers(); self.man.filter_frontiers(); self.man.update_utility(current_pos=W_T_C[:3, 3])
        if reach and self.move_enough:
            self.path = self.man.plan_path_to_goal(W_T_C) or []
            if self.path: self.move_enough = False
        while len(self.path) > 1 and np.linalg.norm(np.array(self.path[0])[:3, 3] - W_T_C[:3, 3]) < 0.25: self.path.pop(0)
        if len(self.path) == 1 and np.linalg.norm(np.array(self.path[0])[:3, 3] - W_T_C[:3, 3]) < 0.25: self.path = []; self.move_enough = True
        nxt = np.array(self.path[0], float) if self.path else None
        return dict(done=False, nfront=len(self.man.valid_frontiers), n_path=len(self.path), n_detect=self.n_detect,
                    next_b=None if nxt is None else nxt.tobytes(), last=len(self.path) == 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, required=True); a = ap.parse_args()
    L = Listener(("127.0.0.1", a.port), authkey=b"acwm"); c = L.accept(); S = None
    while True:
        try: m = c.recv()
        except EOFError: break
        try:
            if m["cmd"] == "init": S = Server(m); c.send(dict(ok=True))
            elif m["cmd"] == "iter": c.send(S.iterate(m))
            elif m["cmd"] == "reset_path": S.path = []; S.move_enough = True; c.send(dict(ok=True))
            elif m["cmd"] == "quit": c.send(dict(ok=True)); break
        except Exception as e:
            import traceback; c.send(dict(error=traceback.format_exc()))
