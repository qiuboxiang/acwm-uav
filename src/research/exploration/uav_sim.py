"""Free-flying 4DoF UAV in habitat-sim (navmesh not used), matching IndoorUAV's camera setup (scaled down)."""
import numpy as np, habitat_sim
from magnum import Vector3
from habitat_sim.utils.common import quat_from_angle_axis

class UAVSim:
    # IndoorUAV action magnitudes (m / deg)
    ACTIONS = {"fwd_s": 0.15, "fwd_l": 0.9, "turn_s": 3, "turn_l": 15}

    def __init__(self, glb, res=(256, 256), hfov=90, cam_h=1.5, seed=0, rec=None):
        self.hfov = hfov; self.rec = rec
        cfg = habitat_sim.SimulatorConfiguration()
        cfg.scene_id = glb; cfg.enable_physics = False; cfg.gpu_device_id = 0
        specs = []
        for uuid, st in (("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)):
            s = habitat_sim.CameraSensorSpec(); s.uuid = uuid; s.sensor_type = st
            s.resolution = list(res); s.hfov = hfov; s.position = Vector3(0, cam_h, 0)
            specs.append(s)
        if rec:  # extra high-resolution recording camera at the same pose (dataset frames); agent perception keeps using the small sensors
            for uuid, st in (("rec_rgb", habitat_sim.SensorType.COLOR), ("rec_depth", habitat_sim.SensorType.DEPTH)):
                s = habitat_sim.CameraSensorSpec(); s.uuid = uuid; s.sensor_type = st; s.resolution = list(rec); s.hfov = hfov; s.position = Vector3(0, cam_h, 0); specs.append(s)
        ag = habitat_sim.agent.AgentConfiguration(); ag.sensor_specifications = specs
        ag.height = cam_h; ag.radius = 0.1
        self.sim = habitat_sim.Simulator(habitat_sim.Configuration(cfg, [ag]))
        self.agent = self.sim.get_agent(0)
        self.rng = np.random.default_rng(seed)

    def set_pose(self, pos, yaw_deg):
        st = habitat_sim.AgentState(); st.position = np.asarray(pos, np.float32)
        st.rotation = quat_from_angle_axis(np.deg2rad(yaw_deg), np.array([0, 1, 0]))
        self.agent.set_state(st); self.yaw = yaw_deg

    def pose(self):
        return np.array(self.agent.get_state().position), self.yaw

    def obs(self):
        o = self.sim.get_sensor_observations(); return o["rgb"][..., :3], o["depth"]

    def step_move(self, dist, dyaw=0.0, dz=0.0):
        """Move along heading by dist (negative = back), then yaw, then vertical. Returns new pose."""
        p, y = self.pose(); r = np.deg2rad(y)
        fwd = np.array([-np.sin(r), 0, -np.cos(r)])  # habitat forward = -z
        self.set_pose(p + fwd * dist + np.array([0, dz, 0]), (y + dyaw) % 360)

    def sample_start(self, tries=200):
        for _ in range(tries):
            p = self.sim.pathfinder.get_random_navigable_point()
            if np.isfinite(p).all():
                return p + np.array([0, 0.5, 0])  # hover 0.5m above floor
        raise RuntimeError("no start")
