import torch
import numpy as np

from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCylinder
from omni.isaac.core.utils.prims import get_prim_at_path

from robots.Four_Wheels import LimoAckermann as Robot

from skrl.utils import omniverse_isaacgym_utils

from utils.terrain_generator import Terrain
from utils.terrain_utils import add_terrain_to_stage

TASK_CFG = {"test": False,
            "device_id": 0,
            "headless": True,
            "sim_device": "gpu",
            "enable_livestream": False,
            "warp": False,
            "seed": 42,
            "task": {"name": "ReachingIiwa",
                     "physics_engine": "physx",
                     "env": {"numEnvs": 1024,
                             "envSpacing": 1.5,}}}

class RobotView(ArticulationView):
    def __init__(self, prim_paths_expr: str, name: str = "robot_view") -> None:
        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)



class ReachingFoodTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.dt = 1 / 120.0

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._action_scale = self._task_cfg["env"]["actionScale"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        

        # observation and action space
        self._num_observations = 89 + 6 + 4 + 2  # hiehgtpoints + IMU vel + torque per wheel + target
        self._num_actions = 81  # Assuming 4 actions: forward, backward, left, right

        RLTask.__init__(self, name, env)

    def set_up_scene(self, scene) -> None:
        self.get_robot()
        self.get_food()

        super().set_up_scene(scene)

        # robot view
        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        scene.add(self._robots)
        # food view
        self._foods = RigidPrimView(prim_paths_expr="/World/envs/.*/food", name="food_view")
        scene.add(self._foods)

    

    def get_robot(self):
        # Assuming LIMO or similar wheeled robot
        robot = Robot(
            prim_path=self.default_zero_env_path + "/robot",
            translation=torch.tensor([0.0, 0.0, 0.0]),
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
            name="robot"
        )
        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("robot")
        )

    def get_food(self):
        food_positions = [
            torch.tensor([1.0, 1.0, 0.1]),  # Bottom-left corner of Room 1
            torch.tensor([10.0, 10.0, 0.1])  # Top-right corner of Room 2
        ]
        initial_food_position = food_positions[np.random.randint(0, 2)]

        food_prim = DynamicCylinder(
            prim_path="/World/envs/.*/food",
            translation=initial_food_position,
            radius=0.1,
            height=0.2,
            name="food",
            color=torch.tensor([0, 0, 1])  # Blue food
        )
        self._scene.add(food_prim)

        # Assign food to physics engine
        omniverse_isaacgym_utils.assign_prim_to_physics_engine(food_prim.prim_path)

        # Store food positions for future reference
        self._food_positions = food_positions
        self._food_prim = food_prim

    def init_data(self) -> None:
        self.robot_default_dof_pos = torch.zeros((self._num_envs, self._num_actions), device=self._device)
        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device)
    
   
   
   
    def _create_trimesh(self, create_mesh=True):
        self.terrain = Terrain(num_robots=1)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
        if create_mesh:
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )
    

    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self._create_trimesh(create_mesh=create_mesh)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)





    def get_observations(self):
        self.measured_heights = self.get_heights()
        heights = (
            torch.clip(self.base_pos[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.height_meas_scale
        )
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.lin_vel_scale,
                self.base_ang_vel * self.ang_vel_scale,
                self.projected_gravity,
                self.commands[:, :3] * self.commands_scale,
                self.dof_pos * self.dof_pos_scale,
                self.dof_vel * self.dof_vel_scale,
                heights,
                self.actions,
            ),
            dim=-1,
        )

    def get_heights(self, env_ids=None):
        
        heights = self.height_samples

        
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
    

    def is_done(self) -> None:
        self.reset_buf.fill_(0)
        # target reached
        self.reset_buf = torch.where(self._computed_distance <= 0.035, torch.ones_like(self.reset_buf), self.reset_buf)
        # max episode length
        self.reset_buf = torch.where(self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)   