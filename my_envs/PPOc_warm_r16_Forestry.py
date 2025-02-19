import torch
import math
import numpy as np
import gym
from gym import spaces

from my_envs.rl_task import RLTask 

from omni.isaac.core.prims import RigidPrim, RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
from omni.isaac.core.utils.torch.maths import torch_rand_float
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage, print_stage_prim_paths
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.prims import GeometryPrim, GeometryPrimView

from pxr import PhysxSchema, UsdPhysics


from my_robots.origin_v10 import AvularOrigin_v10 as Robot_v10 

from my_utils.origin_terrain_generator import *
from my_utils.terrain_utils import *

TASK_CFG = {"test": False,
            "device_id": 0,
            "headless": True,
            "sim_device": "gpu",
            "enable_livestream": True,
            "warp": False,
            "seed": 42,
            "task": {"name": "ReachingFood",
                     "physics_engine": "physx",
                     "env": {"numEnvs": 64, # has to be perfect square
                             "envSpacing": 3.0,
                             "episodeLength": 500,
                             "enableDebugVis": False,
                             "clipObservations": 1000.0,
                             "controlFrequencyInv": 4,
                             "baseInitState": {"pos": [0.0, 0.0, 0.0], # x,y,z [m]
                                              "rot": [1.0, 0.0, 0.0, 0.0], # w,x,y,z [quat]
                                              "vLinear": [0.0, 0.0, 0.0],  # x,y,z [m/s]
                                              "vAngular": [0.0, 0.0, 0.0],  # x,y,z [rad/s]
                                                },
                            "dofInitTorques": [0.0, 0.0, 0.0, 0.0],
                            "dofInitVelocities": [0.0, 0.0, 0.0, 0.0],
                            "terrain": {"staticFriction": 1.0,  # [-]
                                        "dynamicFriction": 1.0,  # [-]
                                        "restitution": 0.0,  # [-]
                                        # rough terrain only:
                                        "curriculum": True,
                                        "maxInitMapLevel": 0,
                                        "mapLength": 16.0,
                                        "mapWidth": 16.0,
                                        "numLevels": 6,
                                        "numTerrains": 2,
                                        # terrain types: [ smooth slope, rough slope, stairs up, stairs down, discrete]
                                        "terrainProportions": [0.35, 0.55, 0.7, 0.85, 1.0],
                                        # tri mesh only:
                                        "slopeTreshold": 0.5,
                                        },
                            "TerrainType": "double room", # rooms, stairs, sloped, mixed_v1, mixed_v2, mixed_v3, custom, custom_mixed      
                            "learn" : {"linearVelocityScale": 2.0,
                                       "angularVelocityScale": 0.25,
                                       "dofPositionScale": 1.0,
                                       "dofVelocityScale": 0.05,
                                       "heightMeasurementScale": 5.0,
                                       "lambdaSlipScale": 10.0,
                                       "terminalReward": 0.0,
                                       "linearVelocityXYRewardScale": 1.0,
                                       "linearVelocityZRewardScale": -4.0,
                                       "angularVelocityZRewardScale": 1.0,
                                       "angularVelocityXYRewardScale": -0.5,
                                       "orientationRewardScale": -0.0,
                                       "torqueRewardScale": -0.0,
                                       "jointAccRewardScale": -0.0,
                                       "baseHeightRewardScale": -0.0,
                                       "actionRateRewardScale": -0.05,
                                       "fallenOverRewardScale": -200.0,
                                       "slipLongitudinalRewardScale": -5.0,
                                       "episodeLength_s": 15.0,
                                       "pushInterval_s": 20.0,},
                            "randomCommandVelocityRanges": {"linear_x": 0.0, # [m/s]
                                                            "linear_y": [-0.5, 0.5], # [m/s]
                                                            "yaw": [-3.14, 3.14], # [rad/s]
                                                            "yaw_constant": 0.5,},   # [rad/s]
                            "control": {"decimation": 4, # decimation: Number of control action updates @ sim DT per policy DT
                                        "stiffness": 0.05, # [N*m/rad] For torque setpoint control
                                        "damping": .005, # [N*m*s/rad]
                                        "actionScale": 10.0,
                                        "wheel_radius": 0.1175,},   # leave room to overshoot or corner 

                            },
                     "sim": {"dt": 0.005,  
                             "use_gpu_pipeline": True,
                             "gravity": [0.0, 0.0, -9.81],
                             "add_ground_plane": False,
                             "use_flatcache": True,
                             "enable_scene_query_support": False,
                             "enable_cameras": False,
                             "default_physics_material": {"static_friction": 1.0,
                                                         "dynamic_friction": 1.0,
                                                         "restitution": 0.0},
                             "physx": {"worker_thread_count": 4,
                                      "solver_type": 1,
                                      "use_gpu": True,
                                      "solver_position_iteration_count": 4,
                                      "solver_velocity_iteration_count": 4,
                                      "contact_offset": 0.01,
                                      "rest_offset": 0.0,
                                      "bounce_threshold_velocity": 0.2,
                                      "friction_offset_threshold": 0.04,
                                      "friction_correlation_distance": 0.025,
                                      "enable_sleeping": True,
                                      "enable_stabilization": True,
                                      "max_depenetration_velocity": 100.0,
                                      "gpu_max_rigid_contact_count": 524288,
                                      "gpu_max_rigid_patch_count": 33554432,
                                      "gpu_found_lost_pairs_capacity": 524288,
                                      "gpu_found_lost_aggregate_pairs_capacity": 262144,
                                      "gpu_total_aggregate_pairs_capacity": 1048576,
                                      "gpu_max_soft_body_contacts": 1048576,
                                      "gpu_max_particle_contacts": 1048576,
                                      "gpu_heap_capacity": 33554432,
                                      "gpu_temp_buffer_capacity": 16777216,
                                      "gpu_max_num_partitions": 8},
                             "robot": {"override_usd_defaults": False,
                                       "fixed_base": False,
                                       "enable_self_collisions": False,
                                       "enable_gyroscopic_forces": False,
                                       "solver_position_iteration_count": 4,
                                       "solver_velocity_iteration_count": 4,
                                       "sleep_threshold": 0.005,
                                       "stabilization_threshold": 0.001,
                                       "density": -1,
                                       "max_depenetration_velocity": 100.0,
                                       "contact_offset": 0.005,
                                       "rest_offset": 0.0,},
}}}

class RobotView(ArticulationView):
    def __init__(
            self, 
            prim_paths_expr: str, 
            name: str = "robot_view",
            track_contact_forces=False,
            prepare_contact_sensors=False,
        ) -> None:
        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self._base = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/main_body",
            name="base_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )

class ReachingTargetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.height_samples = None
        self.custom_origins = False
        self.init_done = False
        self._env_spacing = 0.0
        
        # observation and action space (DQN)
        self._num_observations = 162
        self._num_actions = 4  # Designed discrete action space see pre_physics_step()

        self.observation_space = spaces.Box(
            low=-1.0,  # Replace with a specific lower bound if needed
            high=1.0,  # Replace with a specific upper bound if needed
            shape=(self._num_observations,),
            dtype=np.float32  # Ensure data type is consistent
        )
        
        # Using the shape argument
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._num_actions,),
            dtype=np.float32
        )

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        self.height_points = self.init_height_points()  
        self.measured_heights = None

        self.episode_buf = torch.zeros(self.num_envs, dtype=torch.long)

        self.linear_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self.angular_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_linear_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_angular_velocity = torch.zeros((self.num_envs, 3), device=self.device)

        self.last_torq_error = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.max_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "lin_vel_xy": torch_zeros(),
            "lin_vel_z": torch_zeros(),
            "ang_vel_z": torch_zeros(),
            "ang_vel_xy": torch_zeros(),
            "orient": torch_zeros(),
            "torques": torch_zeros(),
            "joint_acc": torch_zeros(),
            "base_height": torch_zeros(),
            "air_time": torch_zeros(),
            "collision": torch_zeros(),
            "stumble": torch_zeros(),
            "action_rate": torch_zeros(),
            "hip": torch_zeros(),
            "fallen_over": torch_zeros(),
            "slip_longitudinal": torch_zeros(),
        }
        
        # --- NEW: Initialize warm-start flag (disabled by default) ---
        self.warm_start = False
        self.phase_name = ""

        return
        

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # normalization
        self.lin_vel_scale = self._task_cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self._task_cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_pos_scale = self._task_cfg["env"]["learn"]["dofPositionScale"]
        self.dof_vel_scale = self._task_cfg["env"]["learn"]["dofVelocityScale"]
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]
        self.lambda_slip_scale = self._task_cfg["env"]["learn"]["lambdaSlipScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"]
        self.rew_scales["lin_vel_xy"] = self._task_cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["lin_vel_z"] = self._task_cfg["env"]["learn"]["linearVelocityZRewardScale"]
        self.rew_scales["ang_vel_z"] = self._task_cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["ang_vel_xy"] = self._task_cfg["env"]["learn"]["angularVelocityXYRewardScale"]
        self.rew_scales["orient"] = self._task_cfg["env"]["learn"]["orientationRewardScale"]
        self.rew_scales["torque"] = self._task_cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["joint_acc"] = self._task_cfg["env"]["learn"]["jointAccRewardScale"]
        self.rew_scales["base_height"] = self._task_cfg["env"]["learn"]["baseHeightRewardScale"]
        self.rew_scales["action_rate"] = self._task_cfg["env"]["learn"]["actionRateRewardScale"]
        self.rew_scales["fallen_over"] = self._task_cfg["env"]["learn"]["fallenOverRewardScale"]
        self.rew_scales["slip_longitudinal"] = self._task_cfg["env"]["learn"]["slipLongitudinalRewardScale"]

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]
        self.yaw_constant = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw_constant"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.sim_dt = self._task_cfg["sim"]["dt"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.r = self._task_cfg["env"]["control"]["wheel_radius"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
      
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        # env config
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        torques = self._task_cfg["env"]["dofInitTorques"]
        dof_velocities = self._task_cfg["env"]["dofInitVelocities"]
        self.dof_init_state = torques + dof_velocities


    def init_height_points(self):
        # 1mx1.6m rectangle (without center line)
        y = 0.1 * torch.tensor(
            [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False
        )  # 10-50cm on each side
        x = 0.1 * torch.tensor(
            [-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False
        )  # 20-80cm on each side
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

  

    def _create_trimesh(self, create_mesh=True):
        self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
        vertices = self.terrain.vertices
        triangles = self.terrain.triangles
        position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
        if create_mesh:
            add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
        self.height_samples = (
            torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
        )
        print(f"height_samples: {self.height_samples}")

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_terrain()
        self.get_robot()

        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"], copy_from_source=True)

        # robot view
        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        scene.add(self._robots)
        scene.add(self._robots._base)

    def initialize_views(self, scene):
        # initialize terrain variables even if we do not need to re-create the terrain mesh
        self.get_terrain(create_mesh=False)

        super().initialize_views(scene)
        if scene.object_exists("robot_view"):
            scene.remove_object("robot_view", registry_only=True)
        if scene.object_exists("base_view"):
            scene.remove_object("base_view", registry_only=True)
        self._robots = RobotView(
            prim_paths_expr="/World/envs/.*/robot", name="robot_view", track_contact_forces=False
        )
        scene.add(self._robots)
        scene.add(self._robots._base)
      

    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        if not self.curriculum:
            self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
        self.terrain_levels = torch.randint(
            0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,), device=self.device
        )
        self.terrain_types = torch.randint(
            0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device
        )
        self._create_trimesh(create_mesh=create_mesh)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

    def get_robot(self):
        robot_translation = torch.tensor([0.0, 0.0, 0.0])
        robot_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        robot = Robot_v10(
            prim_path=self.default_zero_env_path + "/robot",
            name="robot",
            translation=robot_translation,
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("robot")
        )
        robot.set_robot_properties(self._stage, robot.prim)

        
    def post_reset(self):
        
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_init_state = torch.tensor(self.dof_init_state, dtype=torch.float, device=self.device, requires_grad=False)

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.common_step_counter = 0
        self.extras = {}
        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor(
            [self.lin_vel_scale, self.lin_vel_scale, self.ang_vel_scale],
            device=self.device,
            requires_grad=False,
        )        
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(
            (self.num_envs, 1)
        )
        
        self.torques = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.last_dof_vel = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torq_error = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device, requires_grad=False)


        for i in range(self.num_envs):
            self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
        self.num_dof = self._robots.num_dof 
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_acc = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_effort = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(indices)
        self.init_done = True


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        self.update_terrain_level(env_ids)
        self.base_pos[env_ids] = self.base_init_state[0:3]
        self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
        self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        # Apply random rotation around Z-axis
        random_angle = torch_rand_float(0, 2 * torch.pi, (len(env_ids),), device=self.device)
        quat_z = torch.stack([torch.cos(random_angle / 2), torch.zeros_like(random_angle), torch.zeros_like(random_angle), torch.sin(random_angle / 2)], dim=-1)
        self.base_quat[env_ids] = quat_mul(quat_z, self.base_init_state[3:7].repeat(len(env_ids), 1))
        self.base_velocities[env_ids] = self.base_init_state[7:]


        self.dof_vel[env_ids] = self.dof_init_state[4:8]
        self.dof_effort[env_ids] = self.dof_init_state[0:4]
    
        self._robots.set_world_poses(
            positions=self.base_pos[env_ids].clone(), orientations=self.base_quat[env_ids].clone(), indices=indices
        )
        self._robots.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._robots.set_joint_efforts(self.dof_effort[env_ids].clone(), indices=indices)
        self._robots.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)   

        self.commands[env_ids, 0] = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.commands[env_ids, 1] = torch_rand_float(
            self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids, 3] = torch_rand_float(
            self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
        self.commands[env_ids] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.25).unsqueeze(
            1
        )  # set small commands to zero

        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.episode_buf[env_ids] = 0 

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def update_terrain_level(self, env_ids):
        if not self.init_done or not self.curriculum:
            # do not change on initial reset
            return
        root_pos, _ = self._robots.get_world_poses(clone=False)
        self.distance = torch.norm(root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
                
        self.terrain_levels[env_ids] -= 1 * (
            self.distance < self.commands[env_ids, 0] * self.max_episode_length_s * 0.5
        )
        self.terrain_levels[env_ids] += 1 * (self.distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]


        if self.distance[env_ids] > self.max_distance[env_ids]:
            self.max_distance[env_ids] = self.distance[env_ids]
        if self.distance[env_ids] > self.commands[env_ids, 0] * self.max_episode_length_s * 0.5:
            self.max_distance = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
    def sample_velocity_command(self, env_id: int):
        """
        Return a velocity command (x vel) for a single environment based on the current warm start phase.
        For multi-env, you'll call this for each env.
        """
        if self.warm_start and self.phase_name == "straight":
            return 0.5
        
        if self.warm_start and self.phase_name == "rotate_left":
            return 0.0
        
        if self.warm_start and self.phase_name == "rotate_right":
            return 0.0
        
        if self.warm_start and self.phase_name == "circle_left":
            return 0.5

        if self.warm_start and self.phase_name == "circle_right":
            return 0.5
        
        """
        Return a velocity command (x vel) for a single environment based on the current curriculum task.
        """
        if self.terrain_levels[env_id] == 1:
            # Task 1: normal distribution around 0.5, with sigma in [0.01..0.1]
            # Example: linearly scale sigma with episode_buf or a global counter
            fraction = self.max_distance[env_id] / (self.commands[env_id, 0] * self.max_episode_length_s * 0.5)
            sigma = 0.01 + 0.09 * fraction
            x_vel = torch.normal(mean=0.5, std=sigma, size=(1,), device=self.device).item()
            return max(x_vel, 0.0)  # keep it non-negative if you want

        elif self.terrain_levels == 2:
            # Task 2: sinusoidal with mean=1, frequency + amplitude changes
            # Suppose we let the frequency grow from 0.01..0.1 and amplitude from 0.1..1
            fraction = self.max_distance[env_id] / (self.commands[env_id, 0] * self.max_episode_length_s * 0.5)
            freq = 0.01 + 0.09 * fraction
            amp  = 0.1  
            if fraction > 0.5:
                amp = 0.1 + 0.4  * fraction
            t = float(self.common_step_counter) * self.dt
            x_vel = 0.5 + amp * math.sin(freq * t)
            return max(x_vel, 0.0)

        elif self.terrain_levels == 3:
            # Task 3: range 0..10. Start with 0..5, then up to 10
            # We'll do a simple sub-task switch
            fraction = self.max_distance[env_id] / (self.commands[env_id, 0] * self.max_episode_length_s * 0.5)
            t = float(self.common_step_counter) * self.dt
            scale = 2.0 # m/s
            Noise = 0.5 * t/self.max_episode_length_s
            x_vel = torch.normal(mean=0.0, std=Noise, size=(1,), device=self.device).item().item() + scale * t/self.max_episode_length_s
            return max(x_vel, 0.0)

        else:
            # Task 4 (or any default): final terrain steps/slopes, normal(0.5,0.1)
            x_vel = torch.normal(mean=0.5, std=0.1, size=(1,), device=self.device).item()
            return max(x_vel, 0.0)
        
    def refresh_dof_state_tensors(self):
        self.dof_pos = self._robots.get_joint_positions(clone=False)
        self.dof_vel = self._robots.get_joint_velocities(clone=False)

    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._robots.get_world_poses(clone=False)
        self.base_velocities = self._robots.get_velocities(clone=False)
    

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return
        
        self.actions = actions.clone().to(self.device)

        for _ in range(self.decimation):
            if self.world.is_playing():
                
                wheel_torq = self.action_scale * self.actions  # shape: [num_wheels]
                print(f"wheel_torq: {wheel_torq}")
                
                sign_vel = torch.sign(self.dof_vel)
                sign_torq = torch.sign(wheel_torq)
                over_speed = torch.abs(self.dof_vel) > 4.25 * 2 # 4.25 rad/s is 0.5 m/s the max speed of the robot is 1.0 m/s

                # Condition: over_speed AND same sign of velocity & torque → set torque = 0
                clamp_mask = over_speed & (sign_vel == sign_torq)
                wheel_torq[clamp_mask] = 0.0

                wheel_torqs = torch.clip(wheel_torq, -80.0, 80.0)

                self._robots.set_joint_efforts(wheel_torqs)

                SimulationContext.step(self.world, render=False)
                self.refresh_dof_state_tensors()


          
    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.episode_buf[:] += 1
        
       
        if self.world.is_playing():
            
            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()

            self.common_step_counter += 1
            if self.common_step_counter % self.push_interval == 0:
                self.push_robots()

            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(self.yaw_constant * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

            self.is_done()
            self.get_states()
            self.calculate_metrics()
            
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
            
            self.last_actions[:] = self.actions[:]
            self.last_dof_vel[:] = self.dof_vel[:]

            # sample velocity commands (x, y, yaw, heading)
            # Here we only do x velocity changes from sample_velocity_command
            for i in range(self._num_envs):
                x_cmd = self.sample_velocity_command(i)
                self.commands[i,0] = x_cmd

        # During warm-start, override rewards to zero and include expert actions ---
        if self.warm_start:
            self.rew_buf = torch.zeros_like(self.rew_buf)
            # Add expert actions to extras so that the policy network can “observe” them for supervised loss.
            self.extras["expert_actions"] = self.actions

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def push_robots(self):
        self.base_velocities[:, 0:2] = torch_rand_float(
            -1.0, 1.0, (self.num_envs, 2), device=self.device
        )  # lin vel x/y
        self._robots.set_velocities(self.base_velocities)
    

    def is_done(self):
        self.reset_buf.fill_(0)

        # max episode length
        self.timeout_buf = torch.where(
            self.episode_buf >= self.max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        ) 

        # Calculate the projected gravity in the robot's local frame
        projected_gravity = quat_apply(self.base_quat, self.gravity_vec)

        # Detect if the robot is on its back based on positive Z-axis component of the projected gravity
        positive_gravity_z_threshold = 0.0  # Adjust the threshold if needed
        self.has_fallen = projected_gravity[:, 2] > positive_gravity_z_threshold
        self.reset_buf = self.has_fallen.clone()

        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

    
    def calculate_metrics(self) -> None:
        # During warm-start, disable reward calculations
        if self.warm_start:
            v_wheel = self.r * self.dof_vel  # v_wheel = r * omega
            base_lin_vel_expanded = self.base_lin_vel[:, 0].unsqueeze(1).expand(-1, v_wheel.shape[1])
            self.lambda_slip = (v_wheel - base_lin_vel_expanded) / torch.maximum(v_wheel, base_lin_vel_expanded)

            self.rew_buf = torch.zeros(self.num_envs, device=self.device)
            return self.rew_buf

        # velocity tracking reward
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        rew_lin_vel_xy = torch.exp(-lin_vel_error / 0.25) * self.rew_scales["lin_vel_xy"]
        rew_ang_vel_z = torch.exp(-ang_vel_error / 0.25) * self.rew_scales["ang_vel_z"]

        # other base velocity penalties (necessary)
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2]) * self.rew_scales["lin_vel_z"]
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1) * self.rew_scales["ang_vel_xy"]

        # torque penalty, joint acceleration penalty, etc. can be computed as before…
        rew_fallen_over = self.has_fallen * self.rew_scales["fallen_over"]

        rew_action_rate = (
            torch.sum(torch.square(self.last_actions - self.actions), dim=1) * self.rew_scales["action_rate"]
        )
        v_wheel = self.r * self.dof_vel  # v_wheel = r * omega
        base_lin_vel_expanded = self.base_lin_vel[:, 0].unsqueeze(1).expand(-1, v_wheel.shape[1])
        self.lambda_slip = (v_wheel - base_lin_vel_expanded) / torch.maximum(v_wheel, base_lin_vel_expanded)
        self.k_lambda = 0.3
        rew_slip_longitudinal = torch.prod(torch.exp(-0.5 * (self.lambda_slip / self.k_lambda) ** 2), dim=1) * self.rew_scales["slip_longitudinal"]

        self.rew_buf = (
            rew_lin_vel_xy
            # + rew_ang_vel_z
            + rew_lin_vel_z
            + rew_ang_vel_xy
            + rew_action_rate
            + rew_fallen_over
            # + rew_slip_longitudinal
        )

        print("Reward Components:")
        print("rew_lin_vel_xy:", rew_lin_vel_xy.mean().item())
        print("rew_lin_vel_z:", rew_lin_vel_z.mean().item())
        print("rew_ang_vel_xy:", rew_ang_vel_xy.mean().item())
        print("rew_action_rate:", rew_action_rate.mean().item())
        print("rew_fallen_over:", rew_fallen_over.mean().item())

        self.rew_buf = torch.clip(self.rew_buf, min=0.0, max=None)
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        self.episode_sums["lin_vel_xy"] += rew_lin_vel_xy
        self.episode_sums["ang_vel_z"] += rew_ang_vel_z
        self.episode_sums["lin_vel_z"] += rew_lin_vel_z
        self.episode_sums["ang_vel_xy"] += rew_ang_vel_xy
        self.episode_sums["action_rate"] += rew_action_rate
        self.episode_sums["fallen_over"] += rew_fallen_over
        self.episode_sums["slip_longitudinal"] += rew_slip_longitudinal

        self.reward_components = {
            "rew_lin_vel_xy": rew_lin_vel_xy.mean().item(),
            "rew_ang_vel_z": rew_ang_vel_z.mean().item(),
            "rew_lin_vel_z": rew_lin_vel_z.mean().item(),
            "rew_ang_vel_xy": rew_ang_vel_xy.mean().item(),
            "rew_action_rate": rew_action_rate.mean().item(),
            "rew_fallen_over": rew_fallen_over.mean().item(),
            "rew_slip_longitudinal": rew_slip_longitudinal.mean().item(),
        }
      
        return self.rew_buf


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
                (self.commands[:, 0] * self.commands_scale[0]).unsqueeze(1),    # (num_envs, 1)
                # (self.commands[:, 2] * self.commands_scale[2]).unsqueeze(1),
                self.dof_vel * self.r * self.dof_vel_scale,
                self.actions,
                self.lambda_slip * self.lambda_slip_scale,
                heights,
            ),
            dim=-1,
        )

        # Print the shape and values of each component
        print("base_lin_vel:", self.base_lin_vel.shape, self.base_lin_vel)
        print("base_ang_vel:", self.base_ang_vel.shape, self.base_ang_vel)
        print("projected_gravity:", self.projected_gravity.shape, self.projected_gravity)
        print("commands[:, 0]:", (self.commands[:, 0] * self.commands_scale[0]).unsqueeze(1).shape, (self.commands[:, 0] * self.commands_scale[0]).unsqueeze(1))
        # print("commands[:, 2]:", (self.commands[:, 2] * self.commands_scale[2]).unsqueeze(1).shape, (self.commands[:, 2] * self.commands_scale[2]).unsqueeze(1))
        print("dof_vel:", self.dof_vel.shape, self.dof_vel)
        print("actions:", self.actions.shape, self.actions)
        print("lambda_slip:", self.lambda_slip.shape, self.lambda_slip)
        print("heights:", heights.shape, heights)
                    
        return {self._robots.name: {"obs_buf": self.obs_buf}}
    

    def get_heights(self, env_ids=None):
        if env_ids:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]
            ) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.base_pos[:, 0:3]
            ).unsqueeze(1)

        points += self.terrain.border_size
        points = (points / self.terrain.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        heights = torch.min(heights1, heights2)

        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale


@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


