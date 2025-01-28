import torch
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
                                        "mapLength": 8.0,
                                        "mapWidth": 8.0,
                                        "numLevels": 4,
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
                                       "terminalReward": 0.0,
                                       "linearVelocityXYRewardScale": 1.0,
                                       "linearVelocityZRewardScale": -4.0,
                                       "angularVelocityZRewardScale": 0.5,
                                       "angularVelocityXYRewardScale": -0.05,
                                       "orientationRewardScale": -0.0,
                                       "torqueRewardScale": -0.00002,
                                       "jointAccRewardScale": -0.0005,
                                       "baseHeightRewardScale": -0.0,
                                       "actionRateRewardScale": -0.01,
                                       "fallenOverRewardScale": -1.0,
                                        "slipLongitudinalRewardScale": -0.1,
                                       "episodeLength_s": 15.0,
                                       "pushInterval_s": 20.0,},
                            "randomCommandVelocityRanges": {"linear_x": [-0.5, 0.5], # [m/s]
                                                            "linear_y": [-0.5, 0.5], # [m/s]
                                                            "yaw": [-3.14, 3.14]},   # [rad/s]
                            "control": {"decimation": 4, # decimation: Number of control action updates @ sim DT per policy DT
                                        "stiffness": 40.0, # [N*m/rad] For torque setpoint control
                                        "damping": 2.0, # [N*m*s/rad]
                                        "actionScale": 1.0,
                                        "wheel_radius": 0.1175},   # leave room to overshoot or corner 

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
                                      "contact_offset": 0.005,
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
        
                # observation and action space DQN
        self._num_observations = 188
        self._num_actions = 4  # Designed discrete action space see pre_physics_step()

        self.observation_space = spaces.Box(
            low=float("-10"),  # Replace with a specific lower bound if needed
            high=float("10"),  # Replace with a specific upper bound if needed
            shape=(self.num_observations,),
            dtype=np.float32  # Ensure data type is consistent
        )
        
        # Using the shape argument
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32
        )

        # Define the action range for velocities
        self.min_vel = -0.5  # Example min velocity value
        self.max_vel = 0.5   # Example max velocity value

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        self.height_points = self.init_height_points()  
        self.measured_heights = None

        


        
        self.episode_buf = torch.zeros(self.num_envs, dtype=torch.long)

        self.linear_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self.angular_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_linear_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_angular_velocity = torch.zeros((self.num_envs, 3), device=self.device)

        self.scaled_actions = torch.zeros(self.num_envs, device=self.device)
        self.scaled_delta_diff = torch.zeros(self.num_envs, device=self.device)
        # self.scaled_delta_climb = torch.zeros(self.num_envs, device=self.device)

        self.common_step_counter = 0 # Counter for the first two steps

        



        
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
        }
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

        # command ranges
        self.command_x_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self._task_cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang





        # other
        self.decimation = self._task_cfg["env"]["control"]["decimation"]
        self.dt = self.decimation * self._task_cfg["sim"]["dt"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)
        self.push_interval = int(self._task_cfg["env"]["learn"]["pushInterval_s"] / self.dt + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
      
        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt


        # env config
        self._num_envs = self._task_cfg["env"]["numEnvs"]

        torques = self._task_cfg["env"]["dofInitTorques"]
        dof_velocities = self._task_cfg["env"]["dofInitVelocities"]
        self.dof_init_state = torques + dof_velocities

        self.k_prog = self._task_cfg["env"]["k_prog"]
        self.k_tar = self._task_cfg["env"]["k_tar"]
        self.k_d = self._task_cfg["env"]["k_d"]


        self.decimation = 4


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
            "robot", get_prim_at_path(self.robot_v101.prim_path), self._sim_config.parse_actor_config("robot")
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
        
        self.torques = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )

        self.last_dof_vel = torch.zeros((self.num_envs, 12), dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.num_envs):
            self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]
        self.num_dof = self._robots.num_dof 
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
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
        self.base_quat[env_ids] = self.base_init_state[3:7]
        self.base_velocities[env_ids] = self.base_init_state[7:]


        self.dof_vel[env_ids] = self.dof_init_state[4:8]
        self.dof_effort[env_ids] = self.dof_init_state[0:4]
    
        self._robots.set_world_poses(
            positions=self.base_pos[env_ids].clone(), orientations=self.base_quat[env_ids].clone(), indices=indices
        )
        self._robots.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._robots.set_joint_efforts(self.dof_effort[env_ids].clone(), indices=indices)
        self._robots.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)   

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device
        ).squeeze()
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
        distance = torch.norm(root_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        self.terrain_levels[env_ids] -= 1 * (
            distance < torch.norm(self.commands[env_ids, :2]) * self.max_episode_length_s * 0.25
        )
        self.terrain_levels[env_ids] += 1 * (distance > self.terrain.env_length / 2)
        self.terrain_levels[env_ids] = torch.clip(self.terrain_levels[env_ids], 0) % self.terrain.env_rows
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
    def refresh_dof_state_tensors(self):
        self.dof_pos = self._robots.get_joint_positions(clone=False)
        self.dof_vel = self._robots.get_joint_velocities(clone=False)

    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._robots.get_world_poses(clone=False)
        self.base_vel = self._robots.get_velocities(clone=False)
        self.target_pos, _ = self._targets.get_world_poses(clone=False)


    def pre_physics_step(self, actions):

        if not self.world.is_playing():
            return
        
        # # # If we are still in the first two steps, don't apply any action but advance the simulation
        # if self.common_step_counter < 2:
        #     self.common_step_counter += 1
        #     SimulationContext.step(self.world, render=False)  # Advance simulation
        #     return 

        self.actions = actions.clone().to(self.device)
        # print(f"actions: {self.actions}")

        # Apply the actions to the robot
        self.min_delta = -1.0
        self.max_delta = 1.0

        self.scaled_actions = self.min_vel + (actions[:, 0] + 1) * 0.5 * (self.max_vel - self.min_vel)
        self.scaled_delta_diff = self.min_delta + (actions[:, 1] + 1) * 0.5 * (self.max_delta - self.min_delta)
        # self.scaled_delta_climb = self.min_delta + (self.actions[:, 2] + 1) * 0.5 * (self.max_delta - self.min_delta)

        # print(f"scaled_actions: {self.scaled_actions}")
        # print(f"scaled_delta_diff: {self.scaled_delta_diff}")


        updated_efforts = torch.zeros((self.num_envs, 4), device=self.device)

        # Front left wheel
        updated_efforts[:, 0] = self.scaled_actions + self.scaled_delta_diff # - self.scaled_delta_climb
        # Rear left wheel
        updated_efforts[:, 1] = self.scaled_actions + self.scaled_delta_diff # + self.scaled_delta_climb
        # Front right wheel
        updated_efforts[:, 2] = self.scaled_actions - self.scaled_delta_diff # - self.scaled_delta_climb
        # Rear right wheel
        updated_efforts[:, 3] = self.scaled_actions - self.scaled_delta_diff # + self.scaled_delta_climb

        updated_efforts = torch.clip(updated_efforts, -15.0, 15.0)
        # print(f"updated_efforts: {updated_efforts}")

          
        for i in range(self.decimation):
            if self.world.is_playing():
                self._robots.set_joint_efforts(updated_efforts) 
                SimulationContext.step(self.world, render=False)

          
    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.episode_buf[:] += 1
        
        ids = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
       
        if self.world.is_playing():

            self.refresh_body_state_tensors()

            # prepare quantities            
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                        
            self.is_done()
            self.calculate_metrics()
            
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
            
            self.last_actions[:] = self.actions[:]


        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    

    def is_done(self):
        self.reset_buf.fill_(0)

        # target reached or lost
        self.target_reached = self._computed_distance <= 0.5
        self.reset_buf = torch.where(self.target_reached, torch.ones_like(self.reset_buf), self.reset_buf)

        # max episode length
        self.timeout_buf = torch.where(
            self.episode_buf >= self._max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        ) 
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)  

        # Calculate the projected gravity in the robot's local frame
        projected_gravity = quat_apply(self.base_quat, self.gravity_vec)

        # Detect if the robot is on its back based on positive Z-axis component of the projected gravity
        positive_gravity_z_threshold = 0.0  # Adjust the threshold if needed
        self.fallen = projected_gravity[:, 2] > positive_gravity_z_threshold
        self.reset_buf = torch.where(self.fallen, torch.ones_like(self.reset_buf), self.reset_buf)

        self.out_of_bounds = ((self.base_pos[:, 0] - self.env_origins[:, 0]) < self.bounds[0]) | ((self.base_pos[:, 0] - self.env_origins[:, 0]) > self.bounds[1]) | \
                        ((self.base_pos[:, 1] - self.env_origins[:, 1]) < self.bounds[2]) | ((self.base_pos[:, 1] - self.env_origins[:, 1]) > self.bounds[3])
        self.reset_buf = torch.where(self.out_of_bounds, torch.ones_like(self.reset_buf), self.reset_buf)

        # Check standing still condition every still_check_interval timesteps
        self.standing_still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if not hasattr(self, "still_counter"):
            self.still_counter = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)

        still_lin = (self.base_vel[:, 0].abs() < 0.01)
        still_ang = (self.base_ang_vel[:, 2].abs() < 0.01)
        still_mask = still_lin & still_ang

        self.still_counter[still_mask] += 1
        self.still_counter[~still_mask] = 0

        self.standing_still = (self.still_counter >= 30)
        # Reset the still_counter for environments that have been standing still
        self.still_counter[self.standing_still] = 0

        # print(f"still_counter: {self.still_counter}")

        # Update reset_buf based on standing_still condition
        self.reset_buf = torch.where(self.standing_still, torch.ones_like(self.reset_buf), self.reset_buf)

    
    def calculate_metrics(self) -> None:

        # Efficiency penalty: Penalize large velocities and driving mode mixing
        # Penalize mixing driving modes usefull when climb is active like in a3 environments
        k_mode = -10.0  # Penalty for mixing driving modes
        r_mode = k_mode * ((self.base_vel[:, 0].abs()/1) * (self.base_ang_vel[:, 2].abs()/0.3))  # * self.base_ang_vel[:, 1]**2

        # Check standing still condition every still_check_interval timesteps
        k_still = -0.5  # Penalty for standing still
        self.still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.still_lin = self.base_vel[:, 0].abs() < 0.01 
        self.still_ang = self.base_ang_vel[:, 2].abs() < 0.01
        still = torch.where(self.still_lin & self.still_ang, torch.ones_like(self.still), torch.zeros_like(self.still))
        r_still = k_still * still.float()

        # Sparse Rewards
        target_reached = self.target_reached.float()
        k_tar = self.k_tar  # Completion bonus
        r_tar = k_tar * target_reached

        # Progress reward
        k_prog = self.k_prog
        r_prog = (self.dist_t - self._computed_distance) * k_prog/(self.decimation * self.dt)  

        # Alignment reward
        k_d = self.k_d  # Curvature parameter for the exponential function
        r_head =  torch.exp(-0.5 * (self.yaw_diff / (self._computed_distance / k_d))**2)

        
        # Combine rewards and penalties
        reward = (
            # r_mode    
            # + r_still
            + r_tar
            + r_prog * r_head 
        )

        # print(f"r_mode: {r_mode}")
        # print(f"r_still: {r_still}")
        # print(f"r_tar: {r_tar}")
        # print(f"progress: {self.dist_t - self._computed_distance} [m]")
        # print(f"r_prog: {r_prog}")
        # print(f"self.yaw_diff: {self.yaw_diff} [rad]")
        # print(f"r_head: {r_head}")
        # print(f"reward: {reward}")

        # Store reward components for tracking
        self.reward_components = {
            "r_mode": r_mode.mean().item(),
            "r_still": r_still.mean().item(),
            "r_tar": r_tar.mean().item(),
            "r_prog": r_prog.mean().item(),
            "r_head": r_head.mean().item()
        }
      
        self.rew_buf[:] = reward

        return self.rew_buf


    def get_observations(self):
        ids = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.measured_heights = self.get_heights(env_ids=ids)
        # print(f"measured_heights: {self.measured_heights}")

        heights = (
            torch.clip(self.measured_heights, -1, 1.0) #* self.height_meas_scale
        ) 

        # print(f"heights: {heights}")

        # Count the number of non-zero values in height_samples
        non_zero_count = torch.nonzero(self.height_samples).size(0)
        # print(f"Number of non-zero height samples: {non_zero_count}")

        # print(f"all heightsamples: {self.height_samples}")

        self.refresh_body_state_tensors()
        delta_pos = self.target_pos - self.base_pos

        self.obs_buf = torch.cat(
                (
                    delta_pos[:, 0:2],
                    self.yaw_diff.unsqueeze(-1),
                    self.base_vel[:, 0].unsqueeze(-1),
                    self.base_ang_vel[:, 2].unsqueeze(-1),
                    heights
                ),
                dim=-1,
            )
        
        # print(f"obs_vel: {self.base_vel[:, 0].unsqueeze(-1)}, obs_ang_vel: {self.base_ang_vel[:, 2].unsqueeze(-1)}")

                    
        return {self._robots.name: {"obs_buf": self.obs_buf}}
    

    def get_heights(self, env_ids=None):
        if env_ids.numel() > 0:
            points = quat_apply_yaw(
                self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]
            ) + (self.base_pos[env_ids, 0:3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (
                self.base_pos[:, 0:3]
            ).unsqueeze(1)

        # Add terrain border size
        points += self.terrain.border_size

        # Convert to terrain grid coordinates (account for terrain scaling)
        points = (points / self.terrain.horizontal_scale).long()
        
        # Extract the x and y coordinates for indexing into height_samples
        px = points[:, :, 0].view(-1)  # Flatten x coordinates
        py = points[:, :, 1].view(-1)  # Flatten y coordinates
        
        # Clip the values to stay within the height samples bounds
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)
        
        # Get heights from the height_samples for these coordinates
        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py + 1]
        
        # Use the minimum height as a conservative estimate
        heights = torch.min(heights1, heights2)

        # Return the heights, scaled by the vertical scale
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