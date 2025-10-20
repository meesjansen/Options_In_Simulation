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


from my_robots.origin_v18 import AvularOrigin_v10 as Robot_v10 

from my_utils.origin_terrain_generator import *
from my_utils.terrain_utils import *

TASK_CFG = {"test": False,
            "device_id": 0,
            "headless": True,
            "sim_device": "gpu",
            "enable_livestream": True,
            "warp": False,
            "seed": 42,
            "task": {"name": "TorqueDistributionTask",
                     "physics_engine": "physx",
                     "env": {"numEnvs": 16, 
                             "envSpacing": 3.0,
                             "episodeLength": 500,
                             "enableDebugVis": False,
                             "clipObservations": 1000.0,
                             "controlFrequencyInv": 10,
                             "baseInitState": {"pos": [0.0, 0.0, 0.1], # x,y,z [m]
                                              "rot": [1.0, 0.0, 0.0, 0.0], # w,x,y,z [quat]
                                              "vLinear": [0.0, 0.0, 0.0],  # x,y,z [m/s]
                                              "vAngular": [0.0, 0.0, 0.0],  # x,y,z [rad/s]
                                                },
                            "dofInitTorques": [0.0, 0.0, 0.0, 0.0],
                            "dofInitVelocities": [0.0, 0.0, 0.0, 0.0],
                            "terrain": {"staticFriction": 0.85,  # [-]
                                        "dynamicFriction": 0.85,  # [-]
                                        "restitution": 0.0,  # [-]
                                        # rough terrain only:
                                        "curriculum": False,
                                        "RandSampling": True,
                                        "BoxSampling": False,
                                        "GridSampling": False,
                                        "maxInitMapLevel": 0,
                                        "mapLength": 10.0,
                                        "mapWidth": 10.0,
                                        "numLevels": 6,
                                        "numTerrains": 2,
                                        # tri mesh only:
                                        "slopeTreshold": 0.5,
                                        },
                            "TerrainType": "double room", # rooms, stairs, sloped, mixed_v1, mixed_v2, mixed_v3, custom, custom_mixed      
                            "learn" : {"heightMeasurementScale": 1.0,
                                       "terminalReward": 0.0,
                                       "episodeLength_s": 10.0,}, # [s]
                                       "randomCommandVelocityRanges": {"linear_x":[1.5, 1.5], # [m/s]
                                                                       "linear_y": [-0.5, 0.5], # [m/s]
                                                                       "yaw": [1.0, 1.1], # [rad/s]
                                                                       "yaw_constant": 0.5,},   # [rad/s]
                            "control": {"decimation": 10, # decimation: Number of control action updates @ sim DT per policy DT
                                        "stiffness": 1.0, # [N*m/rad] For torque setpoint control
                                        "damping": .005, # [N*m*s/rad]
                                        "actionScale": 3.0,
                                        "wheel_radius": 0.1175,
                                        },   # leave room to overshoot or corner 
                            },
                     "sim": {"dt": 0.01, # 600 Hz + PGS for skid steer dynamics
                             "use_gpu_pipeline": True,
                             "gravity": [0.0, 0.0, -9.81],
                             "add_ground_plane": True,
                             "add_distant_light": True,
                             "use_flatcache": True,
                             "enable_scene_query_support": False,
                             "enable_cameras": False,
                             "default_physics_material": {"static_friction": 1.0,
                                                         "dynamic_friction": 1.0,
                                                         "restitution": 0.0},
                             "physx": {"worker_thread_count": 4,
                                      "solver_type": 0, # 0: PGS, 1: TGS
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

class TorqueDistributionTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.height_samples = None
        self.custom_origins = False
        self.init_done = False
        self._env_spacing = 0.0
        
       # --- KA-DDPG/KA-PPO modifications: use a state space of 4 and action space of 4 ---
        self._num_observations = 6  # [desired_v, desired_omega, v_delta, omega_delta, linear_acc, angular_acc]
        self._num_actions = 4       # [T_fl, T_rl, T_fr, T_rr]
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._num_observations,),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._num_actions,),
            dtype=np.float32
        )

        # ---------------------------------------------------------------------------
        # Add parameters for the low-fidelity controller (criteria action)
        # You can also move these to the config if desired.
        self.vehicle_mass = 25.0      # [kg]
        self.vehicle_inertia = 1.05    # [kg·m^2]
        # Initialize a max global episode counter for gamma scheduling
        # or a fixed number of episodes needed for the curriculum levels
        self.max_global_episodes = 700.0
        self.max_sim_steps = 700000.0 # 250 episodes of 10s at 100Hz sim and 10Hz control/policy step
        # ---------------------------------------------------------------------------
        

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        self.bounds = torch.tensor([-50.0, 50.0, -50.0, 50.0], device=self.device, dtype=torch.float)

        self.sim_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.episode_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gamma_assist = torch.ones(self.num_envs, dtype=torch.float)

        self.linear_acc = torch.zeros((self.num_envs, 1), device=self.device)
        self.angular_acc = torch.zeros((self.num_envs, 1), device=self.device)
        self.previous_linear_velocity = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.previous_angular_velocity = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        self.v_forward_projected = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.v_lateral_projected = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.v_upward_projected = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {
            "r1: Tracking error reward (squared errors)": torch_zeros(),
            "r2: Convergence reward (squared accelerations)": torch_zeros(),
            "r3: Torque penalty (sum of squared torques)": torch_zeros(),
            "Dense reward": torch_zeros(),
            "Sparse reward": torch_zeros(),
            "Guiding reward": torch_zeros(),
            "Observed reward": torch_zeros(),
            "Final reward": torch_zeros(),
            "r1/Final reward": torch_zeros(),
            "r2/Final reward": torch_zeros(),
            "r3/Final reward": torch_zeros(),
            "Dense reward/Final reward": torch_zeros(),
            "Sparse reward/Final reward": torch_zeros(),
            "Guiding reward/Final reward": torch_zeros(),
            "Observed reward/Final reward": torch_zeros(),
              }
        
        self.terrain_levels = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self.phase_name = ""

        
        return
        

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # normalization
        self.height_meas_scale = self._task_cfg["env"]["learn"]["heightMeasurementScale"]
        self.action_scale = self._task_cfg["env"]["control"]["actionScale"]

        # reward scales
        self.rew_scales = {}
        self.rew_scales["termination"] = self._task_cfg["env"]["learn"]["terminalReward"]

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
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.curriculum = self._task_cfg["env"]["terrain"]["curriculum"]
        self.randsampling = self._task_cfg["env"]["terrain"]["RandSampling"]
        self.boxsampling = self._task_cfg["env"]["terrain"]["BoxSampling"]
        self.gridsampling = self._task_cfg["env"]["terrain"]["GridSampling"]
        self.threshold_high = 10.0
      
        # env config
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        init_torques = self._task_cfg["env"]["dofInitTorques"]
        init_dof_velocities = self._task_cfg["env"]["dofInitVelocities"]
        self.dof_init_state = init_torques + init_dof_velocities


    # def init_height_points(self):
    #     # 1mx1.6m rectangle (without center line)
    #     y = 0.1 * torch.tensor(
    #         [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5], device=self.device, requires_grad=False
    #     )  # 10-50cm on each side
    #     x = 0.1 * torch.tensor(
    #         [-8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False
    #     )  # 20-80cm on each side
    #     grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    #     self.num_height_points = grid_x.numel()
    #     points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
    #     points[:, :, 0] = grid_x.flatten()
    #     points[:, :, 1] = grid_y.flatten()
    #     return points

  

    # def _create_trimesh(self, create_mesh=True):
    #     self.terrain = Terrain(self._task_cfg["env"]["terrain"], num_robots=self.num_envs)
    #     vertices = self.terrain.vertices
    #     triangles = self.terrain.triangles
    #     position = torch.tensor([-self.terrain.border_size, -self.terrain.border_size, 0.0])
    #     if create_mesh:
    #         add_terrain_to_stage(stage=self._stage, vertices=vertices, triangles=triangles, position=position)
    #     self.height_samples = (
    #         torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
    #     )

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        # self.get_terrain()
        self.get_robot()
        # print_stage_prim_paths(self._stage)
        super().set_up_scene(scene)

        # robot view
        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        scene.add(self._robots)
        scene.add(self._robots._base)

        return

 
    def initialize_views(self, scene):
        # initialize terrain variables even if we do not need to re-create the terrain mesh
        # self.get_terrain(create_mesh=False)

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
      

    # def get_terrain(self, create_mesh=True):
    #     self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
    #     if not self.curriculum:
    #         self._task_cfg["env"]["terrain"]["maxInitMapLevel"] = self._task_cfg["env"]["terrain"]["numLevels"] - 1
    #     self.terrain_levels = torch.randint(
    #         0, self._task_cfg["env"]["terrain"]["maxInitMapLevel"] + 1, (self.num_envs,), device=self.device
    #     )
    #     self.terrain_types = torch.randint(
    #         0, self._task_cfg["env"]["terrain"]["numTerrains"], (self.num_envs,), device=self.device
    #     )
    #     self._create_trimesh(create_mesh=create_mesh)
    #     self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

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
        self.episode_step_counter = 0
        self.extras = {}
        self.commands = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )  # x vel, y vel, yaw vel, heading
            
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
        
        # for i in range(self.num_envs):
            # self.env_origins[i] = self.terrain_origins[self.terrain_levels[i], self.terrain_types[i]]

        self.num_dof = self._robots.num_dof 
        self.dof_pos = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_effort = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.v_forward_projected = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.v_lateral_projected = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.v_upward_projected = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.previous_linear_velocity = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.previous_angular_velocity = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(indices)
        self.init_done = True
        

    def reset_idx(self, env_ids):
        
        indices = env_ids.to(dtype=torch.int32)

        # self.update_SI_level(env_ids)
        self.base_pos[env_ids] = self.base_init_state[0:3]
        # self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
        self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        # Apply random rotation around Z-axis
        random_angle = torch_rand_float(0, 2 * torch.pi, (len(env_ids), 1), device=self.device).squeeze(-1)           # squeeze out the extra dimension
        quat_z = torch.stack([
                            torch.cos(random_angle / 2),
                            torch.zeros_like(random_angle),
                            torch.zeros_like(random_angle),
                            torch.sin(random_angle / 2)
                        ], dim=-1)

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

        self.episode_buf[env_ids] = 0 
        self.episode_count[env_ids] += 1

        one = torch.tensor(1.0, device=self.device)
        hundred = torch.tensor(100.0, device=self.device)
        self.gamma_assist = self.gamma_assist.to(device=self.device)

        self.episode_sums["r1/Final reward"] = self.max_episode_length_s * (hundred * (one - self.gamma_assist) * self.episode_sums["r1: Tracking error reward (squared errors)"] / self.episode_sums["Final reward"])
        self.episode_sums["r2/Final reward"] = self.max_episode_length_s * (hundred * (one - self.gamma_assist) * self.episode_sums["r2: Convergence reward (squared accelerations)"] / self.episode_sums["Final reward"])
        self.episode_sums["r3/Final reward"] = self.max_episode_length_s * (hundred * (one - self.gamma_assist) * self.episode_sums["r3: Torque penalty (sum of squared torques)"] / self.episode_sums["Final reward"])
        self.episode_sums["Dense/Final reward"] = self.max_episode_length_s * (hundred * (one - self.gamma_assist) * self.episode_sums["Dense reward"] / self.episode_sums["Final reward"])
        self.episode_sums["Sparse/Final reward"] = self.max_episode_length_s * (hundred * (one - self.gamma_assist) * self.episode_sums["Sparse reward"] / self.episode_sums["Final reward"])
        self.episode_sums["Guiding/Final reward"] = self.max_episode_length_s * (hundred * self.gamma_assist * self.episode_sums["Guiding reward"] / self.episode_sums["Final reward"])
        self.episode_sums["Observed/Final reward"] = self.max_episode_length_s * (hundred * (one - self.gamma_assist) * self.episode_sums["Observed reward"] / self.episode_sums["Final reward"])


        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        self.extras["episode"]["gamma assist"] = torch.mean(self.gamma_assist.float())
        self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

        if not self.curriculum:
            for i in env_ids:
                cmds = self.sample_velocity_command(i)
                x_cmd = cmds[0]
                omega_cmd = cmds[1]
                self.commands[i,0] = x_cmd
                self.commands[i,2] = omega_cmd

        self.fraction = self.episode_sums["r1: Tracking error reward (squared errors)"] / self.threshold_high 



    # def update_SI_level(self, env_ids):
    #     # Only update terrain if initialization and curriculum are active.
    #     if not self.init_done or not self.curriculum:
    #         return

    #     # Use self.episode_sums as the cumulative performance indicator for each environment.
    #     # Define thresholds (tune these values as needed)
    #     threshold_low = 0.3   # indicates poor performance, make terrain easier (reduce difficulty)
    #     threshold_high = 5.0  # indicates strong performance, increase terrain difficulty

    #     # Create boolean masks based on episode sums for the selected env_ids.
    #     low_mask = self.episode_sums["r1: Tracking error reward (squared errors)"][env_ids] < threshold_low
    #     high_mask = self.episode_sums["r1: Tracking error reward (squared errors)"][env_ids] > threshold_high
        
    #     # For environments with low performance, decrease the terrain difficulty.
    #     if low_mask.any():
    #         self.terrain_levels[env_ids][low_mask] = torch.clamp(
    #             self.terrain_levels[env_ids][low_mask] - 1, min=0
    #         )
        
    #     # For environments with high performance, increase the terrain difficulty.
    #     if high_mask.any():
    #         self.terrain_levels[env_ids][high_mask] = torch.clamp(
    #             self.terrain_levels[env_ids][high_mask] + 1, max=5
    #         )
        
    #     # Finally, update the environment origins according to the new terrain level and terrain type.
    #     self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
            
    def sample_velocity_command(self, env_id: int):
        
        if self.curriculum:
            if self.terrain_levels[env_id] == 0:
                # Task 1: normal distribution around 0.5, with sigma in [0.01..0.1]
                # Example: linearly scale sigma with episode_buf or a global counter
                fraction = self.episode_sums["r1: Tracking error reward (squared errors)"][env_id] / self.threshold_high 
                sigma = 0.01 + 0.09 * self.fraction[env_id]
                x_vel = torch.normal(mean=0.5, std=sigma, size=(1,), device=self.device).item()
                return 0.0, max(x_vel, 0.0)  # max(x_vel, 0.0), 0.0

            elif self.terrain_levels[env_id] == 1:
                # Task 2: sinusoidal with mean=1, frequency + amplitude changes
                # Suppose we let the frequency grow from 0.01..0.1 and amplitude from 0.1..1
                fraction = self.episode_sums["r1: Tracking error reward (squared errors)"][env_id] / self.threshold_high 
                freq = 0.01 + 0.09 * self.fraction[env_id]
                amp  = 0.1  
                if self.fraction[env_id] > 0.5:
                    amp = 0.1 + 0.4  * self.fraction[env_id]
                t = float(self.episode_step_counter) * self.dt
                x_vel = 0.5 + amp * math.sin(freq * t)
                return max(x_vel, 0.0), 0.0

            elif self.terrain_levels[env_id] == 2:
                # Task 3: range 0..10. Start with 0..5, then up to 10
                # We'll do a simple sub-task switch
                fraction = self.episode_sums["r1: Tracking error reward (squared errors)"][env_id] / self.threshold_high 
                t = float(self.episode_step_counter) * self.dt
                scale = 0.5 # m/s
                Noise = 0.5 * self.fraction[env_id]
                x_vel = torch.normal(mean=0.0, std=Noise, size=(1,), device=self.device).item() + scale * t/self.max_episode_length_s
                return max(x_vel, 0.0), 0.0

            else:
                # Task 4 (or any default): final terrain steps/slopes, normal(0.5,0.1)
                x_vel = torch.normal(mean=0.5, std=0.1, size=(1,), device=self.device).item()
                return max(x_vel, 0.0), 0.0
            
        elif self.randsampling:
            # Random command generation
            x_vel = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (1,1), device=self.device).squeeze()
            omega = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (1,1), device=self.device).squeeze()
            # x_vel = 0.0 # max 1.0
            omega = 0.0 # max 1.0
            return max(x_vel, 0.0), omega
        
        elif self.boxsampling:
            # Box sampling
            
            return max(x_vel, 0.0), omega
        
        elif self.gridsampling:
            # Grid sampling
            
            return max(x_vel, 0.0), omega
        
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

        # Compute state errors for the low-fidelity controller:
        # Desired longitudinal speed and yaw rate from commands:
        self.desired_v = self.commands[:, 0]
        self.desired_omega = self.commands[:, 2]
        # Current velocities: assume base_velocities has linear (index 0) and angular yaw (index 5)
        self.current_v = self.v_forward_projected
        self.current_omega = self.base_velocities[:, 5]
        self.v_delta = self.desired_v - self.current_v
        self.omega_delta = self.desired_omega - self.current_omega


        self.Kp_omega = 0.665
        # Compute criteria actions for each wheel:
        # Left wheels get: Kp * ( (m*v_delta/dt) - (J*omega_delta/dt) )
        # Right wheels get: Kp * ( (m*v_delta/dt) + (J*omega_delta/dt) )
        self.ac_left = self.Kp * (self.vehicle_mass * (self.v_delta / self.dt)) - self.Kp_omega * (self.vehicle_inertia * (self.omega_delta / self.dt))
        # self.ac_left = self.Kp_omega * (- (self.vehicle_inertia * (self.omega_delta / self.dt)))
        self.ac_right = self.Kp * (self.vehicle_mass * (self.v_delta / self.dt)) + self.Kp_omega * (self.vehicle_inertia * (self.omega_delta / self.dt))
        # self.ac_right = self.Kp_omega * (self.vehicle_inertia * (self.omega_delta / self.dt))


        # Build criteria action vector: [T_fl, T_rl, T_fr, T_rr]
        criteria_action = torch.stack([self.ac_left, self.ac_left, self.ac_right, self.ac_right], dim=1).to(self.device)

        # Compute gamma_assist (decaying assistance) based on global_episode
        self.gamma_assist = torch.clamp(1.0 - (self.sim_steps.float() / self.max_sim_steps), min=0.0).to(self.device)

        # Compute execution action: blend agent action and criteria action
        gamma = self.gamma_assist.view(-1, 1).to(self.device)
        execution_action = (torch.tensor(1.0, device=self.device) - gamma) * self.actions * self.action_scale + gamma * criteria_action

        # print("pre_physics; gamma_assist: ", self.gamma_assist[0])
        # print("pre_physics; self.episode_count.float(): ", self.episode_count.float()[0])
        # print("pre_physics; gamma: ", gamma[0])



        # Compute guiding reward: negative Euclidean distance between agent and criteria actions
        self.guiding_reward = -torch.norm(self.actions * self.action_scale - criteria_action, dim=1).to(self.device)
        self.guiding_reward = self.guiding_reward


        # Apply the blended execution action as torques (assumed direct mapping)
        self.torques = execution_action
        # self.torques = criteria_action


        # # Retrieve the ordered DOF names from your RobotView
        # dof_names = self._robots.dof_names
        # # Print the index and name for each DOF
        # for i, name in enumerate(dof_names):
        #     print(f"DOF index: {self._robots.get_dof_index(name)}, name: {name}")


        for _ in range(self.decimation):
            if self.world.is_playing():
                
                self.wheel_torqs = torch.clip(self.torques, -4.0, 4.0)

                self._robots.set_joint_efforts(self.wheel_torqs)

                SimulationContext.step(self.world, render=False)

        # if hasattr(self, "memory") and len(self.memory) > 0:
        #     print(f"[FIFO DEBUG] Memory length: {len(self.memory)}")
        #     try:
        #         # Print first few 'states' to verify overwrite
        #         states = self.memory.get_tensor_by_name("states", keepdim=False)
        #         print(f"[FIFO DEBUG] Oldest state: {states[0].cpu().numpy()}")
        #         print(f"[FIFO DEBUG] Newest state: {states[len(self.memory) - 1].cpu().numpy()}")
        #     except Exception as e:
        #         print(f"[FIFO DEBUG] Error accessing memory tensor: {e}")

        # print(f"[FIFO DEBUG] memory_index = {self.memory.memory_index}, filled = {self.memory.filled}")

        # print("pre_physics; applied efforts: ", self._robots.get_applied_joint_efforts(clone=False))
        # print("pre_physics; dof vel: ", self._robots.get_joint_velocities(clone=False))

        # print("pre_physics; actions, still x100 for self.action_scale: ", self.actions[0])
        # print("pre_physics; desired_v: ", self.desired_v[0])
        # print("pre_physics; current_v: ", self.current_v[0])
        # print("pre_physics; desired_omega: ", self.desired_omega[0])
        # print("pre_physics; current_omega: ", self.current_omega[0])
        # print("pre_physics; expert torques left: ", self.ac_left[0])
        # print("pre_physics; expert torques right: ", self.ac_right[0])
        # print("pre_physics; executed torques pre clip: ", self.torques[0])
        # print("pre_physics; executed torques post clip: ", self.wheel_torqs[0])
        # print("base velocitites in z: ", self.base_velocities[0, 2])

          
    def post_physics_step(self):
        self.episode_buf[:] += 1
        self.sim_steps += 1
        
       
        if self.world.is_playing():
            
            self.refresh_dof_state_tensors()
            self.refresh_body_state_tensors()

            self.episode_step_counter += 1


            self.is_done()
            self.get_states()
            self.calculate_metrics()
            
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
          
        # prepare quantities
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        forward = quat_apply(self.base_quat, self.forward_vec)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        
        forward_norm = forward / torch.norm(forward, dim=1, keepdim=True)
        lateral_dir = torch.stack([-forward_norm[:, 1], forward_norm[:, 0], torch.zeros_like(forward_norm[:, 0])], dim=1)
        up_local = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))
        up_dir = quat_apply(self.base_quat, up_local)

        v_global = self.base_velocities[:, 0:3]  # shape: [num_env, 3]
        self.v_forward_projected = torch.sum(v_global * forward_norm, dim=1)   # dot product, shape: [num_env]
        self.v_lateral_projected = torch.sum(v_global * lateral_dir, dim=1)      # dot product, shape: [num_env]
        self.v_upward_projected = torch.sum(v_global * up_dir, dim=1)

        # If evaluating heading can determine the yaw rate like below
        # self.commands[:, 2] = torch.clip(self.yaw_constant * wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)

        # sample velocity commands (x, y, yaw, heading)
        # Here we only do x velocity changes from sample_velocity_command
        if self.curriculum:
            for i in range(self._num_envs):
                cmds = self.sample_velocity_command(i)
                x_cmd = cmds[0]
                omega_cmd = cmds[1]
                self.commands[i,0] = x_cmd
                self.commands[i,2] = omega_cmd

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
        # print("episode buf: ", self.episode_buf[0])
        # print("max episode length: ", self.max_episode_length)
        
        # Calculate the projected gravity in the robot's local frame
        projected_gravity = quat_apply(self.base_quat, self.gravity_vec)

        # Detect if the robot is on its back based on positive Z-axis component of the projected gravity
        positive_gravity_z_threshold = 0.0  # Adjust the threshold if needed
        self.has_fallen = projected_gravity[:, 2] > positive_gravity_z_threshold
        self.reset_buf = self.has_fallen.clone()

        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)

        self.out_of_bounds = ((self.base_pos[:, 0] - self.env_origins[:, 0]) < self.bounds[0]) | ((self.base_pos[:, 0] - self.env_origins[:, 0]) > self.bounds[1]) | \
                        ((self.base_pos[:, 1] - self.env_origins[:, 1]) < self.bounds[2]) | ((self.base_pos[:, 1] - self.env_origins[:, 1]) > self.bounds[3])
        self.reset_buf = torch.where(self.out_of_bounds, torch.ones_like(self.reset_buf), self.reset_buf)

        num_timeout = torch.count_nonzero(self.timeout_buf).item()
        num_has_fallen = torch.count_nonzero(self.has_fallen).item()
        num_out_of_bounds = torch.count_nonzero(self.out_of_bounds).item()
        # print(f"Nonzero timeout_buf: {num_timeout}, has_fallen: {num_has_fallen}, out_of_bounds: {num_out_of_bounds}")

    def calculate_metrics(self) -> None:
        
        # Compute linear and angular accelerations (finite differences)
        self.linear_acc = (self.v_forward_projected - self.previous_linear_velocity[:, 0]) / self.dt
        self.angular_acc = (self.base_velocities[:, 5] - self.previous_angular_velocity[:, 2]) / self.dt

        # Update previous velocity records
        self.previous_linear_velocity[:, 0] = self.v_forward_projected
        self.previous_angular_velocity[:, 2] = self.base_velocities[:, 5]

        # Compute reward components:
        # r1: Tracking error reward (squared errors)
        r1 = self.v_delta ** 2 + 5 * (self.omega_delta ** 2)
        # r2: Convergence reward (squared accelerations)
        r2 = self.linear_acc ** 2 + self.angular_acc ** 2
        # r3: Torque penalty (sum of squared torques)
        r3 = torch.sum(self.wheel_torqs ** 2, dim=1)
        # Weight factors (tunable)
        w1, w2, w3 = -25.0, -0.02, -0.09
        rdense = w1 * r1 + w2 * r2 + w3 * r3

        # Sparse reward: bonus if tracking errors are very low
        sparse_reward = torch.where(
            (torch.abs(self.v_delta) < 0.01) &
            (torch.abs(self.omega_delta) < 0.01 ),
            torch.full_like(self.v_delta, 0.2),
            torch.zeros_like(self.v_delta)
        )
        observed_reward = rdense + sparse_reward

        # Final updating reward: blend observed reward with guiding reward
        self.rew_buf = (1 - self.gamma_assist) * observed_reward.to(self.device) + self.gamma_assist * self.guiding_reward
        
        
        self.rew_buf += self.rew_scales["termination"] * self.reset_buf * ~self.timeout_buf

        self.episode_sums["r1: Tracking error reward (squared errors)"] += w1 * r1
        self.episode_sums["r2: Convergence reward (squared accelerations)"] += w2 * r2
        self.episode_sums["r3: Torque penalty (sum of squared torques)"] += w3 * r3
        self.episode_sums["Dense reward"] += rdense
        self.episode_sums["Sparse reward"] += sparse_reward
        self.episode_sums["Guiding reward"] += self.guiding_reward
        self.episode_sums["Observed reward"] += observed_reward
        self.episode_sums["Final reward"] += self.rew_buf

        self.comp_1 = w1 * r1
        self.comp_2 = w2 * r2
        self.comp_3 = w3 * r3
        self.rdense = rdense
        self.rsparse = sparse_reward
        self.robs = observed_reward
        self.rguide = self.guiding_reward
        self.rfinal = self.rew_buf

                       
        return self.rew_buf


    def get_observations(self):
        # self.measured_heights = self.get_heights()
        # heights = (
        #     torch.clip(self.base_pos[:, 2].unsqueeze(1) - self.measured_heights - 0.0622, -1, 1.0) * self.height_meas_scale
        # )
        

        # New observation: 4D vector per environment
        self.obs_buf = torch.cat([self.desired_v.unsqueeze(1), self.desired_omega.unsqueeze(1), self.v_delta.unsqueeze(1), self.omega_delta.unsqueeze(1), self.linear_acc.unsqueeze(1), self.angular_acc.unsqueeze(1)], dim=1)
        # print("self.v_delta[0]", self.v_delta[0])
        # print("self.omega_delta[0]", self.omega_delta[0])
        # print("self.linear_acc[0]", self.linear_acc[0])
        # print("self.angular_acc[0]", self.angular_acc[0])

        # Update when logging other components to wandb
        self.observed_components = {
                    "env0_desired_v": self.desired_v[0].item(),
                    "env0_current_v": self.current_v[0].item(),
                    "env0_desired_omega": self.desired_omega[0].item(),
                    "env0_current_omega": self.current_omega[0].item(),
                    "env0_v_delta": self.v_delta[0].item(),
                    "env0_omega_delta": self.omega_delta[0].item(),
                    "env0_linear_acc": self.linear_acc[0].item(),
                    "env0_angular_acc": self.angular_acc[0].item(), 
                    "env0_episode_count": self.episode_count[0].item(),
                    "env0_torque_apl_fl": self.torques[0, 0].item(),   
                    "env0_torque_apl_rl": self.torques[0, 1].item(),
                    "env0_torque_apl_fr": self.torques[0, 2].item(),
                    "env0_torque_apl_rr": self.torques[0, 3].item(),
                    "env0_exp_left": self.ac_left[0].item(),
                    "env0_exp_right": self.ac_right[0].item(),
                    "env0_policy_torque_fl": self.action_scale * self.actions[0, 0].item(),
                    "env0_policy_torque_rl": self.action_scale * self.actions[0, 1].item(),
                    "env0_policy_torque_fr": self.action_scale * self.actions[0, 2].item(),
                    "env0_policy_torque_rr": self.action_scale * self.actions[0, 3].item(),
                    "env0_perc_r1": 100.0 * (1 - self.gamma_assist[0].item()) * self.comp_1[0].item()/self.rew_buf[0].item(),
                    "env0_perc_r2": 100.0 * (1 - self.gamma_assist[0].item()) * self.comp_2[0].item()/self.rew_buf[0].item(),
                    "env0_perc_r3": 100.0 * (1 - self.gamma_assist[0].item()) * self.comp_3[0].item()/self.rew_buf[0].item(),
                    "env0_perc_dense": 100.0 * (1 - self.gamma_assist[0].item()) * self.rdense[0].item()/self.rew_buf[0].item(),
                    "env0_perc_sparse": 100.0 * (1 - self.gamma_assist[0].item()) * self.rsparse[0].item()/self.rew_buf[0].item(),
                    "env0_perc_observed": 100.0 * (1 - self.gamma_assist[0].item()) * self.robs[0].item()/self.rew_buf[0].item(),
                    "env0_perc_guiding": 100.0 * self.gamma_assist[0].item() * self.rguide[0].item()/self.rew_buf[0].item(),        
                }
                          
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




