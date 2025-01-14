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

from my_utils.terrain_generator import *
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
                             "envSpacing": 10.0,
                             "episodeLength": 300,
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
                            "TerrainType": "rooms", # rooms, stairs, sloped, mixed_v1, mixed_v2, mixed_v3, custom, custom_mixed                         

                            },
                     "sim": {"dt": 0.0083,  # 1 / 120
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
                                      "max_depenetration_velocity": 1000.0,
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
                                       "enable_gyroscopic_forces": True,
                                       "solver_position_iteration_count": 4,
                                       "solver_velocity_iteration_count": 4,
                                       "sleep_threshold": 0.005,
                                       "stabilization_threshold": 0.001,
                                       "density": -1,
                                       "max_depenetration_velocity": 1000.0,
                                       "contact_offset": 0.005,
                                       "rest_offset": 0.0},
                             "target": {"override_usd_defaults": False,
                                        "fixed_base": True,
                                        "make_kinematic": True,
                                        "enable_self_collisions": False,
                                        "enable_gyroscopic_forces": True,
                                        "solver_position_iteration_count": 1,
                                        "solver_velocity_iteration_count": 1,
                                        "sleep_threshold": 0.005,
                                        "stabilization_threshold": 0.001,
                                        "density": -1,
                                        "max_depenetration_velocity": 1000.0,
                                        "contact_offset": 0.005,
                                        "rest_offset": 0.0}}}}

class RobotView(ArticulationView):
    def __init__(self, prim_paths_expr: str, name: str = "robot_view") -> None:
        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)


class ReachingTargetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.height_samples = None
        self.init_done = False
        
        self.dt = 1 / 120.0

        # observation and action space DQN
        self._num_observations = 5  # features (+ height points)
        self._num_actions = 2  # Designed discrete action space see pre_physics_step()

        self.observation_space = spaces.Box(
            low=float("-10"),  # Replace with a specific lower bound if needed
            high=float("10"),  # Replace with a specific upper bound if needed
            shape=(self.num_observations,),
            dtype=np.float32  # Ensure data type is consistent
        )
        # Define the action range for torques
        self.min_torque = -10.0  # Example min torque value
        self.max_torque = 10.0   # Example max torque value


        # Using the shape argument
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32
        )

        self.common_step_counter = 0 # Counter for the first two steps

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        self.height_points = self.init_height_points()  
        self.measured_heights = None
        self.bounds = torch.tensor([-3.0, 3.0, -3.0, 3.0], device=self.device, dtype=torch.float)

        self.episode_buf = torch.zeros(self.num_envs, dtype=torch.long)

        self.linear_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self.angular_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_linear_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_angular_velocity = torch.zeros((self.num_envs, 3), device=self.device)

        self.scaled_actions = torch.zeros(self.num_envs, device=self.device)
        self.scaled_delta_diff = torch.zeros(self.num_envs, device=self.device)
        # self.scaled_delta_climb = torch.zeros(self.num_envs, device=self.device)
        
        return


    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # env config
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_envs = torch.tensor(self._num_envs, dtype=torch.int64)
        self.terrain_type = self._task_cfg["env"]["TerrainType"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dt = self._task_cfg["sim"]["dt"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        torques = self._task_cfg["env"]["dofInitTorques"]
        dof_velocities = self._task_cfg["env"]["dofInitVelocities"]
        self.dof_init_state = torques + dof_velocities

        self.decimation = 4


    def init_height_points(self):
        # 6mx6m rectangle (without center line) 13x13=169 points
        y = 0.5 * torch.tensor(
            [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], device=self.device, requires_grad=False
        )  # 50cm on each side
        x = 0.5 * torch.tensor(
            [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], device=self.device, requires_grad=False
        )  
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    

    def _create_trimesh(self, create_mesh=True):
        self.terrain = Terrain(num_robots=self.num_envs, terrain_type=self.terrain_type)
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
        self.get_target()
        self.get_robot()

        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"], copy_from_source=True)

        # robot view
        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot_*", name="robot_view")
        scene.add(self._robots)
                     
        # food view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)


    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self._create_trimesh(create_mesh=create_mesh)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)


    def get_robot(self):
        robot_translation = torch.tensor([0.0, 0.0, 0.0])
        robot_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.robot_v101 = Robot_v10(
            prim_path=self.default_zero_env_path + "/robot_v10",
            name="robot_v10",
            translation=robot_translation,
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(self.robot_v101.prim_path), self._sim_config.parse_actor_config("robot")
        )
        self.robot_v101.set_robot_properties(self._stage, self.robot_v101.prim)

        
    def get_target(self):
        target = DynamicSphere(prim_path=self.default_zero_env_path + "/target",
                               name="target",
                               radius=0.1,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)


    def post_reset(self):
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_init_state = torch.tensor(self.dof_init_state, dtype=torch.float, device=self.device, requires_grad=False)

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.extras = {}
        
        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        
        self.wheel_torques = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.num_dof = self._robots.num_dof 
        self.env_origins = self.terrain_origins.view(-1, 3)[:self.num_envs]
        self._target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self._target_pos += torch.tensor([2.0, 0.0, 0.1], dtype=torch.float, device=self.device)
        self._target_pos[:, :2] += self.env_origins[:, :2]
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_efforts = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
      
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(indices)
        base_pos, base_quat = self._robots.get_world_poses(clone=False)
        self.dist_t = torch.norm(base_pos - self._target_pos, dim=-1) # Distance to target

        self.init_done = True


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        # Generate random positions for each environment
        num_envs = self.num_envs
        random_positions = []
        for _ in range(num_envs):
            x = random.uniform(-2.5, -0.5)  # Adjust range as needed
            y = random.uniform(-2.25, 2.25)  # Adjust range as needed
            z = 0.15  # Assuming a flat terrain
            random_positions.append([x, y, z])

        # Convert to tensor
        random_positions = torch.tensor(random_positions, device=self.device)

        # Set the initial positions of the environments
        pos = random_positions

        # Generate a random rotation angle around the Z-axis
        theta = random.uniform(-math.pi/2, math.pi/2)  # Angle between -π/2 and π/2
        half_theta = theta / 2.0
        cos_half_theta = math.cos(half_theta)
        sin_half_theta = math.sin(half_theta)

        # Quaternion components in [w, x, y, z] format
        w = cos_half_theta
        x = 0.0
        y = 0.0
        z = sin_half_theta

        # Create the quaternion tensor
        quat = torch.tensor([w, x, y, z], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.dof_vel[env_ids] = self.dof_init_state[4:8]
        self.dof_efforts[env_ids] = self.dof_init_state[0:4]
    
        pos[env_ids, :2] += self.env_origins[env_ids, :2].clone()  # Add only x and y entries from env_origins
        self._robots.set_world_poses(pos[env_ids].clone(), orientations=quat[env_ids].clone(), indices=indices)
        self._robots.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._robots.set_joint_efforts(self.dof_efforts[env_ids].clone(), indices=indices)
        self._robots.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)   

        self._targets.set_world_poses(positions=self._target_pos[env_ids].clone(), indices=indices)

        self.last_actions[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.episode_buf[env_ids] = 0 
        

    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._robots.get_world_poses(clone=False)
        self.base_vel = self._robots.get_velocities(clone=False)
        self.target_pos, _ = self._targets.get_world_poses(clone=False)

        # Extract quaternion components
        w = self.base_quat[:, 0]
        x = self.base_quat[:, 1]
        y = self.base_quat[:, 2]
        z = self.base_quat[:, 3]

        # Compute yaw angle from quaternion
        self.yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y ** 2 + z ** 2))

        # Compute target direction vector (only x and y components)
        target_direction = self.target_pos - self.base_pos  # Shape: [batch_size, 3]
        target_direction_x = target_direction[:, 0]
        target_direction_y = target_direction[:, 1]

        # Compute target angle from target direction
        self.target_yaw = torch.atan2(target_direction_y, target_direction_x)
        self.yaw_diff = (self.target_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi

        # Compute linear and angular velocities in the robot's ergo frame
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_vel[:, 0:3])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_vel[:, 3:6])

        self._computed_distance = torch.norm(self.base_pos - self.target_pos, dim=-1)

        return self.base_lin_vel, self.base_ang_vel


    def pre_physics_step(self, actions):

        if not self.world.is_playing():
            return
        
        # # If we are still in the first two steps, don't apply any action but advance the simulation
        if self.common_step_counter < 2:
            self.common_step_counter += 1
            SimulationContext.step(self.world, render=False)  # Advance simulation
            return 

        self.actions = actions.clone().to(self.device)
        # print(f"actions: {self.actions}")

        # Apply the actions to the robot
        self.min_delta = -10.0
        self.max_delta = 10.0

        self.scaled_actions = self.min_torque + (actions[:, 0] + 1) * 0.5 * (self.max_torque - self.min_torque)
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

        still_lin = (self.base_vel[:, 0].abs() < 0.05)
        still_ang = (self.base_ang_vel[:, 2].abs() < 0.05)
        still_mask = still_lin & still_ang

        self.still_counter[still_mask] += 1
        self.still_counter[~still_mask] = 0

        self.standing_still = (self.still_counter >= 20)

        # print(f"still_counter: {self.still_counter}")

        # Update reset_buf based on standing_still condition
        self.reset_buf = torch.where(self.standing_still, torch.ones_like(self.reset_buf), self.reset_buf)

    
    def calculate_metrics(self) -> None:

        # Efficiency penalty: Penalize large velocities and driving mode mixing
        # Penalize mixing driving modes usefull when climb is active like in a3 environments
        r_mode = -(self.base_vel[:, 0])**2 * (self.base_ang_vel[:, 2]**2)  # * self.base_ang_vel[:, 1]**2

        # Check standing still condition every still_check_interval timesteps
        k_still = -1.0  # Penalty for standing still
        self.still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.still_lin = self.base_vel[:, 0].abs() < 0.05 
        self.still_ang = self.base_ang_vel[:, 2].abs() < 0.05
        still = torch.where(self.still_lin & self.still_ang, torch.ones_like(self.still), torch.zeros_like(self.still))
        r_still = k_still * still.float()

        # Sparse Rewards
        target_reached = self.target_reached.float()
        k_tar = 50.0  # Completion bonus
        r_tar = k_tar * target_reached

        # Progress reward
        k_prog = 5.0
        r_prog = (self.dist_t - self._computed_distance) * k_prog/(self.decimation * self.dt)  

        # Alignment reward
        self.dist_t = self._computed_distance
        k_d = 2.0  # Curvature parameter for the exponential function
        r_head =  torch.exp(-0.5 * (self.yaw_diff / (self.dist_t / k_d))**2)

        
        # Combine rewards and penalties
        reward = (
            r_mode    
            + r_still
            + r_tar
            + r_prog * r_head 
        )

        # print(f"r_mode: {r_mode}")
        # print(f"r_still: {r_still}")
        # print(f"r_tar: {r_tar}")
        # print(f"r_prog: {r_prog}")
        # print(f"self.yaw_diff: {self.yaw_diff}")
        # print(f"r_head: {r_head}")
        # print(f"reward: {reward}")
      
        self.rew_buf[:] = reward

        return self.rew_buf


    def get_observations(self):
        ids = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.measured_heights = self.get_heights(ids)
        heights = self.measured_heights * self.terrain.vertical_scale 

        self.refresh_body_state_tensors()
        delta_pos = self.target_pos - self.base_pos

        self.obs_buf = torch.cat(
                (
                    delta_pos[:, 0:2],
                    self.yaw_diff.unsqueeze(-1),
                    self.base_vel[:, 0].unsqueeze(-1),
                    self.base_ang_vel[:, 2].unsqueeze(-1),
                ),
                dim=-1,
            )
        
        # print(f"obs_buf: {self.obs_buf}")

                    
        return {self._robots.name: {"obs_buf": self.obs_buf}}
    

    def get_heights(self, env_ids=None):
        points = self.height_points[env_ids] + (self.base_pos[env_ids, 0:3]).unsqueeze(1)

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

