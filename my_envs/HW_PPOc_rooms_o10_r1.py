import torch
import numpy as np
import gym
from gym import spaces

from my_envs.rl_task_HW import RLTask 

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


# from my_robots.origin_v10 import AvularOrigin_v10 as Robot_v10
# from my_robots.origin_v11_fixed_arms import AvularOrigin_v10 as Robot_v11
from my_robots.origin_elevated import AvularOrigin_v10 as Robot_v11
from my_robots.origin_v12 import AvularOrigin_v10 as Robot_v12


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
        self._num_observations = 10  # features (+ height points)
        self._num_actions = 4  # Designed discrete action space see pre_physics_step()

        self.observation_space = spaces.Box(
            low=float("-50"),  # Replace with a specific lower bound if needed
            high=float("50"),  # Replace with a specific upper bound if needed
            shape=(self.num_observations,),
            dtype=np.float32  # Ensure data type is consistent
        )
        # Define the action range for torques
        self.min_torque = -7.5  # Example min torque value
        self.max_torque = 7.5   # Example max torque value


        # Using the shape argument
        self.action_space = spaces.Box(
            low=self.min_torque,
            high=self.max_torque,
            shape=(self.num_actions,),
            dtype=np.float32
        )

        self.common_step_counter = 0 # Counter for the first two steps

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        self.height_points = self.init_height_points()  
        self.measured_heights = None
        self.bounds = torch.tensor([-3.0, 3.0, -3.0, 3.0], device=self.device, dtype=torch.float)

        self.still_steps = torch.zeros(self.num_envs)
        self.position_buffer = torch.zeros(self.num_envs, 2)  # Assuming 2D position still condition
        self.counter = 0 # still condition counter
        self.episode_buf = torch.zeros(self.num_envs, dtype=torch.long)

        self.linear_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self.angular_acceleration = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_linear_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_angular_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        
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
        self.dof_init_state_el = torques + torques + dof_velocities + dof_velocities

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
        
        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"], copy_from_source=True)

        # self.get_robot()
        print_stage_prim_paths()


        # # robot view
        # self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot_*", name="robots_view")
        # scene.add(self._robots)
        # # self._robots_v10 = RobotView(prim_paths_expr="/World/envs/.*/robot_f*", name="robot_view")
        # # scene.add(self._robots_v10)
        # # self._robots_elevated = RobotView(prim_paths_expr="/World/envs/.*/robot_e*", name="robot_view_elevated")
        # # scene.add(self._robots_elevated)
                     
        # food view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)


    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self._create_trimesh(create_mesh=create_mesh)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)


    def get_robot(self):
        
        robot_translation = torch.tensor([-2.0, -2.0, 0.0])
        robot_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        # self.robot_v101 = Robot_v10(
        #     prim_path=self.default_zero_env_path + "/robot_f10",
        #     name="robot_f10",
        #     translation=robot_translation,
        #     orientation=robot_orientation,
        # )
        # self._sim_config.apply_articulation_settings(
        #     "robot_f10", get_prim_at_path(self.robot_v101.prim_path), self._sim_config.parse_actor_config("robot")
        # )
        # self.robot_v101.set_robot_properties(self._stage, self.robot_v101.prim)

        # self.robot_v102 = Robot_v10(
        #     prim_path="/World/envs/env_1/robot_f10",
        #     name="robot_f10",
        #     translation=robot_translation + torch.tensor([0.0, 4.0, 0.0]),
        #     orientation=robot_orientation,
        # )
        # self._sim_config.apply_articulation_settings(
        #     "robot_f10", get_prim_at_path(self.robot_v102.prim_path), self._sim_config.parse_actor_config("robot")
        # )
        # self.robot_v102.set_robot_properties(self._stage, self.robot_v102.prim)

        # self.robot_v103 = Robot_v10(
        #     prim_path="/World/envs/env_2/robot_f10",
        #     name="robot_f10",
        #     translation=robot_translation + torch.tensor([4.0, 4.0, 0.0]),
        #     orientation=robot_orientation,
        # )
        # self._sim_config.apply_articulation_settings(
        #     "robot_f10", get_prim_at_path(self.robot_v103.prim_path), self._sim_config.parse_actor_config("robot")
        # )
        # self.robot_v103.set_robot_properties(self._stage, self.robot_v103.prim)



        self.robot_v111 = Robot_v11(
            prim_path="/World/envs/env_0/robot_v11",
            name="robot_v11",
            translation=robot_translation + torch.tensor([0.0, 0.0, 0.0]),
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot_v11", get_prim_at_path(self.robot_v111.prim_path), self._sim_config.parse_actor_config("robot")
        )
        self.robot_v111.set_robot_properties(self._stage, self.robot_v111.prim)

        self.robot_v112 = Robot_v11(
            prim_path="/World/envs/env_1/robot_v11",
            name="robot_v11",
            translation=robot_translation + torch.tensor([0.0, 4.0, 0.0]),
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot_v11", get_prim_at_path(self.robot_v112.prim_path), self._sim_config.parse_actor_config("robot")
        )
        self.robot_v112.set_robot_properties(self._stage, self.robot_v112.prim)

        self.robot_v113 = Robot_v11(
            prim_path="/World/envs/env_2/robot_v11",
            name="robot_v11",
            translation=robot_translation + torch.tensor([4.0, 4.0, 0.0]),
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot_v11", get_prim_at_path(self.robot_v113.prim_path), self._sim_config.parse_actor_config("robot")
        )
        self.robot_v113.set_robot_properties(self._stage, self.robot_v113.prim)



        self.robot_v121 = Robot_v12(
            prim_path="/World/envs/env_3/robot_v12",
            name="robot_v12",
            translation=robot_translation + torch.tensor([0.0, 0.0, 0.0]),
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot_v12", get_prim_at_path(self.robot_v121.prim_path), self._sim_config.parse_actor_config("robot")
        )
        self.robot_v121.set_robot_properties(self._stage, self.robot_v121.prim)

        self.robot_v122 = Robot_v12(
            prim_path="/World/envs/env_4/robot_v12",
            name="robot_v12",
            translation=robot_translation + torch.tensor([0.0, 4.0, 0.0]),
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot_v12", get_prim_at_path(self.robot_v122.prim_path), self._sim_config.parse_actor_config("robot")
        )
        self.robot_v122.set_robot_properties(self._stage, self.robot_v122.prim)

        self.robot_v123 = Robot_v12(
            prim_path="/World/envs/env_5/robot_v12",
            name="robot_v12",
            translation=robot_translation + torch.tensor([4.0, 4.0, 0.0]),
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot_v12", get_prim_at_path(self.robot_v123.prim_path), self._sim_config.parse_actor_config("robot")
        )
        self.robot_v123.set_robot_properties(self._stage, self.robot_v123.prim)

        # robot view
        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot_*", name="robots_view")
        self.scene.add(self._robots)
        # self._robots_v10 = RobotView(prim_paths_expr="/World/envs/.*/robot_f*", name="robot_view")
        # scene.add(self._robots_v10)
        # self._robots_elevated = RobotView(prim_paths_expr="/World/envs/.*/robot_e*", name="robot_view_elevated")
        # scene.add(self._robots_elevated)

        
    def get_target(self):
        target = DynamicSphere(prim_path=self.default_zero_env_path + "/target",
                               name="target",
                               radius=0.05,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)


    def post_reset(self):
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_init_state = torch.tensor(self.dof_init_state, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_init_state_el = torch.tensor(self.dof_init_state_el, dtype=torch.float, device=self.device, requires_grad=False)

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # initialize some data used later on
        self.up_axis_idx = 2
        self.extras = {}
        # self.noise_scale_vec = self._get_noise_scale_vec(self._task_cfg)
        
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
        # self.num_dof_el = self._robots_elevated.num_dof

        self.env_origins = self.terrain_origins.view(-1, 3)[:self.num_envs]
        print(self.env_origins)
        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_pos += torch.tensor([0.0, 0.0, 0.1], dtype=torch.float, device=self.device)
        print(self.target_pos)
        self.target_pos[:, :2] += self.env_origins[:, :2]
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        # self.dof_vel_el = torch.zeros((self.num_envs, self.num_dof_el), dtype=torch.float, device=self.device)
        
        self.dof_efforts = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        # self.dof_efforts_el = torch.zeros((self.num_envs, self.num_dof_el), dtype=torch.float, device=self.device)

        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(indices)
        base_pos, base_quat = self._robots.get_world_poses(clone=False)
        self.last_distance_to_target = torch.norm(base_pos - self.target_pos, dim=-1)

        self.init_done = True


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        # Define square boundary size with some margin to reduce instant resets
        square_size_x = 4.5  # Total width of the square
        square_size_y = 4.5  # Total length of the square

        edge = random.randint(0, 3)

        # Generate x and y positions based on the edge
        if edge == 0:  # Left edge
            x_pos = -square_size_x / 2
            y_pos = random.uniform(-square_size_y / 2, square_size_y / 2)
        elif edge == 1:  # Right edge
            x_pos = square_size_x / 2
            y_pos = random.uniform(-square_size_y / 2, square_size_y / 2)
        elif edge == 2:  # Top edge
            y_pos = square_size_y / 2
            x_pos = random.uniform(-square_size_x / 2, square_size_x / 2)
            if -0.5 < x_pos < 0.5:
                x_pos = 0.5
        else:  # Bottom edge
            y_pos = -square_size_y / 2
            x_pos = random.uniform(-square_size_x / 2, square_size_x / 2)
            if -0.5 < x_pos < 0.5:
                x_pos = 0.5

        # Z position is fixed at 0.15
        z_pos = 0.15

        # Store the position in a list
        pos = torch.tensor([x_pos, y_pos, z_pos], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        # Generate a random rotation angle around the Z-axis
        theta = random.uniform(-math.pi, math.pi)  # Angle between -π and π
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

        # self.dof_vel[env_ids] = self.dof_init_state[4:8]
        # self.dof_vel_el[env_ids] = self.dof_init_state_el[8:]
        self.dof_vel[env_ids] = self.dof_init_state_el[8:]


        # self.dof_efforts[env_ids] = self.dof_init_state[0:4]
        # self.dof_efforts_el[env_ids] = self.dof_init_state_el[0:8]
        self.dof_efforts[env_ids] = self.dof_init_state_el[0:8]
    
        pos[env_ids, :2] += self.env_origins[env_ids, :2].clone()  # Add only x and y entries from env_origins
        self._robots.set_world_poses(pos[env_ids].clone(), orientations=quat[env_ids].clone(), indices=indices)
        self._robots.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)

        # self._robots_v10.set_joint_efforts(self.dof_efforts[:6].clone(), indices=indices)
        # self._robots_elevated.set_joint_efforts(self.dof_efforts_el[6:].clone(), indices=indices)

        # self._robots_v10.set_joint_velocities(velocities=self.dof_vel[:6].clone(), indices=indices)   
        # self._robots_elevated.set_joint_velocities(velocities=self.dof_vel_el[6:].clone(), indices=indices)

        self._robots.set_joint_efforts(self.dof_efforts[:6].clone(), indices=indices)
        self._robots.set_joint_velocities(velocities=self.dof_vel[:6].clone(), indices=indices)   


        self._targets.set_world_poses(positions=self.target_pos[env_ids].clone(), indices=indices)

        self.last_actions[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.episode_buf[env_ids] = 0 
        # self.reset_buf[env_ids] = 0
        

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
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y ** 2 + z ** 2))

        # Compute target direction vector (only x and y components)
        target_direction = self.target_pos - self.base_pos  # Shape: [batch_size, 3]
        target_direction_x = target_direction[:, 0]
        target_direction_y = target_direction[:, 1]

        # Compute target angle from target direction
        target_angle = torch.atan2(target_direction_y, target_direction_x)

        # Compute angle difference and wrap to [0, π]
        angle_difference = torch.abs(target_angle - yaw)
        self.angle_difference = torch.fmod(angle_difference, 2 * np.pi)

        # Compute linear and angular velocities in the robot's ergo frame
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_vel[:, 0:3])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_vel[:, 3:6])

        return self.base_lin_vel, self.base_ang_vel


    def calculate_acceleration(self, dt):
        # Get current velocities
        current_linear_velocity, current_angular_velocity = self.refresh_body_state_tensors()

        # Calculate accelerations if previous velocities are available
        if self.previous_linear_velocity is not None:
            linear_acceleration = (current_linear_velocity - self.previous_linear_velocity) / dt
            angular_acceleration = (current_angular_velocity - self.previous_angular_velocity) / dt
        else:
            # Set accelerations to zero if it's the first frame
            linear_acceleration = np.zeros(3)
            angular_acceleration = np.zeros(3)

        # Update previous velocities
        self.previous_linear_velocity = current_linear_velocity
        self.previous_angular_velocity = current_angular_velocity

        return linear_acceleration, angular_acceleration

        

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return
        
        # # If we are still in the first two steps, don't apply any action but advance the simulation
        if self.common_step_counter < 2:
            self.common_step_counter += 1
            SimulationContext.step(self.world, render=False)  # Advance simulation
            return 

        self.actions = actions.clone().to(self.device)

        # Apply the actions to the robot
        scaled_actions = self.min_torque + (actions + 1) * 0.5 * (self.max_torque - self.min_torque)

        updated_efforts = torch.clip(scaled_actions, -7.5, 7.5) # 10 Nm ~ 100 N per wheel/ 10 kg per wheel

        joint_indices = torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=self.device)

        if self.world.is_playing():
            # self._robots_v10.set_joint_efforts(updated_efforts[:6]) 
            # self._robots_elevated.set_joint_efforts(updated_efforts[6:], joint_indices=joint_indices)
            self._robots.set_joint_efforts(updated_efforts[:6], joint_indices=joint_indices)
            SimulationContext.step(self.world, render=False)

        # print(self._robots.get_applied_joint_efforts(clone=True)) # [:, np.array([1,2,4,5])]
        dof_names = self._robots.dof_names
        print("DOF Names:", dof_names)
        print("Named dof indices:", [self._robots.get_dof_index(dof) for dof in dof_names])
        
                
        
    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.episode_buf[:] += 1
        
        ids = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        # for i in ids:
        #     print(f"ENV{i} timesteps/MaxEpisodeLength {self.episode_buf[i]}/{self._max_episode_length}")


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

        self._computed_distance = torch.norm(self.base_pos - self.target_pos, dim=-1)

        # target reached or lost
        self.target_reached = self._computed_distance <= 0.6
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
        if self.counter == 0:
            self.position_buffer = self.base_pos[:,:2].clone()
            self.counter += 1
        elif self.counter == 20:
            changed_pos = torch.norm((self.position_buffer - self.base_pos[:,:2].clone()), dim=1)
            self.standing_still = changed_pos < 0.05 
            self.counter = 0  # Reset counter
        else:
            self.counter += 1

        # Update reset_buf based on standing_still condition
        self.reset_buf = torch.where(self.standing_still, torch.ones_like(self.reset_buf), self.reset_buf)

    
    def calculate_metrics(self) -> None:
                
        # Define parameters
        gamma = 0.1  # Decay rate for the exponential reward
        dense_reward = 1.0 - torch.exp(gamma * self._computed_distance)  # Exponential decay otherwise
        dense_reward = torch.where(self.target_reached, torch.zeros_like(dense_reward), dense_reward)  # Set dense_reward to zero where target is reached

        # Alignment reward
        angle_difference = torch.where(self.angle_difference > np.pi, 2 * np.pi - self.angle_difference, self.angle_difference)

        # Compute the alignment reward
        k = 1.25  # Curvature parameter for the exponential function
        alignment_reward = (1.0 - torch.exp(k * (angle_difference / np.pi)))
        alignment_reward = alignment_reward.clamp(min=-15.0, max=0.0)

        # Efficiency penalty: Penalize large torques
        current_efforts = self._robots.get_applied_joint_efforts(clone=True)
        torque_penalty = torch.mean(torch.abs(current_efforts), dim=-1) # max 20 Nm per wheel

        # Bonus for reaching the target
        target_reached = self.target_reached.float() * 1000.0
        crashed = self.fallen.float() * 1000.0   # Penalty for crashing

        # Combine rewards and penalties
        reward = (
            0.75 * dense_reward    # Scale progress
            + 0.75 * alignment_reward    # Scale alignment
            - 0.25 * torque_penalty      # Small penalty for torque
            + target_reached      # Completion bonus
            - crashed
        )
      
        # Normalize and handle resets
        # reward = torch.clip(reward, -50.0, 25.0)  # Clip rewards to avoid large gradients
        self.rew_buf[:] = reward

        return self.rew_buf


    def get_observations(self):
        ids = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.measured_heights = self.get_heights(ids)
        heights = self.measured_heights 

        self.refresh_body_state_tensors()
        delta_pos = self.target_pos - self.base_pos

        # compute distance for calculate_metrics() and is_done()
        self._computed_distance = torch.norm(delta_pos, dim=-1)

        self.obs_buf = torch.cat(
            (
                self.base_vel[:, 0:3],
                self.angle_difference.unsqueeze(-1),
                self.projected_gravity,
                delta_pos,
                # heights,
            ),
            dim=-1,
        )
        
        # print(self.obs_buf)

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
    


