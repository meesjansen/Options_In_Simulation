import torch
import numpy as np
import gym
from gym import spaces
import random
import math
from pxr import Gf, UsdGeom, UsdLux

from my_envs.rl_task import RLTask 

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse, quat_apply
from omni.isaac.core.utils.prims import get_prim_at_path, create_prim
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.simulation_context import SimulationContext

from my_robots.origin_v10_meshes import AvularOrigin_v10 as Robot_v10
from my_utils.terrain_generator_v2 import Terrain, add_terrain_to_stage
from my_utils.terrain_utils_v2 import get_axis_params

TASK_CFG = {"test": False,
            "device_id": 0,
            "headless": True,
            "sim_device": "gpu",
            "enable_livestream": True,
            "warp": False,
            "seed": 42,
            "task": {"name": "ReachingFood",
                     "physics_engine": "physx",
                     "env": {"numEnvs": 64,
                             "envSpacing": 10.0,
                             "episodeLength": 500,
                             "enableDebugVis": False,
                             "clipObservations": 1000.0,
                             "controlFrequencyInv": 4,
                             "baseInitState": {"pos": [0.0, 0.0, 0.0],
                                              "rot": [1.0, 0.0, 0.0, 0.0],
                                              "vLinear": [0.0, 0.0, 0.0],
                                              "vAngular": [0.0, 0.0, 0.0],
                                              },
                            "dofInitTorques": [0.0, 0.0, 0.0, 0.0],
                            "dofInitVelocities": [0.0, 0.0, 0.0, 0.0],
                            "TerrainType": "custom"
                            },
                     "sim": {"dt": 0.0083,
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

        # observation and action space
        self.num_height_points = 13*13
        # Non-height features: base_vel(3), angle_diff(1), projected_gravity(3), target_pos(3), base_pos(3) = 13
        # Height features: 169 points * 1 channels = 169
        # total obs = 10 + 169 = 179
        self._num_observations = 179
        self._num_actions = 4

        self.observation_space = spaces.Box(low=-50, high=50, shape=(self._num_observations,), dtype=np.float32)
        self.min_torque = -10.0
        self.max_torque = 10.0
        self.action_space = spaces.Box(low=self.min_torque, high=self.max_torque, shape=(self._num_actions,), dtype=np.float32)

        self.common_step_counter = 0

        self.update_config(sim_config)
        RLTask.__init__(self, name, env)

        self.height_points = self.init_height_points()
        self.measured_heights = None
        self.bounds = torch.tensor([-3.0, 3.0, -3.0, 3.0], device=self.device, dtype=torch.float)

        self.still_steps = torch.zeros(self.num_envs)
        self.position_buffer = torch.zeros(self.num_envs, 2)
        self.counter = 0
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

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_envs = torch.tensor(self._num_envs, dtype=torch.int64)
        self.terrain_type = self._task_cfg["env"]["TerrainType"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.dt = self._task_cfg["sim"]["dt"]
        self.base_init_state = self._task_cfg["env"]["baseInitState"]["pos"] + self._task_cfg["env"]["baseInitState"]["rot"] + \
                               self._task_cfg["env"]["baseInitState"]["vLinear"] + self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.dof_init_state = self._task_cfg["env"]["dofInitTorques"] + self._task_cfg["env"]["dofInitVelocities"]
        self.decimation = 4

    def init_height_points(self):
        y = 0.5 * torch.tensor([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], device=self.device, requires_grad=False)
        x = 0.5 * torch.tensor([-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 6, device=self.device, requires_grad=False)
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
        # self._create_distant_light()
        self.get_terrain()
        self.get_target()
        self.get_robot()

        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"], copy_from_source=True)

        # light_prim = self._stage.GetPrimAtPath("/World/defaultDistantLight").IsValid()
        # print(light_prim)
        # light_prim = self._stage.GetPrimAtPath("/World/defaultDistantLight")
        # print(light_prim.GetTypeName())
        # light_prim.CreateColorAttr().Set(Gf.Vec3f(1.0, 1.0, 1.0))

        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot_*", name="robot_view")
        scene.add(self._robots)
                     
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)

    # def _create_distant_light(self, prim_path="/World/defaultDistantLight", intensity=5000):
        # light = UsdLux.DistantLight.Define(self._stage, prim_path)
        # light.CreateIntensityAttr().Set(intensity)
        # # Set the color of the light to warm yellow-orange
        # warm_yellow_orange = Gf.Vec3f(255/255, 174/255, 66/255)  # Normalize RGB values to [0,1] range
        # light.CreateColorAttr(warm_yellow_orange)

        # light_1 = create_prim(
        #             "/World/Light_2",
        #             "SphereLight",
        #             position=np.array([1.0, 1.0, 1.0]),
        #             attributes={
        #                 "inputs:radius": 0.01,
        #                 "inputs:intensity": 5e3,
        #                 "inputs:color": (1.0, 0.0, 1.0)
        #             }
        #         )

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
                               radius=0.05,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def post_reset(self):
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_init_state = torch.tensor(self.dof_init_state, dtype=torch.float, device=self.device, requires_grad=False)

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.up_axis_idx = 2
        self.extras = {}

        self.gravity_vec = torch.tensor(
            get_axis_params(-1.0, self.up_axis_idx), dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        
        self.wheel_torques = torch.zeros(
            self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.actions = torch.zeros(
            self.num_envs, self._num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.last_actions = torch.zeros(
            self.num_envs, self._num_actions, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.num_dof = self._robots.num_dof 
        self.env_origins = self.terrain_origins.view(-1, 3)[:self.num_envs]
        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_pos += torch.tensor([0.0, 0.0, 0.26], dtype=torch.float, device=self.device)
        self.target_pos[:, :2] += self.env_origins[:, :2]
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_efforts = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
      
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(indices)
        base_pos, base_quat = self._robots.get_world_poses(clone=False)
        self.last_distance_to_target = torch.norm(base_pos - self.target_pos, dim=-1)

        self.init_done = True

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        square_size_x = 4.5
        square_size_y = 4.5
        edge = random.randint(0, 1)

        if edge == 0:
            x_pos = -square_size_x / 2
            y_pos = random.uniform(-square_size_y / 2, square_size_y / 2)
        else:
            x_pos = square_size_x / 2
            y_pos = random.uniform(-square_size_y / 2, square_size_y / 2)

        z_pos = 0.15
        pos = torch.tensor([x_pos, y_pos, z_pos], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        theta = random.uniform(-math.pi, math.pi)
        half_theta = theta / 2.0
        cos_half_theta = math.cos(half_theta)
        sin_half_theta = math.sin(half_theta)
        w = cos_half_theta
        x = 0.0
        y = 0.0
        z = sin_half_theta
        quat = torch.tensor([w, x, y, z], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)

        self.dof_vel[env_ids] = self.dof_init_state[4:8]
        self.dof_efforts[env_ids] = self.dof_init_state[0:4]
    
        pos[env_ids, :2] += self.env_origins[env_ids, :2].clone()
        self._robots.set_world_poses(pos[env_ids].clone(), orientations=quat[env_ids].clone(), indices=indices)
        self._robots.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._robots.set_joint_efforts(self.dof_efforts[env_ids].clone(), indices=indices)
        self._robots.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)   

        self._targets.set_world_poses(positions=self.target_pos[env_ids].clone(), indices=indices)

        self.last_actions[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.episode_buf[env_ids] = 0

    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._robots.get_world_poses(clone=False)
        self.base_vel = self._robots.get_velocities(clone=False)
        self.target_pos, _ = self._targets.get_world_poses(clone=False)

        w = self.base_quat[:, 0]
        x = self.base_quat[:, 1]
        y = self.base_quat[:, 2]
        z = self.base_quat[:, 3]

        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y ** 2 + z ** 2))

        target_direction = self.target_pos - self.base_pos
        target_direction_x = target_direction[:, 0]
        target_direction_y = target_direction[:, 1]
        target_angle = torch.atan2(target_direction_y, target_direction_x)

        angle_difference = torch.abs(target_angle - yaw)
        self.angle_difference = torch.fmod(angle_difference, 2 * np.pi)

        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_vel[:, 0:3])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_vel[:, 3:6])

        return self.base_lin_vel, self.base_ang_vel

    def calculate_acceleration(self, dt):
        current_linear_velocity, current_angular_velocity = self.refresh_body_state_tensors()

        if self.previous_linear_velocity is not None:
            linear_acceleration = (current_linear_velocity - self.previous_linear_velocity) / dt
            angular_acceleration = (current_angular_velocity - self.previous_angular_velocity) / dt
        else:
            linear_acceleration = np.zeros(3)
            angular_acceleration = np.zeros(3)

        self.previous_linear_velocity = current_linear_velocity
        self.previous_angular_velocity = current_angular_velocity

        return linear_acceleration, angular_acceleration

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return
        
        if self.common_step_counter < 2:
            self.common_step_counter += 1
            SimulationContext.step(self.world, render=False)
            return

        self.actions = actions.clone().to(self.device)
        scaled_actions = self.min_torque + (actions + 1) * 0.5 * (self.max_torque - self.min_torque)
        updated_efforts = torch.clip(scaled_actions, -10.0, 10.0)

        if self.world.is_playing():
            self._robots.set_joint_efforts(updated_efforts) 
            SimulationContext.step(self.world, render=False)

        self.linear_acceleration, self.angular_acceleration = self.calculate_acceleration(self.dt)
        for i in range(self.decimation):
            if self.world.is_playing():
                self._robots.set_joint_efforts(updated_efforts) 
                SimulationContext.step(self.world, render=False)

    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.episode_buf[:] += 1

        if self.world.is_playing():
            self.refresh_body_state_tensors()
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

        self.target_reached = self._computed_distance <= 0.3
        self.reset_buf = torch.where(self.target_reached, torch.ones_like(self.reset_buf), self.reset_buf)

        self.timeout_buf = torch.where(
            self.episode_buf >= self._max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        ) 
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)  

        projected_gravity = quat_apply(self.base_quat, self.gravity_vec)
        positive_gravity_z_threshold = 0.0
        self.fallen = projected_gravity[:, 2] > positive_gravity_z_threshold
        self.reset_buf = torch.where(self.fallen, torch.ones_like(self.reset_buf), self.reset_buf)

        self.out_of_bounds = ((self.base_pos[:, 0] - self.env_origins[:, 0]) < self.bounds[0]) | ((self.base_pos[:, 0] - self.env_origins[:, 0]) > self.bounds[1]) | \
                             ((self.base_pos[:, 1] - self.env_origins[:, 1]) < self.bounds[2]) | ((self.base_pos[:, 1] - self.env_origins[:, 1]) > self.bounds[3])
        self.reset_buf = torch.where(self.out_of_bounds, torch.ones_like(self.reset_buf), self.reset_buf)

        self.standing_still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.counter == 0:
            self.position_buffer = self.base_pos[:,:2].clone()
            self.counter += 1
        elif self.counter == 20:
            changed_pos = torch.norm((self.position_buffer - self.base_pos[:,:2].clone()), dim=1)
            self.standing_still = changed_pos < 0.05
            self.counter = 0
        else:
            self.counter += 1
        self.reset_buf = torch.where(self.standing_still, torch.ones_like(self.reset_buf), self.reset_buf)

    def calculate_metrics(self) -> None:
        gamma = 0.1
        dense_reward = 1.0 - torch.exp(gamma * self._computed_distance)
        dense_reward = torch.where(self.target_reached, torch.zeros_like(dense_reward), dense_reward)

        angle_difference = torch.where(self.angle_difference > np.pi, 2 * np.pi - self.angle_difference, self.angle_difference)
        k = 1.25
        alignment_reward = (1.0 - torch.exp(k * (angle_difference / np.pi)))
        alignment_reward = alignment_reward.clamp(min=-15.0, max=0.0)

        current_efforts = self._robots.get_applied_joint_efforts(clone=True)
        torque_penalty = torch.mean(torch.abs(current_efforts), dim=-1)

        target_reached = self.target_reached.float() * 1000.0
        crashed = self.fallen.float() * 1000.0

        reward = (
            dense_reward
            # + alignment_reward
            - 0.5 * torque_penalty
            + target_reached
            - crashed
        )

        self.rew_buf[:] = reward
        return self.rew_buf

    def get_observations(self):
        ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        heights = self.get_heights(ids)
        Nx, Ny, Nz = self.get_normals(ids)

        full_points = self.height_points.clone()
        full_points[:, :, 2] = heights
        full_points[:, :, 3] = Nx
        full_points[:, :, 4] = Ny
        full_points[:, :, 5] = Nz

        height_data = full_points[:, :, 2].reshape(self.num_envs, -1)

        print(height_data)
        non_zero_count = torch.count_nonzero(height_data)
        print(f"Number of non-zero elements in height_data: {non_zero_count}")

        self.refresh_body_state_tensors()
        delta_pos = self.target_pos - self.base_pos
        self._computed_distance = torch.norm(delta_pos, dim=-1)

        # Construct final observation
        self.obs_buf = torch.cat(
            (
                self.base_vel[:, 0:3],
                self.angle_difference.unsqueeze(-1),
                self.projected_gravity,
                delta_pos,
                height_data
            ),
            dim=-1,
        )

        return {self._robots.name: {"obs_buf": self.obs_buf}}

    def get_heights(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        points_2d = self.height_points[env_ids, :, 0:2] + self.env_origins[env_ids, 0:2].unsqueeze(1)
        points_2d[..., :2] += self.terrain.border_size
        points_idx = (points_2d[..., :2] / self.terrain.horizontal_scale).long()

        px = points_idx[..., 0].clamp(0, self.height_samples.shape[0]-2)
        py = points_idx[..., 1].clamp(0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py+1]
        heights = torch.min(heights1, heights2) * self.terrain.vertical_scale
        return heights

    def get_normals(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        points_2d = self.height_points[env_ids, :, 0:2] + self.env_origins[env_ids, 0:2].unsqueeze(1)
        points_2d[..., :2] += self.terrain.border_size
        points_idx = (points_2d[..., :2] / self.terrain.horizontal_scale).long()

        px = points_idx[..., 0].clamp(0, self.terrain.normal_map_x.shape[0]-2)
        py = points_idx[..., 1].clamp(0, self.terrain.normal_map_x.shape[1]-2)

        Nx1 = torch.from_numpy(self.terrain.normal_map_x[px.cpu(), py.cpu()]).to(self.device)
        Nx2 = torch.from_numpy(self.terrain.normal_map_x[(px+1).cpu(), (py+1).cpu()]).to(self.device)
        Nx = (Nx1 + Nx2) / 2.0

        Ny1 = torch.from_numpy(self.terrain.normal_map_y[px.cpu(), py.cpu()]).to(self.device)
        Ny2 = torch.from_numpy(self.terrain.normal_map_y[(px+1).cpu(), (py+1).cpu()]).to(self.device)
        Ny = (Ny1 + Ny2) / 2.0

        Nz1 = torch.from_numpy(self.terrain.normal_map_z[px.cpu(), py.cpu()]).to(self.device)
        Nz2 = torch.from_numpy(self.terrain.normal_map_z[(px+1).cpu(), (py+1).cpu()]).to(self.device)
        Nz = (Nz1 + Nz2) / 2.0

        return Nx, Ny, Nz