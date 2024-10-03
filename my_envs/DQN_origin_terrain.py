import torch
import numpy as np
import gym
from gym import spaces

from omniisaacgymenvs.tasks.base.rl_task import RLTask

from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.maths import torch_rand_float
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.simulation_context import SimulationContext

from my_robots.Origin import AvularOrigin as Robot
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
                     "env": {"numEnvs": 1,
                             "envSpacing": 1.5,
                             "episodeLength": 1000,
                             "enableDebugVis": False,
                             "clipObservations": 1000.0,
                             "controlFrequencyInv": 4,
                             "baseInitState": {"pos": [-2.0, -2.0, 0.62], # x,y,z [m]
                                              "rot": [1.0, 0.0, 0.0, 0.0], # w,x,y,z [quat]
                                              "vLinear": [0.0, 0.0, 0.0],  # x,y,z [m/s]
                                              "vAngular": [0.0, 0.0, 0.0],  # x,y,z [rad/s]
                                                },
                            "baseInitTorques": [40.0, 40.0, 40.0, 40.0]                            
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
                                      "solver_velocity_iteration_count": 1,
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
                                       "solver_velocity_iteration_count": 1,
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
                                        "solver_position_iteration_count": 4,
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


class ReachingFoodTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.dt = 1 / 120.0

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._num_envs = torch.tensor(self._num_envs, dtype=torch.int64)
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["episodeLength"]
        
        # observation and action space DQN
        self._num_observations = 16 # 16 + heightpoints
        self._num_actions = 12  # Assuming 3 discrete actions per wheel
        self.common_step_counter = 0

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)


    def update_config(self, sim_config):
        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.torques = self._task_cfg["env"]["baseInitTorques"]
        self.base_init_state = pos + rot + v_lin + v_ang

        self.decimation = 4

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

    def set_up_scene(self, scene) -> None:
        self._stage = get_current_stage()
        self.get_terrain()
        self.get_robot()
        self.get_target()

        super().set_up_scene(scene, collision_filter_global_paths=["/World/terrain"])

        # robot view
        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot", name="robot_view")
        scene.add(self._robots)
        
        # food view
        self._targets = RigidPrimView(prim_paths_expr="/World/envs/.*/target", name="target_view", reset_xform_properties=False)
        scene.add(self._targets)

    def get_terrain(self, create_mesh=True):
        self.env_origins = torch.zeros((self.num_envs, 3), device=self.device, requires_grad=False)
        self._create_trimesh(create_mesh=create_mesh)
        self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)

    def get_robot(self):
        # Assuming LIMO or similar wheeled robot
        robot_translation = torch.tensor([3.0, 3.0, 0.4])
        robot_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0])
        robot = Robot(
            prim_path=self.default_zero_env_path + "/robot",
            name="robot",
            translation=robot_translation,
            orientation=robot_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "robot", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("robot")
        )
        robot.set_robot_properties(self._stage, robot.prim)

    def get_target(self):
        target = DynamicSphere(prim_path=self.default_zero_env_path + "/target",
                               name="target",
                               radius=0.05,
                               color=torch.tensor([1, 0, 0]))
        self._sim_config.apply_articulation_settings("target", get_prim_at_path(target.prim_path), self._sim_config.parse_actor_config("target"))
        target.set_collision_enabled(False)

    def post_reset(self):
        self.base_init_state = torch.tensor(self.base_init_state, dtype=torch.float, device=self.device, requires_grad=False)

        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

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
        self.env_origins = self.terrain_origins.view(-1, 3)[:self.num_envs]
        self.base_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.base_quat = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
      
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(indices)
        self.init_done = True

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = velocities

        self.base_pos[env_ids] = self.base_init_state[0:3]
        self.base_pos[env_ids, 0:3] += self.env_origins[env_ids]
        self.base_pos[env_ids, 0:2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)
        self.base_quat[env_ids] = self.base_init_state[3:7]
        self.base_velocities[env_ids] = self.base_init_state[7:13]
     
        
        self._robots.set_world_poses(positions=self.base_pos[env_ids].clone(), orientations=self.base_quat[env_ids].clone(), indices=indices)
        self._robots.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._robots.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)

        # reset target
        pos = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 \
            * torch.tensor([0.10, 0.20, 0.20], device=self.device) \
            + torch.tensor([0.60, 0.00, 0.40], device=self.device)

        self._targets.set_world_poses(pos + self._env_pos[env_ids], indices=indices)

        self.last_actions[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # fill extras for reward shaping
        # self.extras["episode"] = {}
        # for key in self.episode_sums.keys():
        #     self.extras["episode"]["rew_" + key] = (
        #         torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
        #     )
        #     self.episode_sums[key][env_ids] = 0.0

    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._robots.get_world_poses(clone=False)
        self.base_velocities = self._robots.get_velocities(clone=False)
        

    def pre_physics_step(self, actions):
        if not self.world.is_playing():
            return
        
        # If we are still in the first two steps, don't apply any action but advance the simulation
        if self.common_step_counter < 2:
            print(f"Skipping actions for first {self.common_step_counter + 1} step(s)")
            self.common_step_counter += 1
            SimulationContext.step(self.world, render=False)  # Advance simulation
            return 

        # There are 12 possible actions
        action_torque_vectors = torch.tensor([
            [10.0, 10.0, 10.0, 10.0],
            [-10.0, -10.0, -10.0, -10.0],
            [10.0, 0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0, 10.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 10.0],
            [0.0, 0.0, 10.0, 0.0],
            [0.0, 10.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [10.0, 10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0, 10.0],
            [-10.0, -10.0, 10.0, 10.0]
        ], device=self.device)

        current_efforts = self._robots.get_applied_joint_efforts(clone=True)
        updated_efforts = torch.zeros_like(current_efforts)

        self.actions = actions.clone().to(self.device)
        print("Action Q-learning:", self.actions)

        for env_id in range(self.num_envs):
            action_index = int(self.actions[env_id].item())  # Get action index for the current environment
            delta_torque = action_torque_vectors[action_index]  # Get the torque change vector for this action
            updated_efforts[env_id] = current_efforts[env_id] + delta_torque  # Update the torque for this environment

        updated_efforts = torch.clip(updated_efforts, -100.0, 100.0)
          
        for i in range(self.decimation):
            if self.world.is_playing():
                
                # self._robots.set_joint_efforts(test_efforts1, indices=np.array([0]),joint_indices=np.array([1, 2, 4, 5]))
                self._robots.set_joint_efforts(updated_efforts)
                print("Applied torques:", updated_efforts)

                self.torques = updated_efforts
                SimulationContext.step(self.world, render=False)
                
        
    def post_physics_step(self):
        self.progress_buf[:] += 1

        if self.world.is_playing():

            self.refresh_body_state_tensors()

            # prepare quantities
            self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 0:3])
            self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.base_velocities[:, 3:6])
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                        
            # if self.add_noise:
            #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            self.is_done()
            self.get_states()
            self.calculate_metrics()
            
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
            
            self.last_actions[:] = self.actions[:]


        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    

    def is_done(self):
        self.timeout_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        )
        self.reset_buf.fill_(0)

        base_pos, base_rot = self._robots.get_world_poses(clone=False)
        target_pos, target_rot = self._targets.get_world_poses(clone=False)
        self._computed_distance = torch.norm(base_pos - target_pos, dim=-1)

        # target reached
        self.reset_buf = torch.where(self._computed_distance <= 0.0035, torch.ones_like(self.reset_buf), self.reset_buf)
        # max episode length
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)  

    def calculate_metrics(self) -> None:
        self.rew_buf[:] = -self._computed_distance


    def discretize_state(self, continuous_state, num_bins=10, low=-1.0, high=1.0):
        # Discretize a continuous state into a discrete value
        bins = torch.linspace(low, high, num_bins, device=self.device)
        discrete_state = torch.bucketize(continuous_state, bins) - 1  # Subtract 1 to match 0-indexing
        return discrete_state
    
    
    def calculate_index_from_obs_buf(self, obs_buf, num_bins_per_feature):
        
        # Initialize the index tensor with zeros
        Q_indices = torch.zeros(obs_buf.shape[0], dtype=torch.long, device=self.device)
        
        # Multiply by powers of the bin sizes to create a unique index
        multiplier = 1
        for i, num_bins in enumerate(reversed(num_bins_per_feature)):
            feature_idx = -(i+1)  # work backwards from the last feature
            Q_indices += obs_buf[:, feature_idx] * multiplier
            multiplier *= num_bins  # Increase the base for the next feature

        # Reshape to (env_id, 1) to match the expected shape
        return Q_indices.view(-1, 1)
    

    def get_observations(self):
        heights = self.get_heights()

        base_pos, base_rot = self._robots.get_world_poses(clone=False)
        target_pos, target_rot = self._targets.get_world_poses(clone=False)
        delta_pos = target_pos - self.env_origins

        # Get current joint efforts (torques)
        current_efforts = self._robots.get_applied_joint_efforts(clone=True)

        # compute distance for calculate_metrics() and is_done()
        self._computed_distance = torch.norm(base_pos - target_pos, dim=-1)

        # Print the observation buffer
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel,
                self.base_ang_vel,
                self.projected_gravity,
                delta_pos,
                # discrete_heights,
                current_efforts 
            ),
            dim=-1,
        )
        print("Observation buffer with measured efforts:", self.obs_buf)
        
        return {self._robots.name: {"obs_buf": self.obs_buf}}
    

    def get_heights(self, env_ids=None):
        
        heights = self.height_samples

        
        return heights.view(self.num_envs, -1) * self.terrain.vertical_scale
