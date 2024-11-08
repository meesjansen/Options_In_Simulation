import torch
import numpy as np
import gym
from gym import spaces

from omniisaacgymenvs.tasks.base.rl_task import RLTask

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
                             "episodeLength": 1000,
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


class ReachingTargetTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:

        self.height_samples = None
        self.init_done = False
        
        self.dt = 1 / 120.0

        # observation and action space DQN
        self._num_observations = 16 + 289  # features + height points
        self._num_actions = 11  # Designed discrete action space see pre_physics_step()

        self.observation_space = spaces.Box(
            low=float("-inf"),  # Replace with a specific lower bound if needed
            high=float("inf"),  # Replace with a specific upper bound if needed
            shape=(self.num_observations,),
            dtype=np.float32  # Ensure data type is consistent
        )
        self.action_space = spaces.Discrete(self._num_actions)

        self.common_step_counter = 0 # Counter for the first two steps

        self.update_config(sim_config)

        RLTask.__init__(self, name, env)

        self.height_points = self.init_height_points()  
        self.measured_heights = None
        self.bounds = torch.tensor([-4.0, 4.0, -4.0, 4.0], device=self.device, dtype=torch.float)



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

        self.decimation = 4

    def init_height_points(self):
        # 8mx8m rectangle (without center line) 17x17=289 points
        y = 0.5 * torch.tensor(
            [ -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False
        )  # 25cm on each side
        x = 0.5 * torch.tensor(
            [ -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8], device=self.device, requires_grad=False
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

        # self.rubber_material = PhysicsMaterial(
        #     prim_path="/World/PhysicsMaterials/RubberMaterial",
        #     static_friction=0.9,
        #     dynamic_friction=0.8,
        #     restitution=0.2
        # )

        # visual_material = OmniPBR(prim_path="/World/material/glass", color=np.array([0.8, 0.1, 0.1]))

        # Define the relative wheel paths for each robot instance
        wheel_prim_paths = [
            "left_front_wheel",
            "left_rear_wheel",
            "right_front_wheel",
            "right_rear_wheel",
        ] 


        # robot view
        self._robots = RobotView(prim_paths_expr="/World/envs/.*/robot_*", name="robot_view")
        scene.add(self._robots)

        # material_path = "/World/PhysicsMaterials/WheelMaterial"
        # material_prim = self._stage.DefinePrim(material_path, "Material")

        # # Apply the PhysxMaterialAPI to the material prim
        # PhysxSchema.PhysxMaterialAPI.Apply(material_prim)


        # # Set material properties if creating a new material
        # material_prim.GetAttribute("physics:staticFriction").Set(0.9)
        # material_prim.GetAttribute("physics:dynamicFriction").Set(0.8)
        # material_prim.GetAttribute("physics:restitution").Set(0.5)

        # # Apply the material to each robot's wheels
        # for robot_prim_path in self._robots.prim_paths:  # Get each robot's prim path
        #     robot_prim_path = robot_prim_path.replace("/main_body", "")
        #     for wheel_relative_path in wheel_prim_paths:
        #         wheel_full_path = f"{robot_prim_path}/{wheel_relative_path}"  # Construct full wheel path
        #         print("Paths to wheels:", wheel_full_path)
        #         wheel_prim = self._stage.GetPrimAtPath(wheel_full_path)
        #         # Apply PhysX Material API to the prim
        #         PhysxSchema.PhysxMaterialAPI.Apply(wheel_prim) 
        #         # Set the material relationship on the wheel prim
        #         wheel_prim.GetRelationship("physics:physicsMaterial").AddTarget(material_prim.GetPath())



                
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
        self.target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.target_pos += torch.tensor([0.0, 0.0, 0.1], dtype=torch.float, device=self.device)
        self.target_pos[:, :2] += self.env_origins[:, :2]
        self.base_velocities = torch.zeros((self.num_envs, 6), dtype=torch.float, device=self.device)
        self.dof_vel = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
        self.dof_efforts = torch.zeros((self.num_envs, self.num_dof), dtype=torch.float, device=self.device)
      
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.reset_idx(indices)
        self.init_done = True

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)

        # Define square boundary size with some margin to reduce instant resets
        square_size_x = 6.5  # Total width of the square
        square_size_y = 6.5  # Total length of the square

        edge = random.randint(0, 3)

        # Generate x and y positions based on the edge
        if edge == 0:  # Left edge
            x_pos = -square_size_x / 2
            y_pos = random.uniform(-square_size_y / 2, square_size_y / 2)
            quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)  # Looking right
        elif edge == 1:  # Right edge
            x_pos = square_size_x / 2
            y_pos = random.uniform(-square_size_y / 2, square_size_y / 2)
            quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)  # Looking left
        elif edge == 2:  # Top edge
            y_pos = square_size_y / 2
            x_pos = random.uniform(-square_size_x / 2, square_size_x / 2)
            quat = torch.tensor([0.7071, 0.0, 0.0, -0.7071], device=self.device)  # Looking down
        else:  # Bottom edge
            y_pos = -square_size_y / 2
            x_pos = random.uniform(-square_size_x / 2, square_size_x / 2)
            quat = torch.tensor([0.7071, 0.0, 0.0, 0.7071], device=self.device)  # Looking up

        # Z position is fixed at 0.4
        z_pos = 0.15

        # Store the position in a list
        pos = torch.tensor([x_pos, y_pos, z_pos], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        quat = quat.repeat(self.num_envs, 1)

        self.dof_vel[env_ids] = self.dof_init_state[4:8]
        self.dof_efforts[env_ids] = self.dof_init_state[0:4]
    
        pos[env_ids, :2] += self.env_origins[env_ids, :2].clone()  # Add only x and y entries from env_origins
        self._robots.set_world_poses(pos[env_ids].clone(), orientations=quat[env_ids].clone(), indices=indices)
        self._robots.set_velocities(velocities=self.base_velocities[env_ids].clone(), indices=indices)
        self._robots.set_joint_efforts(self.dof_efforts[env_ids].clone(), indices=indices)
        self._robots.set_joint_velocities(velocities=self.dof_vel[env_ids].clone(), indices=indices)   

        self._targets.set_world_poses(positions=self.target_pos[env_ids].clone(), indices=indices)

        self.last_actions[env_ids] = 0.0
        self.progress_buf[env_ids] = 0
        self.episode_buf[env_ids] = 0 
        self.reset_buf[env_ids] = 0
        self.rew_buf[env_ids] = 0.0


    def refresh_body_state_tensors(self):
        self.base_pos, self.base_quat = self._robots.get_world_poses(clone=False)
        self.base_vel = self._robots.get_velocities(clone=False)

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
        
        # If we are still in the first two steps, don't apply any action but advance the simulation
        if self.common_step_counter < 2:
            # print(f"Skipping actions for first {self.common_step_counter + 1} step(s)")
            self.common_step_counter += 1
            SimulationContext.step(self.world, render=False)  # Advance simulation
            return 

        # There are 12 possible actions
        action_torque_vectors = torch.tensor([
            [0.5, 0.5, 0.5, 0.5],
            [-0.5, -0.5, -0.5, -0.5],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [-0.5, 0.0, -0.5, 0.0],
            [0.0, -0.5, 0.0, -0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.5],
            [-0.5, -0.5, 0.0, 0.0],
            [0.0, 0.0, -0.5, -0.5],
            [0.0, 0.0, 0.0, 0.0],
        ], device=self.device)

        current_efforts = self._robots.get_applied_joint_efforts(clone=True) # [:, np.array([1,2,4,5])]
        updated_efforts = torch.zeros_like(current_efforts)

        self.actions = actions.clone().to(self.device)
        print("Actions from pre_physics_step:", self.actions)

        for env_id in range(self.num_envs):
            action_index = int(torch.argmax(self.actions[env_id]).item())  # Get action index for the current environment
            print("Action index:", action_index)
            action_index = int(self.actions[env_id].item())
            print("Action index:", action_index)
            delta_torque = action_torque_vectors[action_index]  # Get the torque change vector for this action
            updated_efforts[env_id] = current_efforts[env_id] + delta_torque  # Update the torque for this environment

        updated_efforts = torch.clip(updated_efforts, -10.0, 10.0) # 10 Nm ~ 100 N per wheel/ 10 kg per wheel
        print("max velocities dof: ", self._robots.get_joint_max_velocities())

        if self.world.is_playing():
            self._robots.set_joint_efforts(updated_efforts) 
            print("Applied torques:", updated_efforts)

            SimulationContext.step(self.world, render=False)

        self.linear_acceleration, self.angular_acceleration = self.calculate_acceleration(self.dt)
        print("Linear acceleration:", self.linear_acceleration)
        print("Angular acceleration:", self.angular_acceleration)
          
        for i in range(self.decimation):
            if self.world.is_playing():
                
                self._robots.set_joint_efforts(updated_efforts) 
                print("Applied torques:", updated_efforts)

                SimulationContext.step(self.world, render=False)

        # self._dof_indices = torch.tensor([robot.get_dof_index(dof) for dof in robot.dof_names], dtype=torch.int32, device=self.device)
        # print("Named dof indices:", [self._robots.get_dof_index(dof) for dof in [
        #         "main_body_left_front_wheel", 
        #         "main_body_left_rear_wheel",
        #         "main_body_right_front_wheel",
        #         "main_body_right_rear_wheel"
        # ]])

                
        
    def post_physics_step(self):
        self.progress_buf[:] += 1
        self.episode_buf[:] += 1
        ids = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)

        for i in ids:
            print(f"ENV0 timesteps/MaxEpisodeLength {self.episode_buf[i]}/{self._max_episode_length}")


        if self.world.is_playing():

            self.refresh_body_state_tensors()

            # prepare quantities            
            self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                        
            # if self.add_noise:
            #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec
            self.is_done()
            # self.get_states()
            self.calculate_metrics()
            
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(env_ids) > 0:
                self.reset_idx(env_ids)

            self.get_observations()
            
            self.last_actions[:] = self.actions[:]


        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras
    

    def is_done(self):
        self.timeout_buf = torch.where(
            self.episode_buf >= self._max_episode_length - 1,
            torch.ones_like(self.timeout_buf),
            torch.zeros_like(self.timeout_buf),
        )   
        self.reset_buf.fill_(0)

        base_pos, base_quat = self._robots.get_world_poses(clone=False)
        target_pos, _ = self._targets.get_world_poses(clone=False)
        self._computed_distance = torch.norm(base_pos - target_pos, dim=-1)

        # target reached or lost
        self.target_reached = self._computed_distance <= 0.1
        self.reset_buf = torch.where(self.target_reached, torch.ones_like(self.reset_buf), self.reset_buf)
        print("self.compute_distance", self._computed_distance)
        print("Target reached", self.target_reached)
        print("Reset buffer post distance", self.reset_buf)

        # max episode length
        self.reset_buf = torch.where(self.timeout_buf.bool(), torch.ones_like(self.reset_buf), self.reset_buf)  
        # print("Reset buffer post episode length", self.reset_buf)


        # Calculate the projected gravity in the robot's local frame
        projected_gravity = quat_apply(base_quat, self.gravity_vec)

        # Detect if the robot is on its back based on positive Z-axis component of the projected gravity
        positive_gravity_z_threshold = 0.0  # Adjust the threshold if needed
        self.fallen = projected_gravity[:, 2] > positive_gravity_z_threshold
        self.reset_buf = torch.where(self.fallen, torch.ones_like(self.reset_buf), self.reset_buf)
        # print("Reset buffer post gravity", self.reset_buf)

        self.out_of_bounds = ((self.base_pos[:, 0] - self.env_origins[:, 0]) < self.bounds[0]) | ((self.base_pos[:, 0] - self.env_origins[:, 0]) > self.bounds[1]) | \
                        ((self.base_pos[:, 1] - self.env_origins[:, 1]) < self.bounds[2]) | ((self.base_pos[:, 1] - self.env_origins[:, 1]) > self.bounds[3])
        self.reset_buf = torch.where(self.out_of_bounds, torch.ones_like(self.reset_buf), self.reset_buf)
        # print("Reset buffer post out of bounds", self.reset_buf)

        # Check standing still condition every still_check_interval timesteps
        self.standing_still = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if self.counter == 0:
            self.position_buffer = self.base_pos[:,:2].clone()
            self.counter += 1
        elif self.counter == 10:
            changed_pos = torch.norm((self.position_buffer - self.base_pos[:,:2].clone()), dim=1)
            # print("Changed pos pre standing still", self.reset_buf)
            self.standing_still = changed_pos < 0.05 
            self.counter = 0  # Reset counter
        else:
            self.counter += 1

        # Update reset_buf based on standing_still condition
        self.reset_buf = torch.where(self.standing_still, torch.ones_like(self.reset_buf), self.reset_buf)
        # print("Reset buffer post standing still", self.reset_buf)



    
    def calculate_metrics(self) -> None:
        # computed distance to target as updating reward
        self.rew_buf[:] = 0.1/self._computed_distance * 100.0

        self.rew_buf[self.target_reached] += 100 #target reached

        if self.target_reached.any():
            print("Success")

        # Check fallen condition
        self.rew_buf[self.fallen] += -20.0 # fallen

        # Check out-of-bounds condition
        self.rew_buf[self.out_of_bounds] += -10.0 # out of bounds

        # Check standing still condition
        self.rew_buf[self.standing_still] += -10.0 # standing still

        # Define the allowed range for angular acceleration
        allowed_angular_acceleration = 2.0  # rad/s^2
        # Calculate the excess angular acceleration
        excess_angular_acceleration = torch.clamp(self.angular_acceleration - allowed_angular_acceleration, min=0.0)
        self.rew_buf[:] += -excess_angular_acceleration[:, 0] # Example scaling factor
        self.rew_buf[:] += -excess_angular_acceleration[:, 1] # Example scaling factor
        self.rew_buf[:] += -excess_angular_acceleration[:, 2] # Example scaling factor



        # Define the allowed range for linear velocity and create a similar reward term for self.base_lin_vel not exceeding a certain threshold or result in negative reward
        allowed_linear_velocity = 2.0  # m/s
        excess_linear_velocity = torch.clamp(self.base_lin_vel[:, 0] - allowed_linear_velocity, min=0.0)
        self.rew_buf[:] += -excess_linear_velocity * 10.0  

        print("Reward buffer:", self.rew_buf, "Reward shape" , self.rew_buf.shape)

        return self.rew_buf


    def get_observations(self):
        ids = torch.arange(self._num_envs, dtype=torch.int64, device=self.device)
        self.measured_heights = self.get_heights(ids)
        heights = self.measured_heights * self.terrain.vertical_scale 

        base_pos, _ = self._robots.get_world_poses(clone=False)
        target_pos, _ = self._targets.get_world_poses(clone=False)
        delta_pos = target_pos - base_pos

        # Get current joint efforts (torques)
        _efforts = self._robots.get_applied_joint_efforts(clone=True)
        current_efforts = _efforts #[:, np.array([1,2,4,5])]

        # compute distance for calculate_metrics() and is_done()
        self._computed_distance = torch.norm(delta_pos, dim=-1)

        # Print the observation buffer
        self.obs_buf = torch.cat(
            (
                self.base_vel[:, 0:3],
                self.base_vel[:, 3:6],
                self.projected_gravity,
                delta_pos,
                heights,
                current_efforts 
            ),
            dim=-1,
        )
        # print("Observation buffer:", self.obs_buf.shape)
        
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

