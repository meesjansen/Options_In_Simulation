from omni.isaac.kit import SimulationApp

# Initialize the SimulationApp before any imports from omni.isaac.core
simulation_app = SimulationApp({"headless": True, "enable_livestream": True})

from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from omni.isaac.urdf import _urdf
from omni.isaac.franka.controllers import RMPFlowController
from omni.isaac.franka.tasks import FollowTarget

class HelloWorld(BaseSample):
    def __init__(self) -> None:
        super().__init__()
        return

    def setup_scene(self):
        world = self.get_world()
        world.scene.add_default_ground_plane()
        # Acquire the URDF extension interface
        urdf_interface = _urdf.acquire_urdf_interface()
        # Set the settings in the import config
        import_config = _urdf.ImportConfig()
        import_config.merge_fixed_joints = False
        import_config.convex_decomp = False
        import_config.import_inertia_tensor = True
        import_config.fix_base = True
        import_config.make_default_prim = True
        import_config.self_collision = False
        import_config.create_physics_scene = True
        import_config.import_inertia_tensor = False
        import_config.default_drive_strength = 10000000.0
        import_config.default_position_drive_damping = 100000.0
        import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        import_config.distance_scale = 100
        import_config.density = 0.0
        # Get the urdf file path
        extension_path = get_extension_path_from_name("omni.isaac.urdf")
        root_path = extension_path + "/data/urdf/robots/franka_description/robots"
        file_name = "panda_arm_hand.urdf"
        # Finally import the robot
        imported_robot = urdf_interface.parse_urdf(root_path, file_name, import_config)
        prim_path = urdf_interface.import_robot(root_path, file_name, imported_robot, import_config)
        # Now lets use it with one of the tasks defined under omni.isaac.franka
        # Similar to what was covered in Tutorial 6 Adding a Manipulator in the Required Tutorials
        my_task = FollowTarget(name="follow_target_task", franka_prim_path=prim_path, franka_robot_name="fancy_franka", target_name="target")
        world.add_task(my_task)
        return

    async def setup_post_load(self):
        self._world = self.get_world()
        self._franka = self._world.scene.get_object("fancy_franka")
        self._controller = RMPFlowController(name="target_follower_controller", robot_prim_path=self._franka.prim_path)
        self._world.add_physics_callback("sim_step", callback_fn=self.physics_step)
        await self._world.play_async()
        return

    async def setup_post_reset(self):
        self._controller.reset()
        await self._world.play_async()
        return

    def physics_step(self, step_size):
        world = self.get_world()
        observations = world.get_observations()
        actions = self._controller.forward(
            target_end_effector_position=observations["target"]["position"],
            target_end_effector_orientation=observations["target"]["orientation"],
        )
        self._franka.apply_action(actions)
        return