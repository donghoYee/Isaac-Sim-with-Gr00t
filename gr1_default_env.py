from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from isaacsim.core.api import World
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import XFormPrim
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api.objects import DynamicCuboid
import matplotlib.pyplot as plt
import gr1_config



def main():
    ## 1. setup scene
    print("## 1. setup scene")
    world = World()
    scene: Scene = world.scene
    scene.add_default_ground_plane()
    
    # get gr1 robot usd
    assets_root_path = get_assets_root_path()
    robot_asset_path = assets_root_path + "/Isaac/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"
    
    # add the robot to the scene
    add_reference_to_stage(usd_path=robot_asset_path, prim_path="/World/gr1")
    gr1: Robot = scene.add(Robot(
        prim_path="/World/gr1", 
        name="gr1",
        position=np.array([0,0,0.95]),
    ))
    
    gr1_articulation_controller: ArticulationController = gr1.get_articulation_controller()

    # adding camera
    camera = Camera(
        prim_path="/World/camera",
        name="camera",
        position=np.array([0.12, 0.0, 1.55]),
        frequency=20,
        resolution=(256, 160),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 60, 0]), degrees=True),
    )
    camera.set_focal_length(1.3) # smaller => wider range of view
    camera.set_clipping_range(0.1, 2)
    
    
    # add table
    table_asset_path = assets_root_path + "/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01_physics.usd"
    add_reference_to_stage(usd_path=table_asset_path, prim_path="/World/table")
    table: XFormPrim = scene.add(XFormPrim(
        prim_paths_expr="/World/table",
        name = "table",
        visibilities=np.array([True,])
    ))
    
    # add cube for testing
    cube = scene.add(DynamicCuboid(
        prim_path="/World/random_cube",
        name="random_cube",
        position=np.array([0.5, 0.0, 1.5]),
        scale=np.array([0.0515, 0.0515, 0.0515]),
        color=np.array([0, 0, 1.0])))
    
    
    ## 2. setup_post_load
    world.reset()
    print("## 2. setup post-load")
    #print("DOF: " + str(gr1.num_dof)) # prints 2
    #print("Default Joint Positions: " + str(gr1.get_joint_positions()))
    #print("DOF names: " + str(gr1.dof_names))
    camera.initialize()
    camera.add_motion_vectors_to_frame()


    
    ## 3. run simulation
    print("## 3. run simulation")
    world.reset()
    gr1.set_world_pose(np.array([0,0,0.95])) # must set world position here!
    gr1.set_joint_positions(positions=gr1_config.default_joint_position)
    table.set_world_poses(positions=np.array([[0.53,0,0]]),
                          orientations=rot_utils.euler_angles_to_quats(np.array([0, 0, 90.0]), degrees=True).reshape((1,4)))


    # first, just initialize the world and wait
    for step in range(100):
        world.step(render=True)
    # for the actual simulation
    for step in range(5):
        current_joint_positions = gr1.get_joint_positions()
        #gr1_articulation_controller.apply_action(ArticulationAction(joint_positions=gr1_config.default_joint_position))
        #print(f"Current joint positions: {joint_positions}")
        world.step(render=True)
        image: np.ndarray = camera.get_rgba()
        plt.imshow(image[:, :, :3])
        plt.show()
        
        
        
    print("Joint Positions: " + str(gr1.get_joint_positions()))
        
    print("Simulation finished")
    simulation_app.close()
    







if __name__ == "__main__":
    main()



