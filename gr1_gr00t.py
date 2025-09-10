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
import gr1_gr00t_utils



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
    PROMPT = "pick up the blue cube"
    for simulation_num in range(5):
        print(f"Starting simulation {simulation_num}")
        world.reset()
        gr1.set_world_pose(np.array([0,0,0.95])) # must set world position here!
        gr1.set_joint_positions(positions=gr1_config.default_joint_position)
        table.set_world_poses(positions=np.array([[0.53,0,0]]),
                            orientations=rot_utils.euler_angles_to_quats(np.array([0, 0, 90.0]), degrees=True).reshape((1,4)))


        # first, just initialize the world and wait
        print("Waiting to initialize")
        for step in range(100):
            world.step(render=True)
        # for the actual simulation
        
        print("Start Simulation")
        for step in range(20):
            current_joint_positions = gr1.get_joint_positions()
            world.step(render=True)
            obs: np.ndarray = camera.get_rgba()
            
            # inference to gr00t server
            print(f"Simulation {simulation_num} step {step} calling gr00t inference")
            gr00t_inference_input = gr1_gr00t_utils.make_gr00t_input(task=PROMPT, obs=obs[:,:,:3], joint_positions=current_joint_positions)
            gr00t_output = gr1_gr00t_utils.request_gr00t_inference(payload=gr00t_inference_input, url="http://localhost:9876/inference")
            
            for timestep in range(16):
                action_joint_position = gr1_gr00t_utils.make_joint_position_from_gr00t_output(gr00t_output, timestep=timestep)
                gr1_articulation_controller.apply_action(ArticulationAction(joint_positions=action_joint_position))
                world.step(render=True)
                        
        print(f"Simulation {simulation_num} finished")
    simulation_app.close()
    
    
    








if __name__ == "__main__":
    main()
