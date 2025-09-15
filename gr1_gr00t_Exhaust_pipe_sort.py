from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from isaacsim.core.api import World
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import XFormPrim, RigidPrim
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
import matplotlib.pyplot as plt
import gr1_config
import gr1_gr00t_utils
import cv2



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
        prim_path="/World/gr1/head_yaw_link/camera",
        name="camera",
        translation=np.array([0.13, 0.0, 0.07]),
        frequency=60,
        resolution=(256, 200),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 60, 0]), degrees=True),
    )
    camera.set_focal_length(1.0) # smaller => wider range of view
    camera.set_clipping_range(0.1, 2)
    # camera.set_opencv_pinhole_properties(fx=1000, fy=1000)
    
    
    # add table
    table_asset_path = assets_root_path + "/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01_physics.usd"
    add_reference_to_stage(usd_path=table_asset_path, prim_path="/World/table")
    table: XFormPrim = scene.add(XFormPrim(
        prim_paths_expr="/World/table",
        name = "table",
        visibilities=np.array([True,])
    ))
    
    # add prim for testing
    object = scene.add(DynamicCylinder(
        prim_path="/World/random_cylinder",
        name="random_cylinder",
        position=np.array([0.42, -0.0, 1.2]),
        scale=np.array([0.02, 0.02, 0.25]),
        color=np.array([0.1, 0.1, 1.0]),
        mass = 2,
    ))
    
    # add bin
    bin_asset_path = assets_root_path + "/Isaac/Props/KLT_Bin/small_KLT_visual_collision.usd"
    add_reference_to_stage(usd_path=bin_asset_path, prim_path="/World/bin")
    bin: RigidPrim = scene.add(RigidPrim(
        prim_paths_expr="/World/bin",
        name = "bin",
        scales=np.array([[1.3, 1.5, 0.3]]),
        visibilities=np.array([True,]),
    ))
    
    ## 2. setup_post_load
    world.reset()
    print("## 2. setup post-load")
    #print("DOF: " + str(gr1.num_dof)) # prints 2
    #print("Default Joint Positions: " + str(gr1.get_joint_positions()))
    #print("DOF names: " + str(gr1.dof_names))
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    
    
    gr1_articulation_controller.set_gains(kps = np.array([3000.0]*54), kds = np.array([100.0]*54)) # p is the stiffness, d is the gain



    
    ## 3. run simulation
    print("## 3. run simulation")
    PROMPT = "Pickup the blue pipe and place it into the pink bin."
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(f'./result.mp4', fourcc, 30, (256, 200), isColor=True)
    for simulation_num in range(5):
        
        print(f"Starting simulation {simulation_num}")
        world.reset()
        gr1.set_world_pose(np.array([0,0,0.95])) # must set world position here!
        gr1.set_joint_positions(positions=gr1_config.default_joint_position)
        table.set_world_poses(positions=np.array([[0.55,0,0.00]]),
                            orientations=rot_utils.euler_angles_to_quats(np.array([0, 0, 90.0]), degrees=True).reshape((1,4)))

        object.set_world_pose(np.array([0.33 + 0.1 * (np.random.rand() - 0.5), 0.05 +  0.05 * (np.random.rand() - 0.5), 1.1]))
        bin.set_world_poses(positions=np.array([[0.5,-0.15,1.0]]),
                            orientations=rot_utils.euler_angles_to_quats(np.array([0, 0, 90.0]), degrees=True).reshape((1,4)))
        
        # first, just initialize the world and wait
        print("Waiting to initialize")
        for step in range(100):
            world.step(render=True)
        # for the actual simulation
        
        print("Start Simulation")
        for step in range(30):
            current_joint_positions = gr1.get_joint_positions()
            #print("current pos")
            #print(current_joint_positions)
            world.step(render=True)
            obs: np.ndarray = camera.get_rgba()
            video.write(cv2.cvtColor(obs[:,:,:3], cv2.COLOR_RGB2BGR))
            
            # inference to gr00t server
            print(f"Simulation {simulation_num} step {step} calling gr00t inference")
            gr00t_inference_input = gr1_gr00t_utils.make_gr00t_input(task=PROMPT, obs=obs[:,:,:3], joint_positions=current_joint_positions)
            gr00t_output = gr1_gr00t_utils.request_gr00t_inference(payload=gr00t_inference_input, url="http://localhost:9876/inference")
            
            for timestep in range(0, 16):
                action_joint_position = gr1_gr00t_utils.make_joint_position_from_gr00t_output(gr00t_output, timestep=timestep)
                gr1_articulation_controller.apply_action(ArticulationAction(joint_positions=action_joint_position))
                world.step(render=True)
                obs: np.ndarray = camera.get_rgba()
                video.write(cv2.cvtColor(obs[:,:,:3], cv2.COLOR_RGB2BGR))
        
        print(f"Simulation {simulation_num} finished")
        
    video.release()    
    simulation_app.close()
    
    
    








if __name__ == "__main__":
    main()
