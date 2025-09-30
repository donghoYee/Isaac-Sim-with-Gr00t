# this uses the issacsim conda environment

from isaacsim import SimulationApp


# here are the parameters
EPISODE_NUM = 2
EACH_EPISODE_LEN = 30
RESULT_VIDEO_FILE = "./results/ExhaustPipe_plus_NutPouring_batch32_pipe_nut.mp4"
#LOAD_WORLD_FILE = "./environments/gr1_NutPouring.usd"
LOAD_WORLD_FILE = "./environments/gr1_exhaust_pipe.usd"

TASK = "Pick up the red beaker and tilt it to pour out 1 green nut into yellow bowl. Pick up the yellow bowl and place it on the metallic measuring scale."
#TASK = "Pickup the blue pipe and place it into the blue bin." # Exhaust Pipe task

# setting for the camera
CAMERA_HEIGHT = 200 # width is fixed to 256
CAMERA_FOCAL_LENGTH = 1.2
CAMERA_FORWARD_DIST = 0.25 # 0.25 for exhaust pipe
CAMERA_ANGLE = 70 # 70 for exhaust pipe

INFERENCE_SERVER_URL = "http://localhost:9876/inference"


simulation_app = SimulationApp({
    "headless": False, 
    "create_new_stage": False,
    "open_usd" : LOAD_WORLD_FILE,
    "sync_loads": True, # wait until asset loads
}) 

from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.sensors.camera.camera import Camera
import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
from isaacsim.core.prims import XFormPrim, RigidPrim
from isaacsim.core.utils.types import ArticulationAction
import cv2
import gr1_config, gr1_gr00t_utils





def main():
    ## 1. setup scene
    print("## 1. setup scene")
    world = World()
    scene: Scene = world.scene
    gr1: Robot = scene.add(Robot(
        prim_path="/World/gr1", 
        name="gr1",
    ))
    gr1_articulation_controller: ArticulationController = gr1.get_articulation_controller()
    
    # adding camera
    camera = Camera(
        prim_path="/World/gr1/head_yaw_link/camera",
        name="camera",
        translation=np.array([CAMERA_FORWARD_DIST, 0.0, 0.07]),
        frequency=60,
        resolution=(256, CAMERA_HEIGHT),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, CAMERA_ANGLE, 0]), degrees=True),
    )
    camera.set_focal_length(CAMERA_FOCAL_LENGTH) # smaller => wider range of view
    camera.set_clipping_range(0.1, 2)
    
    
    ## 2. setup_post_load
    world.reset()
    print("## 2. setup post-load")
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    gr1_articulation_controller.set_gains(kps = np.array([3000.0]*54), kds = np.array([100.0]*54)) # p is the stiffness, d is the gain
    
    
    ## 3. run simulation
    print("## 3. run simulation")
    fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
    video = cv2.VideoWriter(RESULT_VIDEO_FILE, fourcc, 30, (256, CAMERA_HEIGHT), isColor=True)
    
    for episode_idx in range(EPISODE_NUM):
        print(f"Starting episode {episode_idx}")
        world.reset()
        
        # set gr1 default position
        gr1.set_joint_positions(positions=gr1_config.default_joint_position)
        
        # first, just initialize the world and wait
        print("Waiting to initialize")
        for step in range(100):
            world.step(render=True)
            
        # for the actual simulation
        print("Start episode")
        for step in range(EACH_EPISODE_LEN):
            world.step(render=True)
            obs: np.ndarray = camera.get_rgba()
            image = obs[:, :, :3]
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)) 
            current_joint_positions = gr1.get_joint_positions()
            
            # inference to gr00t server
            print(f"Episode {episode_idx} step {step} calling gr00t inference")        
            gr00t_inference_input = gr1_gr00t_utils.make_gr00t_input(task=TASK, obs=image, joint_positions=current_joint_positions)
            gr00t_output = gr1_gr00t_utils.request_gr00t_inference(payload=gr00t_inference_input, url=INFERENCE_SERVER_URL)
            for timestep in range(0, 16):
                action_joint_position = gr1_gr00t_utils.make_joint_position_from_gr00t_output(gr00t_output, timestep=timestep)
                gr1_articulation_controller.apply_action(ArticulationAction(joint_positions=action_joint_position))
                if timestep == 15: break # at the end, do not step, as it will be done by the outer loop
                world.step(render=True)
                obs: np.ndarray = camera.get_rgba()
                image = obs[:, :, :3]
                video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            
 
        
        print(f"Episode {episode_idx} finished")
        
    video.release()
    simulation_app.close()

    
if __name__ == "__main__":
    main()