from isaacsim import SimulationApp
LOAD_WORLD_FILE = "./environments/gr1_default.usd"
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
        translation=np.array([0.13, 0.0, 0.07]),
        frequency=60,
        resolution=(256, 200),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 60, 0]), degrees=True),
    )
    camera.set_focal_length(1.0) # smaller => wider range of view
    camera.set_clipping_range(0.1, 2)
    
    
    ## 2. setup_post_load
    world.reset()
    print("## 2. setup post-load")
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    gr1_articulation_controller.set_gains(kps = np.array([3000.0]*54), kds = np.array([100.0]*54)) # p is the stiffness, d is the gain
    
    
    ## 3. run simulation
    print("## 3. run simulation")
    
    for simulation_num in range(5):
        
        print(f"Starting simulation {simulation_num}")
        world.reset()
        
        # first, just initialize the world and wait
        print("Waiting to initialize")
        for step in range(100):
            world.step(render=True)
        # for the actual simulation
        
        print("Start Simulation")
        for step in range(1000):
            current_joint_positions = gr1.get_joint_positions()
            world.step(render=True)
            obs: np.ndarray = camera.get_rgba()            
            gr1_articulation_controller.apply_action(ArticulationAction(joint_positions=current_joint_positions+0.01))
        
        print(f"Simulation {simulation_num} finished")
        
    simulation_app.close()

    
    
    
if __name__ == "__main__":
    main()