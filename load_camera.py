from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) 
from isaacsim.core.api import World
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.sensors.camera.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import matplotlib.pyplot as plt

def main():
    ## 1. setup scene
    print("## 1. setup scene")
    world = World()
    scene: Scene = world.scene
    scene.add_default_ground_plane()
    
    # adding camera
    camera = Camera(
        prim_path="/World/camera",
        name="camera",
        position=np.array([0.0, 0.0, 2.0]),
        frequency=20,
        resolution=(256, 256),
        orientation=rot_utils.euler_angles_to_quats(np.array([0, 90, 0]), degrees=True),
    )
        
    
    ## 2. setup_post_load
    world.reset()
    camera.initialize()
    camera.add_motion_vectors_to_frame()
    
    ## 3. run simulation
    print("## 3. run simulation")
    world.reset()
    
    # first, just initialize the world and wait
    for step in range(100):
        world.step(render=True)
    # for the actual simulation
    for step in range(5):
        world.step(render=True)
        print(camera.get_rgba().shape)
        plt.imshow(camera.get_rgba()[:, :, :3])
        plt.show()
        
    print("Simulation finished")
    simulation_app.close()


if __name__ == "__main__":
    main()