from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False}) 

from isaacsim.core.api import World
import numpy as np
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.nucleus import get_assets_root_path
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.api.robots import Robot
from isaacsim.core.api.controllers.articulation_controller import ArticulationController
from isaacsim.core.utils.types import ArticulationAction






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
    ))
    
    gr1_articulation_controller: ArticulationController = gr1.get_articulation_controller()
    
    
    ## 2. setup_post_load
    world.reset()
    print("## 2. setup post-load")
    print("DOF: " + str(gr1.num_dof)) # prints 2
    print("Default Joint Positions: " + str(gr1.get_joint_positions()))
    print("DOF names: " + str(gr1.dof_names))
    
    ## 3. run simulation
    print("## 3. run simulation")
    world.reset()
    gr1.set_world_pose(np.array([0,0,0.95])) # must set world position here!

    for step in range(1000):
        current_joint_positions = gr1.get_joint_positions()
        gr1_articulation_controller.apply_action(ArticulationAction(joint_positions=current_joint_positions+0.01))
        #print(f"Current joint positions: {joint_positions}")
        world.step(render=True)
        
    print("Joint Positions: " + str(gr1.get_joint_positions()))
        
    print("Simulation finished")
    simulation_app.close()
    







if __name__ == "__main__":
    main()



