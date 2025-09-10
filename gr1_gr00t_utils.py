import numpy as np
import gr1_config
import requests


def make_gr00t_input(task: str, obs: np.ndarray, joint_positions: np.ndarray) -> dict:
    """
    make inference_ready version of input
    obs: (256, height, 3)
    joint_positions: (54, )
    """
    gr00t_input = {}
    gr00t_input["task"] = task
    gr00t_input["obs"] = make_square_img(obs).tolist()
    gr00t_input["state"] = {}
    for joint_part, joint_indexes in gr1_config.gr00t_joints_index.items():
        gr00t_input["state"][joint_part] = joint_positions[joint_indexes].tolist()
    
    return gr00t_input
    
    
    
def make_square_img(obs: np.ndarray) -> np.ndarray:
    obs_shape = obs.shape
    padding_height = int((256 - obs_shape[0]) / 2)
    output = np.zeros(shape=(256,256,3), dtype=np.uint8)
    output[padding_height:padding_height+obs_shape[0]] = obs
    return output
    
    
    
def request_gr00t_inference(payload: dict, url = "http://localhost:9876/inference") -> dict:
    response = requests.post(url, json=payload)
    return response.json()



def make_joint_position_from_gr00t_output(output: dict, timestep=1) -> np.ndarray:
    # timestep goes from 0 to 15
    joint_positions = np.zeros(shape = (54, ), dtype=float)
    for joint_part_name, actions in output.items():
        gr1_joint_part_name = joint_part_name[7:] # remove the action part
        joint_positions[gr1_config.gr00t_joints_index[gr1_joint_part_name]] = np.array(actions[timestep], dtype = float) # move to t + 1
    return joint_positions
