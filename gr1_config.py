  
gr1_gr00t_joint_space = {
    "left_arm": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_wrist_yaw_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
    ],
    "right_arm": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_yaw_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
    ],
    "left_hand": [
        "L_pinky_proximal_joint",
        "L_ring_proximal_joint",
        "L_middle_proximal_joint",
        "L_index_proximal_joint",
        "L_thumb_proximal_yaw_joint",
        "L_thumb_proximal_pitch_joint",
        # Mimic joints
        # - L_index_intermediate_joint
        # - L_middle_intermediate_joint
        # - L_ring_intermediate_joint
        # - L_pinky_intermediate_joint
        # - L_thumb_distal_joint
    ],
    "right_hand": [
        "R_pinky_proximal_joint",
        "R_ring_proximal_joint",
        "R_middle_proximal_joint",
        "R_index_proximal_joint",
        "R_thumb_proximal_yaw_joint",
        "R_thumb_proximal_pitch_joint",
        # Mimic joints
        # - R_index_intermediate_joint
        # - R_middle_intermediate_joint
        # - R_ring_intermediate_joint
        # - R_pinky_intermediate_joint
        # - R_thumb_distal_joint
    ]
}


gr1_joints_index = {
    'left_hip_roll_joint': 0,
    'right_hip_roll_joint': 1,
    'waist_yaw_joint': 2,
    'left_hip_yaw_joint': 3,
    'right_hip_yaw_joint': 4,
    'waist_pitch_joint': 5,
    'left_hip_pitch_joint': 6,
    'right_hip_pitch_joint': 7,
    'waist_roll_joint': 8,
    'left_knee_pitch_joint': 9,
    'right_knee_pitch_joint': 10,
    'head_roll_joint': 11,
    'left_shoulder_pitch_joint': 12,
    'right_shoulder_pitch_joint': 13,
    'left_ankle_pitch_joint': 14,
    'right_ankle_pitch_joint': 15,
    'head_pitch_joint': 16,
    'left_shoulder_roll_joint': 17,
    'right_shoulder_roll_joint': 18,
    'left_ankle_roll_joint': 19,
    'right_ankle_roll_joint': 20,
    'head_yaw_joint': 21,
    'left_shoulder_yaw_joint': 22,
    'right_shoulder_yaw_joint': 23,
    'left_elbow_pitch_joint': 24,
    'right_elbow_pitch_joint': 25,
    'left_wrist_yaw_joint': 26,
    'right_wrist_yaw_joint': 27,
    'left_wrist_roll_joint': 28,
    'right_wrist_roll_joint': 29,
    'left_wrist_pitch_joint': 30,
    'right_wrist_pitch_joint': 31,
    'L_index_proximal_joint': 32,
    'L_middle_proximal_joint': 33,
    'L_pinky_proximal_joint': 34,
    'L_ring_proximal_joint': 35,
    'L_thumb_proximal_yaw_joint': 36,
    'R_index_proximal_joint': 37,
    'R_middle_proximal_joint': 38,
    'R_pinky_proximal_joint': 39,
    'R_ring_proximal_joint': 40,
    'R_thumb_proximal_yaw_joint': 41,
    'L_index_intermediate_joint': 42,
    'L_middle_intermediate_joint': 43,
    'L_pinky_intermediate_joint': 44,
    'L_ring_intermediate_joint': 45,
    'L_thumb_proximal_pitch_joint': 46,
    'R_index_intermediate_joint': 47,
    'R_middle_intermediate_joint': 48,
    'R_pinky_intermediate_joint': 49,
    'R_ring_intermediate_joint': 50,
    'R_thumb_proximal_pitch_joint': 51,
    'L_thumb_distal_joint': 52,
    'R_thumb_distal_joint': 53,
}

# this corresponds to the index of the joint on the whole gr1 robot to the needed gr00t joint above
gr00t_joints_index = {
    "left_arm": [gr1_joints_index[gr1_joint_name] for gr1_joint_name in gr1_gr00t_joint_space["left_arm"]],
    "right_arm": [gr1_joints_index[gr1_joint_name] for gr1_joint_name in gr1_gr00t_joint_space["right_arm"]],
    "left_hand": [gr1_joints_index[gr1_joint_name] for gr1_joint_name in gr1_gr00t_joint_space["left_hand"]],
    "right_hand": [gr1_joints_index[gr1_joint_name] for gr1_joint_name in gr1_gr00t_joint_space["right_hand"]],
}




if __name__ == "__main__":
    print(gr00t_joints_index)