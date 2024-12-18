import math
import time
from itertools import chain
from turtledemo.penrose import start

import numpy as np
from scipy.spatial.transform import Rotation as R

from lab1.task2_inverse_kinematics import MetaData

def solve_ccd_ik(meta_data: MetaData, joint_offsets, joint_positions, joint_orientations, target_position, precision, max_interation):
    joint_chain, path_name, end_part, root_part = meta_data.get_path_from_root_to_end() # joint_chain是root到end，end_part是从end到根骨骼，root_part是root到根骨骼
    joint_num = len(meta_data.joint_name)
    max_angle = np.radians(10)  # 单次最大旋转角度(弧度)
    distance = np.linalg.norm(joint_positions[joint_chain[-1]] - target_position)

    def update_joint(joint_index_in_chain):
        joint_index = joint_chain[joint_index_in_chain]
        joint_position = joint_positions[joint_index]
        joint_orientation = joint_orientations[joint_index]

        # 计算旋转
        tip_position = joint_positions[joint_chain[-1]]

        vec_to_tip = tip_position - joint_position
        vec_to_tip /= np.linalg.norm(vec_to_tip)

        vec_to_target = target_position - joint_position
        vec_to_target /= np.linalg.norm(vec_to_target)

        angle_to_rotate = np.arccos(np.clip(np.dot(vec_to_target, vec_to_tip), -1.0, 1.0))
        angle_to_rotate = np.clip(angle_to_rotate, -max_angle, max_angle)  # 保证效果的关键！！！
        if angle_to_rotate < precision:
            return

        rotation_axis = np.cross(vec_to_tip, vec_to_target)
        norm = np.linalg.norm(rotation_axis)
        if norm < 1e-6:
            return
        rotation_axis /= norm

        rotvec = angle_to_rotate * rotation_axis
        rotation = R.from_rotvec(rotvec)

        # 更新骨骼链

        if joint_index in root_part:
            joint_index_in_chain += 1

        parent_indices = joint_chain[joint_index_in_chain : ] # 用于更新其他骨骼

        while joint_index_in_chain < len(joint_chain):
            index = joint_chain[joint_index_in_chain]
            joint_orientations[index] = (R.from_quat(joint_orientations[index]) * rotation).as_quat()
            parent_in_chain_index = joint_chain[joint_index_in_chain - 1]  # 在骨骼链中的父节点

            if index in end_part:
                joint_positions[index] = joint_positions[parent_in_chain_index] + R.from_quat(joint_orientations[parent_in_chain_index]).apply(joint_offsets[index])
            elif index in root_part:
                joint_positions[index] = joint_positions[parent_in_chain_index] - R.from_quat(joint_orientations[index]).apply(joint_offsets[parent_in_chain_index])

            joint_index_in_chain += 1

        # 更新其余骨骼

        for index in range(0, joint_num):
            if index in joint_chain:
                continue

            parent_index = meta_data.joint_parent[index]
            if parent_index not in parent_indices:
                continue

            joint_orientations[index] = (R.from_quat(joint_orientations[index]) * rotation).as_quat()
            joint_positions[index] = joint_positions[parent_index] + R.from_quat(joint_orientations[parent_index]).apply(joint_offsets[index])
            parent_indices.append(index)

    for iteration in range(max_interation):
        for index_in_chain in range(len(joint_chain) - 2, 0, -1):
            update_joint(index_in_chain)

            if np.linalg.norm(joint_positions[joint_chain[-1]] - target_position) < precision:
                return joint_positions, joint_orientations
        distance = np.linalg.norm(joint_positions[joint_chain[-1]] - target_position)
        print(f"第{iteration}次: {distance}")

    return joint_positions, joint_orientations

def part1_inverse_kinematics(meta_data: MetaData, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    joint_offsets = []
    joint_parent = meta_data.joint_parent
    init_pos = meta_data.joint_initial_position
    for i in range(len(joint_parent)):
        parent_index = joint_parent[i]
        if parent_index < 0:
            joint_offsets.append(np.zeros(3))
            continue
        joint_offsets.append(init_pos[i] - init_pos[parent_index])

    return solve_ccd_ik(meta_data, joint_offsets, joint_positions, joint_orientations, target_pose, 0.01, 100)

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations