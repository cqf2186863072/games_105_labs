import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from lab1.task2_inverse_kinematics import MetaData


def solve_ccd_ik(meta_data: MetaData, joint_positions, joint_orientations, target_position, precision, max_interation):
    # end_part是从end到根骨骼，root_part是root到根骨骼
    joint_chain, end_part, root_part = meta_data.get_path_from_root_to_end()

    def update_joint(joint_index_in_chain):
        joint_index = joint_chain[joint_index_in_chain]
        joint_position = joint_positions[joint_index]
        joint_orientation = joint_orientations[joint_index]
        tip_position = joint_positions[joint_chain[-1]]

        # 计算旋转
        vec_to_tip = tip_position - joint_position
        vec_to_tip /= np.linalg.norm(vec_to_tip)

        vec_to_target = target_position - joint_position
        vec_to_target /= np.linalg.norm(vec_to_target)

        angle_to_rotate = np.arccos(np.clip(np.dot(vec_to_target, vec_to_tip), -1.0, 1.0))
        if angle_to_rotate < precision:
            return

        rotation_axis = np.cross(vec_to_tip, vec_to_target)
        rotation_axis /= np.linalg.norm(rotation_axis)

        rotvec = angle_to_rotate * rotation_axis
        rotation = R.from_rotvec(rotvec)

        # 应用旋转
        if joint_index in end_part:
            current_orientation = R.from_quat(joint_orientation)
            new_orientation = current_orientation * rotation

            joint_orientations[joint_index] = new_orientation

        # 更新所有子joint的位置和朝向




def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
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

    return joint_positions, joint_orientations

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