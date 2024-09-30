import numpy as np
from numpy.ma.core import shape
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

    parent_stack = [-1]
    current_parent_index = -1

    for line in lines:
        line = line.strip()

        if line.startswith('HIERARCHY') or line.startswith('MOTION'):
            continue

        if line.startswith('ROOT') or line.startswith('JOINT'):
            parts = line.split()
            joint_name.append(parts[1])
            current_parent_index = parent_stack[-1]
            joint_parent.append(current_parent_index)
            parent_stack.append(len(joint_name) - 1)

        elif line.startswith('End Site'):
            joint_name.append(joint_name[parent_stack[-1]] + "_end")
            joint_parent.append(parent_stack[-1])
            parent_stack.append(parent_stack[-1]) # 仅占位，因为马上会pop

        elif line.startswith('OFFSET'):
            parts = line.split()
            offset = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            joint_offset.append(offset)

        elif line.startswith('}'):
            parent_stack.pop()

    joint_offset = np.array(joint_offset)

    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    num_joints = len(joint_name)
    joint_positions = np.zeros((num_joints, 3))
    joint_orientations = np.zeros((num_joints, 4))

    frame_data = motion_data[frame_id]

    root_position = frame_data[:3]
    root_orientation = frame_data[3:6]
    root_rotation_quat = R.from_euler('XYZ', root_orientation, degrees=True).as_quat()

    joint_positions[0] = root_position
    joint_orientations[0] = root_rotation_quat

    channel_index = 6
    for i in range(1, num_joints):
        if joint_name[i].endswith('_end'):
            continue

        parent_index = joint_parent[i]

        parent_position = joint_positions[parent_index]
        parent_rotation = joint_orientations[parent_index]

        offset = joint_offset[i]
        local_rotation = frame_data[channel_index : channel_index + 3]
        channel_index += 3

        local_rotation_quat = R.from_euler('XYZ', local_rotation, degrees=True).as_quat()
        global_rotation_quat = R.from_quat(joint_orientations[parent_index]) * R.from_quat(local_rotation_quat)
        joint_orientations[i] = global_rotation_quat.as_quat()

        global_position = parent_position + R.from_quat(parent_rotation).apply(offset)
        joint_positions[i] = global_position

    return joint_positions, joint_orientations

def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)

    A_motion_data = load_motion_data(A_pose_bvh_path)
    num_frames = A_motion_data.shape[0]

    joint_map = {A_joint_index: T_joint_name.index(A_joint_name) for A_joint_index in range(0, len(A_joint_name)) if A_joint_name[A_joint_index] in T_joint_name}

    motion_data = np.zeros((num_frames, len(T_joint_name) * 3 + 3))

    for frame_id in range(num_frames):
        for A_joint_index, T_joint_index in joint_map.items():
            A_joint_rotation = A_motion_data[frame_id, A_joint_index * 3 + 3 : A_joint_index * 3 + 6]
            A_joint_rotation = R.from_euler('XYZ', A_joint_rotation, degrees=True)

            T_joint_coord = T_joint_offset[T_joint_index]
            A_joint_coord = A_joint_offset[A_joint_index]

            rotation_matrix = A_joint_rotation.as_matrix()
            transformed_coord = rotation_matrix @ (A_joint_coord - T_joint_coord) + T_joint_coord

            T_joint_rotation = R.from_matrix(rotation_matrix).as_euler('XYZ', degrees=True)
            motion_data[frame_id, T_joint_index * 3 + 3 : T_joint_index * 3 + 6] = T_joint_rotation

    return motion_data

