import argparse
import json
import sys
sys.path.append('../')

from utils import *
from fixed.model.ekf import ExtendedKalmanFilterPyTorch

import torch
import torch.nn as nn

import math

def quaternion_to_euler(q):
    """
    Convert a quaternion into Euler angles (roll, pitch, yaw)
    roll is rotation around x axis, pitch is rotation around y axis and
    yaw is rotation around z axis.
    
    Parameters:
    q (tuple): A quaternion in the form (w, x, y, z)
    
    Returns:
    tuple: Euler angles (roll, pitch, yaw)
    """
    # Extract the values from q
    w, x, y, z = q
    
    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    # Calculate pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp) # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    
    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return [roll, pitch, yaw]

# left: [-acc_y, acc_z, acc_x]
# right: [acc_y, -acc_z, acc_x]
def fix_acc(acc):
    global hand
    ax, ay, az = acc[2], -acc[0], acc[1]
    if hand == 'left':
        new_acc = [ax, ay, az]
    else:
        new_acc = [ax, -ay, -az]
    return new_acc

# left: [gyro_y, -gyro_z, -gyro_x]
# right: [-gyro_y, gyro_z, -gyro_x]
def fix_gyro(gyro):
    global hand
    gx, gy, gz = -gyro[2], gyro[0], -gyro[1]
    if hand == 'left':
        new_gyro = [gx, gy, gz]
    else:
        new_gyro = [gx, -gy, -gz]
    return new_gyro

# left: [gyro_y, -gyro_z, -gyro_x]
# right: [-gyro_y, gyro_z, -gyro_x]
def angles_to_unity(roll_pitch_yaw):
    global hand
    roll, pitch, yaw = roll_pitch_yaw
    x, y, z = pitch, -yaw, -roll
    if hand == 'left':
        new_angles = [x, y, z]
    else:
        new_angles = [-x, -y, z]
    return new_angles

# left: [gyro_y, -gyro_z, -gyro_x]
# right: [-gyro_y, gyro_z, -gyro_x]
def unity_to_angles(roll_pitch_yaw):
    global hand
    roll, pitch, yaw = roll_pitch_yaw
    x, y, z = -yaw, roll, -pitch
    if hand == 'left':
        new_angles = [x, y, z]
    else:
        new_angles = [x, -y, -z]
    return new_angles

# Global variables
parser = argparse.ArgumentParser(description='')
parser.add_argument('--hand', help='If you collect the left hand data, set "left", else "right".', choices=['left', 'right'], required=True)
hand = parser.parse_args().hand

with open(f'./dataset_{hand}_lsm_fixed.json', 'r', encoding='utf-8') as f_info:
    dataset = json.load(f_info)

data_file = open(f'dataset_{hand}.json', 'w', encoding='utf-8')
new_dataset = {
    'data':[]
}

for data in dataset['data']:
    new_data = {}
    ekf = ExtendedKalmanFilterPyTorch()
    roll, pitch, yaw = unity_to_angles(quaternion_to_euler(data['past_wrist_rotation'][-1]))
    ekf.x = torch.tensor([
        [roll],
        [pitch],
        [yaw]
    ])
    new_data['past_wrist_position'] = data['past_wrist_position']
    new_data['past_wrist_rotation'] = rotation_matrix_to_6d(compute_rotation_matrix_from_quaternion(torch.tensor(data['past_wrist_rotation']).unsqueeze(0))).squeeze(0).tolist()
    new_data['past_head_position'] = data['past_head_position']
    new_data['past_head_rotation'] = rotation_matrix_to_6d(compute_rotation_matrix_from_quaternion(torch.tensor(data['past_head_rotation']).unsqueeze(0))).squeeze(0).tolist()
    new_data['acc'] = data['acc']
    new_data['gyro'] = data['gyro']
    new_data['head_position'] = data['head_position']
    new_data['head_rotation'] = rotation_matrix_to_6d(compute_rotation_matrix_from_quaternion(torch.tensor(data['head_rotation']).unsqueeze(0))).squeeze(0).tolist()
    new_data['kalman_filter_output'] = []
    for i in range(len(data['acc'])):
        out = ekf(z=calc_z(fix_acc(new_data['acc'][i])), u=calc_u(fix_gyro(new_data['gyro'][i]), dt=0.005))
        new_data['kalman_filter_output'].append(rotation_matrix_for_unity_from_radians(angles_to_unity(out)).tolist())
    new_data['kalman_filter_output'] = rotation_matrix_to_6d(torch.tensor(new_data['kalman_filter_output']).unsqueeze(0)).squeeze(0).tolist()
    new_data['wrist_position'] = data['wrist_position']
    new_data['wrist_rotation'] = rotation_matrix_to_6d(compute_rotation_matrix_from_quaternion(torch.tensor(data['wrist_rotation']).unsqueeze(0).unsqueeze(0))).squeeze(0).squeeze(0).tolist()
    new_dataset['data'].append(new_data)
json.dump(new_dataset, data_file, indent=4)
data_file.close()