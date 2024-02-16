import http.server
import socketserver
import threading
import socket
import json
from datetime import datetime
import sys
sys.path.append('./model/')

from utils import *
from joycon_manage import *
from model.pose_estimation import PoseEstimationNetwork
from model.ekf import ExtendedKalmanFilterForPoseEstimation
from model.filters import *

import numpy as np
import torch
import torch.nn as nn

# Global variables
UDP_HOST = '127.0.0.1'
UDP_PORT_LEFT = 8081
UDP_PORT_RIGHT = 8082
UDP_PORT_TAP = 8083
udp_server_running = False
httpd = None

received_tracking_data_left = None
latest_tracking_data_left = None

received_tracking_data_right = None
latest_tracking_data_right = None

left_hand_info = {
    'past_wrist_position': [],
    'past_wrist_rotation': [],
    'past_head_position': [],
    'past_head_rotation': [],
    'acceleration': [],
    'gyroscope': [],
    'ekf_output':[],
    'head_position': [],
    'head_rotation': []
}

right_hand_info = {
    'past_wrist_position': [],
    'past_wrist_rotation': [],
    'past_head_position': [],
    'past_head_rotation': [],
    'acceleration': [],
    'gyroscope': [],
    'ekf_output':[],
    'head_position': [],
    'head_rotation': []
}

def make_input(hand_info_orig):
    hand_info = hand_info_orig.copy()
    if len(hand_info['past_wrist_position']) < 5 or len(hand_info['acceleration']) <= 8 or len(hand_info['head_position']) == 0:
        return None, None
    try:
        wrist_pos = torch.tensor(hand_info['past_wrist_position']).unsqueeze(0)
        wrist_rot = rotation_matrix_to_6d(compute_rotation_matrix_from_quaternion(torch.tensor(hand_info['past_wrist_rotation']).unsqueeze(0)))
        past_head_pos = torch.tensor(hand_info['past_head_position']).unsqueeze(0)
        past_head_rot = rotation_matrix_to_6d(compute_rotation_matrix_from_quaternion(torch.tensor(hand_info['past_head_rotation']).unsqueeze(0)))
        ekf_pos = torch.tensor(hand_info['ekf_output'][:, :3]).unsqueeze(0)
        ekf_rot = rotation_matrix_to_6d(compute_rotation_matrix_from_quaternion(torch.tensor(hand_info['ekf_output'][:, 3:]).unsqueeze(0)))
        upscale_pos = nn.Upsample(size=(ekf_pos.size(1), 3), mode='bicubic', align_corners=True)
        upscale_rot = nn.Upsample(size=(ekf_pos.size(1), 4), mode='bicubic', align_corners=True)
        head_pos = upscale_pos(torch.tensor(hand_info['head_position']).unsqueeze(0).unsqueeze(0)).squeeze(0)
        head_rot = rotation_matrix_to_6d(compute_rotation_matrix_from_quaternion(upscale_rot(torch.tensor(hand_info['head_rotation']).unsqueeze(0).unsqueeze(0)).squeeze(0)))
        
        x = torch.cat([wrist_pos, wrist_rot, past_head_pos, past_head_rot], dim = 2)
        y = torch.transpose(torch.cat([ekf_pos, ekf_rot, head_pos, head_rot], dim = 2), 1, 2)
        return x, y
    except Exception as e:
        print(e)
        print(f'wrist_pos:{wrist_pos.size()}')
        print(f'wrist_rot:{wrist_rot.size()}')
        print(f'past_head_pos:{past_head_pos.size()}')
        print(f'past_head_rot:{past_head_rot.size()}')
        print(f'ekf_output:{ekf_pos.size()}')
        print(f'head_pos:{head_pos.size()}')
        print(f'head_rot:{head_rot.size()}')
        return None, None

def apply_offset_of_tap_position(pos, rot_matrix, offset=[[0.05], [0.0], [0.0]]):
    return (rot_matrix @ torch.tensor(offset)).T + pos

def is_tapped(sensor_history):
    try:
        if np.sqrt(np.sum(np.array(sensor_history[-1]) ** 2)) > 30.0:
            return True
        return False
    except Exception as e:
        raise e

def is_tapped_sequence(sensor_history):
    try:
        if len(sensor_history) >= 10:
            if np.max(np.sqrt(np.sum(np.array(sensor_history[-11:-1]) ** 2, axis=1))) > 30.0:
                return True
        else:
            if np.max(np.sqrt(np.sum(np.array(sensor_history[:]) ** 2, axis=1))) > 30.0:
                return True
        return False
    except Exception as e:
        raise e

def predict_part_of_body_left():
    global received_tracking_data_left
    global latest_tracking_data_left
    global left_hand_info
    if latest_tracking_data_left['wrist_position_l'] is not None:
        try:
            wrist_pos = [
                latest_tracking_data_left['wrist_position_l']['x'],
                latest_tracking_data_left['wrist_position_l']['y'],
                latest_tracking_data_left['wrist_position_l']['z']
            ]
            wrist_rot = [
                latest_tracking_data_left['wrist_rotation_l']['w'],
                latest_tracking_data_left['wrist_rotation_l']['x'],
                latest_tracking_data_left['wrist_rotation_l']['y'],
                latest_tracking_data_left['wrist_rotation_l']['z']
            ]
            wrist_position = torch.tensor(wrist_pos)
            rot_matrix = compute_rotation_matrix_from_quaternion(torch.tensor(wrist_rot).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        except:
            wrist_position = torch.tensor(left_hand_info['past_wrist_position'][-1])
            rot_matrix = compute_rotation_matrix_from_quaternion(torch.tensor(left_hand_info['past_wrist_rotation'][-1]).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        wrist_position = apply_offset_of_tap_position(wrist_position, rot_matrix).numpy()
        parts = [
            part for part in received_tracking_data_left.keys() if 'Position' in part and not 'Hand' in part
        ]
        part_positions = np.array([
            [
                received_tracking_data_left[part]['x'],
                received_tracking_data_left[part]['y'],
                received_tracking_data_left[part]['z']
            ] if not 'head' in part else [
                received_tracking_data_left[part]['x'] - 0.05,
                received_tracking_data_left[part]['y'] / 0.92,
                received_tracking_data_left[part]['z'] - 0.05
            ] for part in parts 
        ])
        diff_parts = part_positions - wrist_position
        part_distance = np.linalg.norm(diff_parts, axis=1)
        nearest_part_index = np.argmin(part_distance)
        return parts[nearest_part_index]
    else:
        return None

def predict_part_of_body_right():
    global received_tracking_data_right
    global latest_tracking_data_right
    global right_hand_info
    if latest_tracking_data_right['wrist_position_r'] is not None:
        try:
            wrist_pos = [
                latest_tracking_data_right['wrist_position_r']['x'],
                latest_tracking_data_right['wrist_position_r']['y'],
                latest_tracking_data_right['wrist_position_r']['z']
            ]
            wrist_rot = [
                latest_tracking_data_right['wrist_rotation_r']['w'],
                latest_tracking_data_right['wrist_rotation_r']['x'],
                latest_tracking_data_right['wrist_rotation_r']['y'],
                latest_tracking_data_right['wrist_rotation_r']['z']
            ]
            wrist_position = torch.tensor(wrist_pos)
            rot_matrix = compute_rotation_matrix_from_quaternion(torch.tensor(wrist_rot).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        except:
            wrist_position = torch.tensor(right_hand_info['past_wrist_position'][-1])
            rot_matrix = compute_rotation_matrix_from_quaternion(torch.tensor(right_hand_info['past_wrist_rotation'][-1]).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        wrist_position = apply_offset_of_tap_position(wrist_position, rot_matrix, offset=[[-0.03], [0.0], [0.0]]).numpy()
        parts = [
            part for part in received_tracking_data_right.keys() if 'Position' in part and not 'Hand' in part
        ]
        part_positions = np.array([
            [
                received_tracking_data_right[part]['x'],
                received_tracking_data_right[part]['y'],
                received_tracking_data_right[part]['z']
            ] if not 'head' in part else [
                received_tracking_data_right[part]['x'] + 0.05,
                received_tracking_data_right[part]['y'] / 0.92,
                received_tracking_data_right[part]['z'] - 0.05
            ] for part in parts
        ])
        diff_parts = part_positions - wrist_position
        part_distance = np.linalg.norm(diff_parts, axis=1)
        nearest_part_index = np.argmin(part_distance)
        return parts[nearest_part_index]
    else:
        return None

def udp_server_left():
    global udp_server_running
    global received_tracking_data_left
    global latest_tracking_data_left

    global left_hand_info

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((UDP_HOST, UDP_PORT_LEFT))
    print(f"UDP server listening on {UDP_HOST}:{UDP_PORT_LEFT}")

    latest_tracking_data_left = {
        'wrist_position_l': None,
        'wrist_rotation_l': None,
    }

    while udp_server_running:
        data, addr = server_socket.recvfrom(2048)
        received_tracking_data_left = json.loads(data.decode('utf-8'))
        # print(f"received message: {received_tracking_data['isTrackedLeft']} from {addr}")

        if received_tracking_data_left['headPosition'] is not None:
            head_pos = [
                received_tracking_data_left['headPosition']['x'],
                received_tracking_data_left['headPosition']['y'],
                received_tracking_data_left['headPosition']['z']
            ]

            head_rot = [
                received_tracking_data_left['headRotation']['w'],
                received_tracking_data_left['headRotation']['x'],
                received_tracking_data_left['headRotation']['y'],
                received_tracking_data_left['headRotation']['z']
            ]
        
            if len(left_hand_info['past_head_position']) < 5:
                left_hand_info['past_head_position'].append(head_pos)
            else:
                left_hand_info['past_head_position'][0:-1] = left_hand_info['past_head_position'][1:]
                left_hand_info['past_head_position'][-1] = head_pos

            if len(left_hand_info['past_head_rotation']) < 5:
                left_hand_info['past_head_rotation'].append(head_rot)
            else:
                left_hand_info['past_head_rotation'][0:-1] = left_hand_info['past_head_rotation'][1:]
                left_hand_info['past_head_rotation'][-1] = head_rot

        if received_tracking_data_left['isTrackedLeft']:
            left_hand_info['head_position'] = []
            left_hand_info['head_rotation'] = []
            wrist_pos_l = [
                received_tracking_data_left['leftHandPosition']['x'],
                received_tracking_data_left['leftHandPosition']['y'],
                received_tracking_data_left['leftHandPosition']['z']
            ]
            wrist_rot_l = [
                received_tracking_data_left['leftHandRotation']['w'],
                received_tracking_data_left['leftHandRotation']['x'],
                received_tracking_data_left['leftHandRotation']['y'],
                received_tracking_data_left['leftHandRotation']['z']
            ]

            if len(left_hand_info['past_wrist_position']) < 5:
                left_hand_info['past_wrist_position'].append(wrist_pos_l)
            else:
                left_hand_info['past_wrist_position'][0:-1] = left_hand_info['past_wrist_position'][1:]
                left_hand_info['past_wrist_position'][-1] = wrist_pos_l
            
            if len(left_hand_info['past_wrist_rotation']) < 5:
                left_hand_info['past_wrist_rotation'].append(wrist_rot_l)
            else:
                left_hand_info['past_wrist_rotation'][0:-1] = left_hand_info['past_wrist_rotation'][1:]
                left_hand_info['past_wrist_rotation'][-1] = wrist_rot_l

            latest_tracking_data_left['wrist_position_l'] = received_tracking_data_left['leftHandPosition']
            latest_tracking_data_left['wrist_rotation_l'] = received_tracking_data_left['leftHandRotation']
        else:
            if received_tracking_data_left['headPosition'] is not None:
                left_hand_info['head_position'].append(head_pos)
                left_hand_info['head_rotation'].append(head_rot)

            if latest_tracking_data_left['wrist_position_l'] is not None:
                # print(f"wrist_position_l:{latest_tracking_data_left['wrist_position_l']}")
                response = json.dumps(latest_tracking_data_left, ensure_ascii=False, indent=2)
                server_socket.sendto(response.encode('utf-8'), addr)

    server_socket.close()
    print("UDP server stopped")

def udp_server_right():
    global udp_server_running
    global received_tracking_data_right
    global latest_tracking_data_right

    global right_hand_info

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((UDP_HOST, UDP_PORT_RIGHT))
    print(f"UDP server listening on {UDP_HOST}:{UDP_PORT_RIGHT}")

    latest_tracking_data_right = {
        'wrist_position_r': None,
        'wrist_rotation_r': None
    }

    while udp_server_running:
        data, addr = server_socket.recvfrom(2048)
        received_tracking_data_right = json.loads(data.decode('utf-8'))
        # print(f"received message: {received_tracking_data_right} from {addr}")
        if received_tracking_data_right['headPosition'] is not None:
            head_pos = [
                received_tracking_data_right['headPosition']['x'],
                received_tracking_data_right['headPosition']['y'],
                received_tracking_data_right['headPosition']['z']
            ]

            head_rot = [
                received_tracking_data_right['headRotation']['w'],
                received_tracking_data_right['headRotation']['x'],
                received_tracking_data_right['headRotation']['y'],
                received_tracking_data_right['headRotation']['z']
            ]
        
            if len(right_hand_info['past_head_position']) < 5:
                right_hand_info['past_head_position'].append(head_pos)
            else:
                right_hand_info['past_head_position'][0:-1] = right_hand_info['past_head_position'][1:]
                right_hand_info['past_head_position'][-1] = head_pos

            if len(right_hand_info['past_head_rotation']) < 5:
                right_hand_info['past_head_rotation'].append(head_rot)
            else:
                right_hand_info['past_head_rotation'][0:-1] = right_hand_info['past_head_rotation'][1:]
                right_hand_info['past_head_rotation'][-1] = head_rot

        if received_tracking_data_right['isTrackedRight']:
            right_hand_info['head_position'] = []
            right_hand_info['head_rotation'] = []
            wrist_pos_r = [
                received_tracking_data_right['rightHandPosition']['x'],
                received_tracking_data_right['rightHandPosition']['y'],
                received_tracking_data_right['rightHandPosition']['z']
            ]
            wrist_rot_r = [
                received_tracking_data_right['rightHandRotation']['w'],
                received_tracking_data_right['rightHandRotation']['x'],
                received_tracking_data_right['rightHandRotation']['y'],
                received_tracking_data_right['rightHandRotation']['z']
            ]

            if len(right_hand_info['past_wrist_position']) < 5:
                right_hand_info['past_wrist_position'].append(wrist_pos_r)
            else:
                right_hand_info['past_wrist_position'][0:-1] = right_hand_info['past_wrist_position'][1:]
                right_hand_info['past_wrist_position'][-1] = wrist_pos_r
            
            if len(right_hand_info['past_wrist_rotation']) < 5:
                right_hand_info['past_wrist_rotation'].append(wrist_rot_r)
            else:
                right_hand_info['past_wrist_rotation'][0:-1] = right_hand_info['past_wrist_rotation'][1:]
                right_hand_info['past_wrist_rotation'][-1] = wrist_rot_r
            
            latest_tracking_data_right['wrist_position_r'] = received_tracking_data_right['rightHandPosition']
            latest_tracking_data_right['wrist_rotation_r'] = received_tracking_data_right['rightHandRotation']
        else:
            if received_tracking_data_right['headPosition'] is not None:
                right_hand_info['head_position'].append(head_pos)
                right_hand_info['head_rotation'].append(head_rot)
            if latest_tracking_data_right['wrist_position_r'] is not None:
                # print(f"wrist_position_r:{latest_tracking_data_right['wrist_position_r']}")
                response = json.dumps(latest_tracking_data_right, ensure_ascii=False, indent=2)
                server_socket.sendto(response.encode('utf-8'), addr)

    server_socket.close()
    print("UDP server stopped")

def get_data_from_imu():
    global udp_server_running
    global latest_tracking_data_left
    global latest_tracking_data_right
    global received_tracking_data_left
    global received_tracking_data_right

    global left_hand_info
    global right_hand_info

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"UDP server listening on {UDP_HOST}:{UDP_PORT_TAP}")

    # joycon settings
    joycon_l, joycon_r = joycon_activate()
    tapped_flag_l = False
    tapped_flag_r = False
    
    latest_tap_data = {
        'is_tapped_l': False,
        'tapped_part_l': None,
        'is_tapped_r': False,
        'tapped_part_r': None
    }
    raw_l = RawDataForPoseEstimation()
    raw_r = RawDataForPoseEstimation()

    while udp_server_running:
        input_report_l = joycon_l.read(49)
        input_report_r = joycon_r.read(49)
        for i in range(3):
            if received_tracking_data_left is not None:
                if received_tracking_data_left['isTrackedLeft']:
                    left_hand_info['acceleration'] = []
                    left_hand_info['gyroscope'] = []

            if received_tracking_data_right is not None:
                if received_tracking_data_right['isTrackedRight']:
                    right_hand_info['acceleration'] = []
                    right_hand_info['gyroscope'] = []
            
            left_hand_info['acceleration'].append(get_accel_left(input_report_l, sample_idx=i))
            left_hand_info['gyroscope'].append(get_gyro_left(input_report_l, sample_idx=i))
            right_hand_info['acceleration'].append(get_accel_right(input_report_r, sample_idx=i))
            right_hand_info['gyroscope'].append(get_gyro_right(input_report_r, sample_idx=i))

            if received_tracking_data_left is not None:
                acc = left_hand_info['acceleration'][-1]
                gyro = left_hand_info['gyroscope'][-1]
                u = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]])
                if received_tracking_data_left['isTrackedLeft']:
                    w_pos = left_hand_info['past_wrist_position'][-1]
                    w_rot = left_hand_info['past_wrist_rotation'][-1]
                    z = np.array([w_pos[0], w_pos[1], w_pos[2], w_rot[0], w_rot[1], w_rot[2], w_rot[3]])
                    wrist_pos_l, wrist_rot_l = raw_l.forward(z=z, u=u)
                else:
                    wrist_pos_l, wrist_rot_l = raw_l.forward(u=u)
                
                latest_tracking_data_left['wrist_position_l'] = {
                    'x': float(wrist_pos_l[0]),
                    'y': float(wrist_pos_l[1]),
                    'z': float(wrist_pos_l[2])
                }
                latest_tracking_data_left['wrist_rotation_l'] = {
                    'w': float(wrist_rot_l[0]),
                    'x': float(wrist_rot_l[1]),
                    'y': float(wrist_rot_l[2]),
                    'z': float(wrist_rot_l[3])
                }

            if received_tracking_data_right is not None:
                acc = right_hand_info['acceleration'][-1]
                gyro = right_hand_info['gyroscope'][-1]
                u = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]])
                if received_tracking_data_right['isTrackedRight']:
                    w_pos = right_hand_info['past_wrist_position'][-1]
                    w_rot = right_hand_info['past_wrist_rotation'][-1]
                    z = np.array([w_pos[0], w_pos[1], w_pos[2], w_rot[0], w_rot[1], w_rot[2], w_rot[3]])
                    wrist_pos_r, wrist_rot_r = raw_r.forward(z=z, u=u)
                else:
                    wrist_pos_r, wrist_rot_r = raw_r.forward(u=u)
                
                latest_tracking_data_right['wrist_position_r'] = {
                    'x': float(wrist_pos_r[0]),
                    'y': float(wrist_pos_r[1]),
                    'z': float(wrist_pos_r[2])
                }
                latest_tracking_data_right['wrist_rotation_r'] = {
                    'w': float(wrist_rot_r[0]),
                    'x': float(wrist_rot_r[1]),
                    'y': float(wrist_rot_r[2]),
                    'z': float(wrist_rot_r[3])
                }

            if latest_tracking_data_left['wrist_position_l'] is not None:
                if len(left_hand_info['past_wrist_position']) > 0:
                    if is_tapped(left_hand_info['acceleration']):
                        if not latest_tap_data['is_tapped_l']:
                            tapped_part = predict_part_of_body_left()
                            latest_tap_data['is_tapped_l'] = True
                            latest_tap_data['tapped_part_l'] = tapped_part
                    else:
                        latest_tap_data['is_tapped_l'] = False
                        latest_tap_data['tapped_part_l'] = None
            
            if latest_tracking_data_right['wrist_position_r'] is not None:
                if len(right_hand_info['past_wrist_position']) > 0:
                    if is_tapped(right_hand_info['acceleration']):
                        if not latest_tap_data['is_tapped_r']:
                            tapped_part = predict_part_of_body_right()
                            latest_tap_data['is_tapped_r'] = True
                            latest_tap_data['tapped_part_r'] = tapped_part
                    else:
                        latest_tap_data['is_tapped_r'] = False
                        latest_tap_data['tapped_part_r'] = None

            if latest_tap_data['is_tapped_l']:
                if not tapped_flag_l:
                    tapped_flag_l = True
                    message = json.dumps(latest_tap_data, ensure_ascii=False, indent=2)
                    sock.sendto(message.encode('utf-8'), (UDP_HOST, UDP_PORT_TAP))
                    print(f'Left Hand: tapped({latest_tap_data["tapped_part_l"]}) {datetime.now().strftime("%H:%M:%S")}')
            else:
                if tapped_flag_l:
                    tapped_flag_l = False

            if latest_tap_data['is_tapped_r']:
                if not tapped_flag_r:
                    tapped_flag_r = True
                    message = json.dumps(latest_tap_data, ensure_ascii=False, indent=2)
                    sock.sendto(message.encode('utf-8'), (UDP_HOST, UDP_PORT_TAP))
                    print(f'Right Hand: tapped({latest_tap_data["tapped_part_r"]}) {datetime.now().strftime("%H:%M:%S")}')
            else:
                if tapped_flag_r:
                    tapped_flag_r = False
    sock.close()
    print("closing socket")

def predict_data_left():
    global udp_server_running
    global received_tracking_data_left
    global latest_tracking_data_left

    global left_hand_info

    # model = PoseEstimationNetwork()
    # model.load_state_dict(torch.load('./model/model_left.pth'))

    # start_wrist_position = np.zeros(3)
    # wrist_position = np.zeros(3)

    # model.eval()
    # raw = RawDataForPoseEstimation()

    while udp_server_running:
        if received_tracking_data_left is not None:
            pass
            # if not received_tracking_data_left['isTrackedLeft']:
            #     x, y = make_input(left_hand_info, 'left')
            #     if x is None or y is None:
            #         continue
            #     wrist_pos_l, wrist_rot_l = raw.forward(x, y)
            #     if np.all(start_wrist_position == np.zeros(3)):
            #         start_wrist_position = np.array([float(wrist_pos_l[0][0]), float(wrist_pos_l[0][1]), float(wrist_pos_l[0][2])])
            #     wrist_position = np.array([float(wrist_pos_l[0][0]), float(wrist_pos_l[0][1]), float(wrist_pos_l[0][2])])
            #     correct_wrist_position = np.array(left_hand_info['past_wrist_position'][-1]) + (wrist_position - start_wrist_position) * 0.25
            #     wrist_rot_l = rotation_matrix_to_quaternion(sixd_to_rotation_matrix(wrist_rot_l[0]))
            #     latest_tracking_data_left['wrist_position_l'] = {
            #         'x': correct_wrist_position[0],
            #         'y': correct_wrist_position[1],
            #         'z': correct_wrist_position[2]
            #     }
            #     latest_tracking_data_left['wrist_rotation_l'] = {
            #         'x': float(wrist_rot_l[0]),
            #         'y': float(wrist_rot_l[1]),
            #         'z': float(wrist_rot_l[2]),
            #         'w': float(wrist_rot_l[3])
            #     }
            #     # print(f'Left:{correct_wrist_position}')
            # else:
            #     start_wrist_position = np.array(left_hand_info['past_wrist_position'][-1])

def predict_data_right():
    global udp_server_running
    global received_tracking_data_right
    global latest_tracking_data_right

    global right_hand_info

    # model = PoseEstimationNetwork()
    # model.load_state_dict(torch.load('./model/model_right.pth'))

    # start_wrist_position = np.zeros(3)
    # wrist_position = np.zeros(3)

    # model.eval()

    while udp_server_running:
        if received_tracking_data_right is not None:
            pass
            # if not received_tracking_data_right['isTrackedRight']:
            #     x, y = make_input(right_hand_info, 'right')
            #     if x is None or y is None:
            #         continue
            #     wrist_pos_r, wrist_rot_r = model(x, y)
            #     if np.all(start_wrist_position == np.zeros(3)):
            #         start_wrist_position = np.array([float(wrist_pos_r[0][0]), float(wrist_pos_r[0][1]), float(wrist_pos_r[0][2])])
            #     wrist_position = np.array([float(wrist_pos_r[0][0]), float(wrist_pos_r[0][1]), float(wrist_pos_r[0][2])])
            #     correct_wrist_position = np.array(left_hand_info['past_wrist_position'][-1])
            #     wrist_rot_r = rotation_matrix_to_quaternion(sixd_to_rotation_matrix(wrist_rot_r[0]))
            #     latest_tracking_data_right['wrist_position_r'] = {
            #         'x': correct_wrist_position[0],
            #         'y': correct_wrist_position[1],
            #         'z': correct_wrist_position[2]
            #     }
            #     latest_tracking_data_right['wrist_rotation_r'] = {
            #         'x': float(wrist_rot_r[0]),
            #         'y': float(wrist_rot_r[1]),
            #         'z': float(wrist_rot_r[2]),
            #         'w': float(wrist_rot_r[3])
            #     }
                # print(f'Right:{correct_wrist_position}')

def save_experiment_log():
    global experiment_log

    with open(experiment_log['filename'], 'w', encoding='utf-8') as f:
        json.dump(experiment_log, f, indent=4)
    
    print(f'saved:{experiment_log["filename"]}')

class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global udp_server_running
        global left_hand_info
        global right_hand_info
        if self.path == '/create_session':
            if not udp_server_running:
                udp_server_running = True
                threading.Thread(target=udp_server_left).start()
                threading.Thread(target=udp_server_right).start()
                threading.Thread(target=get_data_from_imu).start()
                threading.Thread(target=predict_data_left).start()
                threading.Thread(target=predict_data_right).start()
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "success", "message": "UDP server started"}')
            else:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"status": "error", "message": "UDP server is already running"}')
        elif self.path == '/stop_server':
            udp_server_running = False
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status": "success", "message": "Server stopping"}')

            save_experiment_log()
            sys.exit(0)

            # global httpd
            # httpd.shutdown()
            # print("HTTP server stopped")
            # sys.exit(0)
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status": "success", "message": "UDP server started"}')
        return

# def run():
# global httpd
PORT = 8080
handler = RequestHandler
httpd = socketserver.TCPServer(("", PORT), handler)
print(f"HTTP server serving at port {PORT}")
httpd.serve_forever()

# if __name__ == "__main__":
#     run()