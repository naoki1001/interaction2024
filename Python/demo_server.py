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
from model.pose_estimation import *
from model.filters import *

import numpy as np

# Global variables
udp_server_running = False
UDP_HOST = '127.0.0.1'
UDP_PORT_TAP = 8083

def apply_offset_of_tap_position(pos, rot_matrix, offset=[[0.05], [0.0], [0.0]]):
    return (rot_matrix @ torch.tensor(offset)).T + pos

def is_tapped(sensor_history):
    try:
        if np.sqrt(np.sum(np.array(sensor_history[-1]) ** 2)) > 30.0:
            return True
        return False
    except Exception as e:
        raise e

def predict_part_of_body_left():
    global received_tracking_data
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
        parts = [part for part in received_tracking_data_left.keys()]
        part_positions = np.array([
            [
                received_tracking_data[part]['x'],
                received_tracking_data[part]['y'],
                received_tracking_data[part]['z']
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

def send_tap_data():
    global udp_server_running

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

    left_hand_acc = []
    right_hand_acc = []

    while udp_server_running:
        input_report_l = joycon_l.read(49)
        input_report_r = joycon_r.read(49)
        for i in range(3):
            left_hand_acc.append(get_accel_left(input_report_l, sample_idx=i))
            right_hand_acc.append(get_accel_right(input_report_r, sample_idx=i))

            if is_tapped(left_hand_acc):
                if not latest_tap_data['is_tapped_l']:
                    latest_tap_data['is_tapped_l'] = True
                    latest_tap_data['tapped_part_l'] = None
            else:
                latest_tap_data['is_tapped_l'] = False
                latest_tap_data['tapped_part_l'] = None

            if is_tapped(right_hand_acc):
                if not latest_tap_data['is_tapped_r']:
                    latest_tap_data['is_tapped_r'] = True
                    latest_tap_data['tapped_part_r'] = None
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

if __name__ == "__main__":
    udp_server_running = True
    thread_tapdata = threading.Thread(target=send_tap_data)
    thread_tapdata.start()