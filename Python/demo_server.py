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
UDP_PORT = 8081
UDP_PORT_BODY = 8082
UDP_PORT_TAP = 8083

latest_tracking_data = None
received_tracking_data = None

model_left = RawDataForPoseEstimation()
model_right = RawDataForPoseEstimation()

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
    global latest_tracking_data
    if latest_tracking_data is not None:
        try:
            wrist_pos = [
                latest_tracking_data['leftHandPosition']['x'],
                latest_tracking_data['leftHandPosition']['y'],
                latest_tracking_data['leftHandPosition']['z']
            ]
            wrist_rot = [
                latest_tracking_data['leftHandRotation']['w'],
                latest_tracking_data['leftHandRotation']['x'],
                latest_tracking_data['leftHandRotation']['y'],
                latest_tracking_data['leftHandRotation']['z']
            ]
            wrist_position = torch.tensor(wrist_pos)
            rot_matrix = compute_rotation_matrix_from_quaternion(torch.tensor(wrist_rot).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        except:
            return None
        
        wrist_position = apply_offset_of_tap_position(wrist_position, rot_matrix).numpy()
        parts = [part for part in received_tracking_data.keys() if not 'LeftShoulder' in part and not 'RightHip' in part]
        part_positions = np.array([
            [
                received_tracking_data[part]['x'],
                received_tracking_data[part]['y'],
                received_tracking_data[part]['z']
            ] for part in parts if not 'LeftShoulder' in part and not 'RightHip' in part
        ])
        diff_parts = part_positions - wrist_position
        part_distance = np.linalg.norm(diff_parts, axis=1)
        nearest_part_index = np.argmin(part_distance)
        return parts[nearest_part_index]
    else:
        return None

def predict_part_of_body_right():
    global received_tracking_data
    global latest_tracking_data
    if latest_tracking_data is not None:
        try:
            wrist_pos = [
                latest_tracking_data['rightHandPosition']['x'],
                latest_tracking_data['rightHandPosition']['y'],
                latest_tracking_data['rightHandPosition']['z']
            ]
            wrist_rot = [
                latest_tracking_data['rightHandRotation']['w'],
                latest_tracking_data['rightHandRotation']['x'],
                latest_tracking_data['rightHandRotation']['y'],
                latest_tracking_data['rightHandRotation']['z']
            ]
            wrist_position = torch.tensor(wrist_pos)
            rot_matrix = compute_rotation_matrix_from_quaternion(torch.tensor(wrist_rot).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        except:
            return None
        
        wrist_position = apply_offset_of_tap_position(wrist_position, rot_matrix, offset=[[-0.05], [0.0], [0.0]]).numpy()
        parts = [part for part in received_tracking_data.keys() if not 'RightShoulder' in part and not 'LeftHip' in part]
        part_positions = np.array([
            [
                received_tracking_data[part]['x'],
                received_tracking_data[part]['y'],
                received_tracking_data[part]['z']
            ] for part in parts if not 'RightShoulder' in part and not 'LeftHip' in part
        ])
        diff_parts = part_positions - wrist_position
        part_distance = np.linalg.norm(diff_parts, axis=1)
        nearest_part_index = np.argmin(part_distance)
        return parts[nearest_part_index]
    else:
        return None

def udp_server():
    global udp_server_running
    global latest_tracking_data
    global left_hand_info
    global right_hand_info

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((UDP_HOST, UDP_PORT))
    print(f"UDP server listening on {UDP_HOST}:{UDP_PORT}")

    while udp_server_running:
        data, addr = server_socket.recvfrom(2048)
        latest_tracking_data = json.loads(data.decode('utf-8'))

        if latest_tracking_data['headPosition'] is not None:
            head_pos = [
                latest_tracking_data['headPosition']['x'],
                latest_tracking_data['headPosition']['y'],
                latest_tracking_data['headPosition']['z']
            ]

            head_rot = [
                latest_tracking_data['headRotation']['w'],
                latest_tracking_data['headRotation']['x'],
                latest_tracking_data['headRotation']['y'],
                latest_tracking_data['headRotation']['z']
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

        if latest_tracking_data['isTrackedLeft']:
            left_hand_info['head_position'] = []
            left_hand_info['head_rotation'] = []
            wrist_pos_l = [
                latest_tracking_data['leftHandPosition']['x'],
                latest_tracking_data['leftHandPosition']['y'],
                latest_tracking_data['leftHandPosition']['z']
            ]
            wrist_rot_l = [
                latest_tracking_data['leftHandRotation']['w'],
                latest_tracking_data['leftHandRotation']['x'],
                latest_tracking_data['leftHandRotation']['y'],
                latest_tracking_data['leftHandRotation']['z']
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
        else:
            if latest_tracking_data['headPosition'] is not None:
                left_hand_info['head_position'].append(head_pos)
                left_hand_info['head_rotation'].append(head_rot)

        if latest_tracking_data['isTrackedRight']:
            right_hand_info['head_position'] = []
            right_hand_info['head_rotation'] = []
            wrist_pos_r = [
                latest_tracking_data['rightHandPosition']['x'],
                latest_tracking_data['rightHandPosition']['y'],
                latest_tracking_data['rightHandPosition']['z']
            ]
            wrist_rot_r = [
                latest_tracking_data['rightHandRotation']['w'],
                latest_tracking_data['rightHandRotation']['x'],
                latest_tracking_data['rightHandRotation']['y'],
                latest_tracking_data['rightHandRotation']['z']
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
        else:
            if latest_tracking_data['headPosition'] is not None:
                right_hand_info['head_position'].append(head_pos)
                right_hand_info['head_rotation'].append(head_rot)
            
        # if not latest_tracking_data['isTrackedLeft'] or not latest_tracking_data['isTrackedRight']:
        #     data = {
        #         'wrist_position_l': latest_tracking_data['leftHandPosition'],
        #         'wrist_rotation_l': latest_tracking_data['leftHandRotation'],
        #         'wrist_position_r': latest_tracking_data['rightHandPosition'],
        #         'wrist_rotation_r': latest_tracking_data['rightHandRotation']
        #     }
        #     response = json.dumps(data, ensure_ascii=False, indent=2)
        #     server_socket.sendto(response.encode('utf-8'), addr)

    server_socket.close()
    print("UDP server stopped")

def udp_server_body():
    global udp_server_running
    global received_tracking_data
    global latest_tracking_data

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((UDP_HOST, UDP_PORT_BODY))
    print(f"UDP server listening on {UDP_HOST}:{UDP_PORT_BODY}")

    while udp_server_running:
        data, addr = server_socket.recvfrom(2048)
        received_tracking_data = json.loads(data.decode('utf-8'))
        received_tracking_data['Head']['y'] += 0.15

    server_socket.close()
    print("UDP server stopped")

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
            if latest_tracking_data is not None:
                if latest_tracking_data['isTrackedLeft']:
                    left_hand_acc = []
                if latest_tracking_data['isTrackedRight']:
                    right_hand_acc = []

            if len(left_hand_acc) > 10:
                left_hand_acc = []
            if len(right_hand_acc) > 10:
                right_hand_acc = []

            left_hand_acc.append(get_accel_left(input_report_l, sample_idx=i))
            right_hand_acc.append(get_accel_right(input_report_r, sample_idx=i))

            if is_tapped(left_hand_acc):
                if not latest_tap_data['is_tapped_l']:
                    tapped_part = predict_part_of_body_left()
                    latest_tap_data['is_tapped_l'] = True
                    latest_tap_data['tapped_part_l'] = tapped_part
            else:
                latest_tap_data['is_tapped_l'] = False
                latest_tap_data['tapped_part_l'] = None

            if is_tapped(right_hand_acc):
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

if __name__ == "__main__":
    udp_server_running = True
    thread_tapdata = threading.Thread(target=send_tap_data)
    thread_hand = threading.Thread(target=udp_server)
    thread_body = threading.Thread(target=udp_server_body)
    thread_tapdata.start()
    thread_hand.start()
    thread_body.start()