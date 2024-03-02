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
latest_predicted_data = {
    'leftHandPosition': None,
    'leftHandRotation': None,
    'rightHandPosition': None,
    'rightHandRotation': None
}

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

def udp_server():
    global udp_server_running
    global latest_tracking_data
    global latest_predicted_data
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
        data = {
            'wrist_position_l': latest_predicted_data['leftHandPosition'],
            'wrist_rotation_l': latest_predicted_data['leftHandRotation'],
            'wrist_position_r': latest_predicted_data['rightHandPosition'],
            'wrist_rotation_r': latest_predicted_data['rightHandRotation']
        }
        response = json.dumps(data, ensure_ascii=False, indent=2)
        server_socket.sendto(response.encode('utf-8'), addr)

    server_socket.close()
    print("UDP server stopped")


def update_tracking():
    global udp_server_running
    global latest_tracking_data
    global latest_predicted_data

    # joycon settings
    joycon_l, joycon_r = joycon_activate()
    left_hand_acc = []
    left_hand_gyro = []
    right_hand_acc = []
    right_hand_gyro = []

    model_left = RawDataForPoseEstimation()
    model_right = RawDataForPoseEstimation()

    while udp_server_running:
        input_report_l = joycon_l.read(49)
        input_report_r = joycon_r.read(49)
        for i in range(3):
            left_hand_acc.append(get_accel_left(input_report_l, sample_idx=i))
            left_hand_gyro.append(get_gyro_left(input_report_r, sample_idx=i))
            right_hand_acc.append(get_accel_right(input_report_l, sample_idx=i))
            right_hand_gyro.append(get_gyro_right(input_report_r, sample_idx=i))

            if latest_tracking_data is not None:
                u = np.array([
                    left_hand_acc[-1][0], left_hand_acc[-1][1], left_hand_acc[-1][2], 
                    left_hand_gyro[-1][0], left_hand_gyro[-1][1], left_hand_gyro[-1][2]
                ])
                if latest_tracking_data['isTrackedLeft']:
                    z = np.array([
                        latest_tracking_data['leftHandPosition']['x'],
                        latest_tracking_data['leftHandPosition']['y'],
                        latest_tracking_data['leftHandPosition']['z'],
                        latest_tracking_data['leftHandRotation']['w'],
                        latest_tracking_data['leftHandRotation']['x'],
                        latest_tracking_data['leftHandRotation']['y'],
                        latest_tracking_data['leftHandRotation']['z']
                    ])
                    wrist_pos_l, wrist_rot_l = model_left.forward(z=z, u=u)
                else:
                    wrist_pos_l, wrist_rot_l = model_left.forward(u=u)

                latest_predicted_data['leftHandPosition'] = {
                    'x': wrist_pos_l[0],
                    'y': wrist_pos_l[1],
                    'z': wrist_pos_l[2]
                }
                latest_predicted_data['leftHandRotation'] = {
                    'w': wrist_rot_l[0],
                    'x': wrist_rot_l[1],
                    'y': wrist_rot_l[2],
                    'z': wrist_rot_l[3]
                }

                u = np.array([
                    right_hand_acc[-1][0], right_hand_acc[-1][1], right_hand_acc[-1][2], 
                    right_hand_gyro[-1][0], right_hand_gyro[-1][1], right_hand_gyro[-1][2]
                ])
                if latest_tracking_data['isTrackedRight']:
                    z = np.array([
                    latest_tracking_data['rightHandPosition']['x'],
                    latest_tracking_data['rightHandPosition']['y'],
                    latest_tracking_data['rightHandPosition']['z'],
                    latest_tracking_data['rightHandRotation']['w'],
                    latest_tracking_data['rightHandRotation']['x'],
                    latest_tracking_data['rightHandRotation']['y'],
                    latest_tracking_data['rightHandRotation']['z']
                ])
                    wrist_pos_r, wrist_rot_r = model_right.forward(z=z, u=u)
                else:
                    wrist_pos_r, wrist_rot_r = model_right.forward(u=u)

                latest_predicted_data['rightHandPosition'] = {
                    'x': wrist_pos_r[0],
                    'y': wrist_pos_r[1],
                    'z': wrist_pos_r[2]
                }
                latest_predicted_data['rightHandRotation'] = {
                    'w': wrist_rot_r[0],
                    'x': wrist_rot_r[1],
                    'y': wrist_rot_r[2],
                    'z': wrist_rot_r[3]
                }

    print("closing socket")


if __name__ == "__main__":
    udp_server_running = True
    thread_hand = threading.Thread(target=udp_server)
    thread_tracking = threading.Thread(target=update_tracking)
    thread_hand.start()
    thread_tracking.start()