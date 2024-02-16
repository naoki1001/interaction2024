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
from model.filters import *

import numpy as np

# Global variables
udp_server_running = False
UDP_HOST = '127.0.0.1'
UDP_PORT_TAP = 8083

def is_tapped(sensor_history):
    try:
        if np.sqrt(np.sum(np.array(sensor_history[-1]) ** 2)) > 30.0:
            return True
        return False
    except Exception as e:
        raise e

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
    threading.Thread(target=send_tap_data).start()