import http.server
import socketserver
import threading
import socket
import argparse
import json
import sys
import os
sys.path.append('../')

from joycon_manage import *

import torch
import torch.nn as nn

# Global variables
UDP_HOST = '127.0.0.1'
UDP_PORT = 8081
udp_server_running = False
httpd = None
parser = argparse.ArgumentParser(description='')
parser.add_argument('--hand', help='If you collect the left hand data, set "left", else "right".', choices=['left', 'right'], required=True)
parser.add_argument('-i', '--input', help='If you define the filename for dataset, add this option', required=False)
parser.add_argument('-o', '--output', help='If you define the filename for dataset, add this option', required=False)
hand = parser.parse_args().hand
input_file = parser.parse_args().input
output_file = parser.parse_args().output

received_tracking_data = None

data_info = {
    'head_position':[],
    'head_rotation':[],
    'wrist_position':[],
    'wrist_rotation':[],
    'wrist_accel':[],
    'wrist_gyro':[],
    'imu_start_num':[],
    'head_start_num':[]
}

data_out_info = {
    'past_head_position':[],
    'past_head_rotation':[],
    'past_wrist_position':[],
    'past_wrist_rotation':[],
    'wrist_accel':[],
    'wrist_gyro':[],
    'head_position':[],
    'head_rotation':[],
    'tracker_position':[],
    'tracker_rotation':[],
}

def udp_server():
    global udp_server_running
    global received_tracking_data
    global data_info
    global data_out_info

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((UDP_HOST, UDP_PORT))
    print(f"UDP server listening on {UDP_HOST}:{UDP_PORT}")

    past_wrist_position = []
    past_wrist_rotation = []
    past_head_position = []
    past_head_rotation = []

    while udp_server_running:
        data, addr = server_socket.recvfrom(2048)
        received_tracking_data = json.loads(data.decode('utf-8'))

        head_pos = [
            received_tracking_data['headPosition']['x'],
            received_tracking_data['headPosition']['y'],
            received_tracking_data['headPosition']['z']
        ]

        head_rot = [
            received_tracking_data['headRotation']['w'],
            received_tracking_data['headRotation']['x'],
            received_tracking_data['headRotation']['y'],
            received_tracking_data['headRotation']['z']
        ]
        
        data_info['head_position'].append(head_pos)
        data_info['head_rotation'].append(head_rot)
            
        if received_tracking_data['isTracked']:
            wrist_pos = [
                received_tracking_data['handPosition']['x'],
                received_tracking_data['handPosition']['y'],
                received_tracking_data['handPosition']['z']
            ]
            wrist_rot = [
                received_tracking_data['handRotation']['w'],
                received_tracking_data['handRotation']['x'],
                received_tracking_data['handRotation']['y'],
                received_tracking_data['handRotation']['z']
            ]

            if len(past_head_position) < 5:
                past_head_position.append(head_pos)
            else:
                past_head_position[0:-1] = past_head_position[1:]
                past_head_position[-1] = head_pos

            if len(past_head_rotation) < 5:
                past_head_rotation.append(head_rot)
            else:
                past_head_rotation[0:-1] = past_head_rotation[1:]
                past_head_rotation[-1] = head_rot

            if len(past_wrist_position) < 5:
                past_wrist_position.append(wrist_pos)
            else:
                past_wrist_position[0:-1] = past_wrist_position[1:]
                past_wrist_position[-1] = wrist_pos
            
            if len(past_wrist_rotation) < 5:
                past_wrist_rotation.append(wrist_rot)
            else:
                past_wrist_rotation[0:-1] = past_wrist_rotation[1:]
                past_wrist_rotation[-1] = wrist_rot
            
            data_info['wrist_position'].append(wrist_pos)
            data_info['wrist_rotation'].append(wrist_rot)

            if len(data_info['wrist_position']) >= 5:
                data_info['imu_start_num'].append(len(data_info['wrist_accel']) - 1)
                data_info['head_start_num'].append(len(data_info['head_position']) - 1)
        else:
            wrist_pos = [
                received_tracking_data['trackerPosition']['x'],
                received_tracking_data['trackerPosition']['y'],
                received_tracking_data['trackerPosition']['z']
            ]
            wrist_rot = [
                received_tracking_data['trackerRotation']['w'],
                received_tracking_data['trackerRotation']['x'],
                received_tracking_data['trackerRotation']['y'],
                received_tracking_data['trackerRotation']['z']
            ]
            if len(past_head_position) >= 5:
                data_out_info['past_head_position'].append(past_head_position)
                data_out_info['past_head_rotation'].append(past_head_rotation)
                data_out_info['past_wrist_position'].append(past_wrist_position)
                data_out_info['past_wrist_rotation'].append(past_wrist_rotation)
                data_out_info['wrist_accel'].append(data_info['wrist_accel'][data_info['imu_start_num'][-1]:])
                data_out_info['wrist_gyro'].append(data_info['wrist_gyro'][data_info['imu_start_num'][-1]:])
                data_out_info['head_position'].append(data_info['head_position'][data_info['head_start_num'][-1]:])
                data_out_info['head_rotation'].append(data_info['head_rotation'][data_info['head_start_num'][-1]:])
                data_out_info['tracker_position'].append(wrist_pos)
                data_out_info['tracker_rotation'].append(wrist_rot)

    server_socket.close()
    print("UDP server stopped")

def get_data_from_imu():
    global udp_server_running
    global data_info
    global data_out_info
    global hand

    # joycon settings
    joycon_l, joycon_r = joycon_activate()
    while udp_server_running:
        if hand == 'left':
            input_report = joycon_l.read(49)
            input_report = joycon_r.read(49)
            for i in range(3):
                data_info['wrist_accel'].append(get_accel_left(input_report, sample_idx=i))
                data_info['wrist_gyro'].append(get_gyro(input_report, sample_idx=i))
        else:
            input_report = joycon_r.read(49)
            for i in range(3):
                data_info['wrist_accel'].append(get_accel_right(input_report, sample_idx=i))
                data_info['wrist_gyro'].append(get_gyro(input_report, sample_idx=i))

def create_dataset():
    global data_info
    global data_out_info
    global hand, input_file, output_file

    with open('./data_info.json', 'w', encoding='utf-8') as f_info:
        json.dump(data_info, f_info, indent=4)

    with open('./data_out_info.json', 'w', encoding='utf-8') as f_out_info:
        json.dump(data_out_info, f_out_info, indent=4)

    upscale_pos = nn.Upsample(size=(1, 8, 3), mode='trilinear', align_corners=True)
    upscale_rot = nn.Upsample(size=(1, 8, 4), mode='trilinear', align_corners=True)

    if input_file:
        try:
            with open(input_file, 'r+', encoding='utf-8') as data_file:
                dataset = json.load(data_file)
        except Exception as e:
            print(e)
            dataset = {
                'data':[]
            }
    else:
        dataset = {
            'data':[]
        }
    
    if output_file:
        if os.path.exists(output_file):
            data_file = open(output_file, 'r+', encoding='utf-8')
            dataset = json.load(data_file)
        else:
            data_file = open(output_file, 'w', encoding='utf-8')
    else:
        data_file = open(f'dataset_{hand}_new.json', 'w', encoding='utf-8')

    # data for inside of view
    for i in range(len(data_info['wrist_position']) - 8):
        for j in range(1, 4):
            if data_info['imu_start_num'][i+j] - data_info['imu_start_num'][i] == 0:
                continue
            data = {}
            data['past_wrist_position'] = data_info['wrist_position'][i:i+5]
            data['past_wrist_rotation'] = data_info['wrist_rotation'][i:i+5]
            data['past_head_position'] = data_info['head_position'][i:i+5]
            data['past_head_rotation'] = data_info['head_rotation'][i:i+5]
            if data_info['imu_start_num'][i+j] - data_info['imu_start_num'][i] < 8:
                data['acc'] = upscale_pos(torch.tensor(
                    data_info['wrist_accel'][data_info['imu_start_num'][i]:data_info['imu_start_num'][i+j]]
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
                data['gyro'] = upscale_pos(torch.tensor(
                    data_info['wrist_gyro'][data_info['imu_start_num'][i]:data_info['imu_start_num'][i+j]]
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
                data['head_position'] = upscale_pos(torch.tensor(
                    data_info['head_position'][data_info['head_start_num'][i]:data_info['head_start_num'][i+j]]
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
                data['head_rotation'] = upscale_rot(torch.tensor(
                    data_info['head_rotation'][data_info['head_start_num'][i]:data_info['head_start_num'][i+j]]
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
            else:
                upscale_position = nn.Upsample(size=(1, data_info['imu_start_num'][i+j] - data_info['imu_start_num'][i], 3), mode='trilinear', align_corners=True)
                upscale_rotation = nn.Upsample(size=(1, data_info['imu_start_num'][i+j] - data_info['imu_start_num'][i], 4), mode='trilinear', align_corners=True)
                data['acc'] = data_info['wrist_accel'][data_info['imu_start_num'][i]:data_info['imu_start_num'][i+j]]
                data['gyro'] = data_info['wrist_gyro'][data_info['imu_start_num'][i]:data_info['imu_start_num'][i+j]]
                data['head_position'] = upscale_position(torch.tensor(
                    data_info['head_position'][data_info['head_start_num'][i]:data_info['head_start_num'][i+j]]
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
                data['head_rotation'] = upscale_rotation(torch.tensor(
                    data_info['head_rotation'][data_info['head_start_num'][i]:data_info['head_start_num'][i+j]]
                ).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
            data['wrist_position'] = data_info['wrist_position'][i+j+4]
            data['wrist_rotation'] = data_info['wrist_rotation'][i+j+4]
            dataset['data'].append(data)
    
    # data for out side of view
    for i in range(len(data_out_info['tracker_position'])):
        data = {}
        data['past_wrist_position'] = data_out_info['past_wrist_position'][i]
        data['past_wrist_rotation'] = data_out_info['past_wrist_rotation'][i]
        data['past_head_position'] = data_out_info['past_head_position'][i]
        data['past_head_rotation'] = data_out_info['past_head_rotation'][i]
        if len(data_out_info['wrist_accel'][i]) < 8:
            data['acc'] = upscale_pos(torch.tensor(data_out_info['wrist_accel'][i]).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
            data['gyro'] = upscale_pos(torch.tensor(data_out_info['wrist_gyro'][i]).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
            data['head_position'] = upscale_pos(torch.tensor(data_out_info['head_position'][i]).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
            data['head_rotation'] = upscale_rot(torch.tensor(data_out_info['head_rotation'][i]).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
        else:
            upscale_position = nn.Upsample(size=(1, len(data_out_info['wrist_accel'][i]), 3), mode='trilinear', align_corners=True)
            upscale_rotation = nn.Upsample(size=(1, len(data_out_info['wrist_accel'][i]), 4), mode='trilinear', align_corners=True)
            data['acc'] = data_out_info['wrist_accel'][i]
            data['gyro'] = data_out_info['wrist_gyro'][i]
            data['head_position'] = upscale_position(torch.tensor(data_out_info['head_position'][i]).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
            data['head_rotation'] = upscale_rotation(torch.tensor(data_out_info['head_rotation'][i]).unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).squeeze(0).tolist()
        data['wrist_position'] = data_out_info['tracker_position'][i]
        data['wrist_rotation'] = data_out_info['tracker_rotation'][i]
        dataset['data'].append(data)
    
    json.dump(dataset, data_file, indent=4)
    data_file.close()


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global udp_server_running
        if self.path == '/create_session':
            if not udp_server_running:
                udp_server_running = True
                threading.Thread(target=udp_server).start()
                threading.Thread(target=get_data_from_imu).start()
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
            create_dataset()
            print('data saved.')
        else:
            self.send_response(404)
            self.end_headers()

PORT = 8080
handler = RequestHandler
httpd = socketserver.TCPServer(("", PORT), handler)
print(f"HTTP server serving at port {PORT}")
httpd.serve_forever()