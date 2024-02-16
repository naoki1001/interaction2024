import argparse
import json
import sys
import os
import time
sys.path.append('../')

from joycon_manage import *
from utils import *
from model.ekf import ExtendedKalmanFilterPyTorch

import torch
import torch.nn as nn

acc_history_l = []
gyro_history_l = []
ekf_history_l = []

acc_history_r = []
gyro_history_r = []
ekf_history_r = []

parts_of_body = ['head', 'shoulder', 'back', 'hip'] 

parser = argparse.ArgumentParser(description='')
parser.add_argument('--hand', help='If you collect the left hand data, set "left", else "right".', choices=['left', 'right'], required=True)
parser.add_argument('--part', help='Input the part of body to collect dataset.', choices=parts_of_body, required=True)
hand = parser.parse_args().hand
part = parser.parse_args().part

def is_tapped(sensor_history):
    try:
        if np.sqrt(np.sum(np.array(sensor_history[-1]) ** 2)) > 30.0:
            return True
        return False
    except Exception as e:
        raise e

def main():
    # joycon settings
    joycon_l, joycon_r = joycon_activate()

    print('start!')

    dt = 0.005
    ekf_l = ExtendedKalmanFilterPyTorch()
    ekf_r = ExtendedKalmanFilterPyTorch()

    while True:
        input_report_l = joycon_l.read(49)
        input_report_r = joycon_r.read(49)

        for i in range(3):
            acc = get_accel_left(input_report_l, sample_idx=i)
            gyro = get_gyro(input_report_l, sample_idx=i)
            # # calc_zとcalc_uを適用してzとuを計算
            z = calc_z(acc)
            u = calc_u(gyro, dt)

            x = ekf_l(z=z, u=u)
            while torch.any(torch.isnan(x)):
                x = ekf_l(z=z, u=u)
                time.sleep(0.1)
            print(x)

if __name__ == '__main__':
    main()