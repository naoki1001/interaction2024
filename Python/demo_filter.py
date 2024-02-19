import threading
import json
from datetime import datetime
import sys
sys.path.append('./model/')

from utils import *
from joycon_manage import *
from model.filters import *

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

history_p_l = []
history_v_l = []
history_q_l = []
history_p_r = []
history_v_r = []
history_q_r = []
fig1_l, axis1_l = plt.subplots()
fig2_l, axis2_l = plt.subplots()
fig3_l, axis3_l = plt.subplots()
fig1_r, axis1_r = plt.subplots()
fig2_r, axis2_r = plt.subplots()
fig3_r, axis3_r = plt.subplots()
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
labels = ['x', 'y', 'z', 'w', 'x', 'y', 'z']

def main():
    sample_n = 20

    # joycon settings
    joycon_l, joycon_r = joycon_activate()
    
    raw = RawDataForPoseEstimation()

    while True:
        input_report_l = joycon_l.read(49)
        input_report_r = joycon_r.read(49)
        for i in range(3):
            acc = get_accel_left(input_report_l, sample_idx=i)
            gyro = get_gyro_left(input_report_l, sample_idx=i)
            u = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]])
            wrist_pos_l, wrist_vel_l, wrist_rot_l = raw.get_next_state(u=u)
            history_p_l.append(wrist_pos_l)
            history_v_l.append(wrist_vel_l)
            history_q_l.append(wrist_rot_l)
            
            acc = get_accel_right(input_report_r, sample_idx=i)
            gyro = get_gyro_right(input_report_r, sample_idx=i)
            u = np.array([acc[0], acc[1], acc[2], gyro[0], gyro[1], gyro[2]])
            wrist_pos_r, wrist_vel_r, wrist_rot_r = raw.get_next_state(u=u)
            history_p_r.append(wrist_pos_r)
            history_v_r.append(wrist_vel_r)
            history_q_r.append(wrist_rot_r)
        
        if len(history_p_l) > sample_n:
            p_l = np.array(history_p_l)[-1 - sample_n:-1, :]
            v_l = np.array(history_v_l)[-1 - sample_n:-1, :]
            q_l = np.array(history_q_l)[-1 - sample_n:-1, :]
            p_r = np.array(history_p_r)[-1 - sample_n:-1, :]
            v_r = np.array(history_v_r)[-1 - sample_n:-1, :]
            q_r = np.array(history_q_r)[-1 - sample_n:-1, :]

            p_x_l, = axis1_l.plot(np.arange(sample_n), p_l[:, 0], color=colors[0], linestyle='-', label=labels[0])
            p_y_l, = axis1_l.plot(np.arange(sample_n), p_l[:, 1], color=colors[1], linestyle='-', label=labels[1])
            p_z_l, = axis1_l.plot(np.arange(sample_n), p_l[:, 2], color=colors[2], linestyle='-', label=labels[2])
            v_x_l, = axis2_l.plot(np.arange(sample_n), v_l[:, 0], color=colors[0], linestyle='-', label=labels[0])
            v_y_l, = axis2_l.plot(np.arange(sample_n), v_l[:, 1], color=colors[1], linestyle='-', label=labels[1])
            v_z_l, = axis2_l.plot(np.arange(sample_n), v_l[:, 2], color=colors[2], linestyle='-', label=labels[2])
            q_w_l, = axis3_l.plot(np.arange(sample_n), q_l[:, 0], color=colors[3], linestyle='-', label=labels[3])
            q_x_l, = axis3_l.plot(np.arange(sample_n), q_l[:, 1], color=colors[4], linestyle='-', label=labels[4])
            q_y_l, = axis3_l.plot(np.arange(sample_n), q_l[:, 2], color=colors[5], linestyle='-', label=labels[5])
            q_z_l, = axis3_l.plot(np.arange(sample_n), q_l[:, 3], color=colors[6], linestyle='-', label=labels[6])
            p_x_r, = axis1_r.plot(np.arange(sample_n), p_r[:, 0], color=colors[0], linestyle='-', label=labels[0])
            p_y_r, = axis1_r.plot(np.arange(sample_n), p_r[:, 1], color=colors[1], linestyle='-', label=labels[1])
            p_z_r, = axis1_r.plot(np.arange(sample_n), p_r[:, 2], color=colors[2], linestyle='-', label=labels[2])
            v_x_r, = axis2_r.plot(np.arange(sample_n), v_r[:, 0], color=colors[0], linestyle='-', label=labels[0])
            v_y_r, = axis2_r.plot(np.arange(sample_n), v_r[:, 1], color=colors[1], linestyle='-', label=labels[1])
            v_z_r, = axis2_r.plot(np.arange(sample_n), v_r[:, 2], color=colors[2], linestyle='-', label=labels[2])
            q_w_r, = axis3_r.plot(np.arange(sample_n), q_r[:, 0], color=colors[3], linestyle='-', label=labels[3])
            q_x_r, = axis3_r.plot(np.arange(sample_n), q_r[:, 1], color=colors[4], linestyle='-', label=labels[4])
            q_y_r, = axis3_r.plot(np.arange(sample_n), q_r[:, 2], color=colors[5], linestyle='-', label=labels[5])
            q_z_r, = axis3_r.plot(np.arange(sample_n), q_r[:, 3], color=colors[6], linestyle='-', label=labels[6])

            plt.pause(0.0001)

            p_x_l.remove()
            p_y_l.remove()
            p_z_l.remove()
            v_x_l.remove()
            v_y_l.remove()
            v_z_l.remove()
            q_w_l.remove()
            q_x_l.remove()
            q_y_l.remove()
            q_z_l.remove()
            p_x_r.remove()
            p_y_r.remove()
            p_z_r.remove()
            v_x_r.remove()
            v_y_r.remove()
            v_z_r.remove()
            q_w_r.remove()
            q_x_r.remove()
            q_y_r.remove()
            q_z_r.remove()


if __name__ == '__main__':
    main()