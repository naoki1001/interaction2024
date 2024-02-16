import numpy as np
import hid
import time

VENDOR_ID = 0x057E
L_PRODUCT_ID = 0x2006
R_PRODUCT_ID = 0x2007

L_ACCEL_OFFSET_X = 350
L_ACCEL_OFFSET_Y = 0
L_ACCEL_OFFSET_Z = 0 # 4081 重力?
R_ACCEL_OFFSET_X = 350
R_ACCEL_OFFSET_Y = 0
R_ACCEL_OFFSET_Z = 0 #-4081 重力?

def write_output_report(joycon_device, packet_number, command, subcommand, argument):
    joycon_device.write(command
                        + packet_number.to_bytes(1, byteorder='big')
                        + b'\x00\x01\x40\x40\x00\x01\x40\x40'
                        + subcommand
                        + argument)

def is_left(my_product_id):
    return my_product_id == L_PRODUCT_ID

def to_int16le_from_2bytes(hbytebe, lbytebe):
    uint16le = (lbytebe << 8) | hbytebe 
    int16le = uint16le if uint16le < 32768 else uint16le - 65536
    return int16le

def get_nbit_from_input_report(input_report, offset_byte, offset_bit, nbit):
    return (input_report[offset_byte] >> offset_bit) & ((1 << nbit) - 1)

def get_button_down(input_report):
    return get_nbit_from_input_report(input_report, 5, 0, 1)

def get_button_up(input_report):
    return get_nbit_from_input_report(input_report, 5, 1, 1)

def get_button_right(input_report):
    return get_nbit_from_input_report(input_report, 5, 2, 1)

def get_button_left(input_report):
    return get_nbit_from_input_report(input_report, 5, 3, 1)

def get_stick_left_horizontal(input_report):
    return get_nbit_from_input_report(input_report, 6, 0, 8) | (get_nbit_from_input_report(input_report, 7, 0, 4) << 8)

def get_stick_left_vertical(input_report):
    return get_nbit_from_input_report(input_report, 7, 4, 4) | (get_nbit_from_input_report(input_report, 8, 0, 8) << 4)

def get_stick_right_horizontal(input_report):
    return get_nbit_from_input_report(input_report, 9, 0, 8) | (get_nbit_from_input_report(input_report, 10, 0, 4) << 8)

def get_stick_right_vertical(input_report):
    return get_nbit_from_input_report(input_report, 10, 4, 4) | (get_nbit_from_input_report(input_report, 11, 0, 8) << 4)

def get_accel_x(input_report, product_id, sample_idx=0):
    if sample_idx not in [0, 1, 2]:
        raise IndexError('sample_idx should be between 0 and 2')
    return (to_int16le_from_2bytes(get_nbit_from_input_report(input_report, 13 + sample_idx * 12, 0, 8),
                                   get_nbit_from_input_report(input_report, 14 + sample_idx * 12, 0, 8))
            - (L_ACCEL_OFFSET_X if is_left(product_id) else R_ACCEL_OFFSET_X)) * 0.000244 * 9.80665

def get_accel_y(input_report, product_id, sample_idx=0):
    if sample_idx not in [0, 1, 2]:
        raise IndexError('sample_idx should be between 0 and 2')
    return (to_int16le_from_2bytes(get_nbit_from_input_report(input_report, 15 + sample_idx * 12, 0, 8),
                                   get_nbit_from_input_report(input_report, 16 + sample_idx * 12, 0, 8))
            - (L_ACCEL_OFFSET_Y if is_left(product_id) else R_ACCEL_OFFSET_Y)) * 0.000244 * 9.80665

def get_accel_z(input_report, product_id, sample_idx=0):
    if sample_idx not in [0, 1, 2]:
        raise IndexError('sample_idx should be between 0 and 2')
    return (to_int16le_from_2bytes(get_nbit_from_input_report(input_report, 17 + sample_idx * 12, 0, 8),
                                   get_nbit_from_input_report(input_report, 18 + sample_idx * 12, 0, 8))
            - (L_ACCEL_OFFSET_Z if is_left(product_id) else R_ACCEL_OFFSET_Z)) * 0.000244 * 9.80665

def get_accel_left(input_report, sample_idx=0):
    acc_x = get_accel_x(input_report, L_PRODUCT_ID, sample_idx=sample_idx)
    acc_y = get_accel_y(input_report, L_PRODUCT_ID, sample_idx=sample_idx)
    acc_z = get_accel_z(input_report, L_PRODUCT_ID, sample_idx=sample_idx)
    acc = [acc_x, acc_y, acc_z]
    return acc

def get_accel_right(input_report, sample_idx=0):
    acc_x = get_accel_x(input_report, R_PRODUCT_ID, sample_idx=sample_idx)
    acc_y = get_accel_y(input_report, R_PRODUCT_ID, sample_idx=sample_idx)
    acc_z = get_accel_z(input_report, R_PRODUCT_ID, sample_idx=sample_idx)
    acc = [acc_x, -acc_y, -acc_z]
    return acc

def get_gyro_x(input_report, sample_idx=0):
    if sample_idx not in [0, 1, 2]:
        raise IndexError('sample_idx should be between 0 and 2')
    return to_int16le_from_2bytes(get_nbit_from_input_report(input_report, 19 + sample_idx * 12, 0, 8),
                                  get_nbit_from_input_report(input_report, 20 + sample_idx * 12, 0, 8)) * 0.06103 / 180.0 * np.pi

def get_gyro_y(input_report, sample_idx=0):
    if sample_idx not in [0, 1, 2]:
        raise IndexError('sample_idx should be between 0 and 2')
    return to_int16le_from_2bytes(get_nbit_from_input_report(input_report, 21 + sample_idx * 12, 0, 8),
                                  get_nbit_from_input_report(input_report, 22 + sample_idx * 12, 0, 8)) * 0.06103 / 180.0 * np.pi

def get_gyro_z(input_report, sample_idx=0):
    if sample_idx not in [0, 1, 2]:
        raise IndexError('sample_idx should be between 0 and 2')
    return to_int16le_from_2bytes(get_nbit_from_input_report(input_report, 23 + sample_idx * 12, 0, 8),
                                  get_nbit_from_input_report(input_report, 24 + sample_idx * 12, 0, 8)) * 0.06103 / 180.0 * np.pi

def get_gyro_left(input_report, sample_idx=0):
    gyro_x = get_gyro_x(input_report, sample_idx=sample_idx)
    gyro_y = get_gyro_y(input_report, sample_idx=sample_idx)
    gyro_z = get_gyro_z(input_report, sample_idx=sample_idx)
    return [gyro_x, gyro_y, gyro_z]

def get_gyro_right(input_report, sample_idx=0):
    gyro_x = get_gyro_x(input_report, sample_idx=sample_idx)
    gyro_y = get_gyro_y(input_report, sample_idx=sample_idx)
    gyro_z = get_gyro_z(input_report, sample_idx=sample_idx)
    return [gyro_x, -gyro_y, -gyro_z]

def joycon_activate():
    joycon_l = hid.device()
    joycon_r = hid.device()
    joycon_l.open(VENDOR_ID, L_PRODUCT_ID)
    joycon_r.open(VENDOR_ID, R_PRODUCT_ID)

    # 6軸センサーを有効化
    write_output_report(joycon_l, 0, b'\x01', b'\x40', b'\x01')
    write_output_report(joycon_r, 0, b'\x01', b'\x40', b'\x01')
    # 設定を反映するためには時間間隔が必要
    time.sleep(0.02)
    # 60HzでJoy-Conの各データを取得するための設定
    write_output_report(joycon_l, 1, b'\x01', b'\x03', b'\x30')
    write_output_report(joycon_r, 1, b'\x01', b'\x03', b'\x30')
    return joycon_l, joycon_r