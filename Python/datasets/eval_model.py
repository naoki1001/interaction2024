import torch
import torch.nn as nn

import argparse

def get_randam_data(hand='left', sensor_size=32, person_num=100, data_size=1000):
    height_table = (torch.randn(person_num) * 6.9 + 167.7) / 100
    

def eval_model(hand='left'):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--hand', help='If you collect the left hand data, set "left", else "right".', choices=['left', 'right'], required=True)
    hand = parser.parse_args().hand
    get_randam_data(hand)
    eval_model(hand)