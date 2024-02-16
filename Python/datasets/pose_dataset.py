import json

import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset['data'])
    
    def __getitem__(self, index):
        data = self.dataset['data'][index]
        wrist_pos = torch.tensor(data['past_wrist_position'])
        wrist_rot = torch.tensor(data['past_wrist_rotation'])
        past_head_pos = torch.tensor(data['past_head_position'])
        past_head_rot = torch.tensor(data['past_head_rotation'])
        acc = torch.tensor(data['acc'])
        gyro = torch.tensor(data['gyro'])
        head_pos = torch.tensor(data['head_position'])
        head_rot = torch.tensor(data['head_rotation'])
        ekf_output = torch.tensor(data['kalman_filter_output'])
        pos = torch.tensor(data['wrist_position'])
        rot = torch.tensor(data['wrist_rotation'])
        x = torch.cat([wrist_pos, wrist_rot, past_head_pos, past_head_rot], dim = 1)
        y = torch.transpose(torch.cat([acc, gyro, head_pos, head_rot, ekf_output], dim = 1), 0, 1)
        return x, y, pos, rot

class PoseDatasetPerSensorSize(Dataset):
    def __init__(self, dataset_path, sensor_size=8):
        self.dataset = {'data':[]}
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        for data in dataset['data']:
            if len(data['acc']) == sensor_size:
                self.dataset['data'].append(data)

    def __len__(self):
        return len(self.dataset['data'])

    def __getitem__(self, index):
        data = self.dataset['data'][index]
        wrist_pos = torch.tensor(data['past_wrist_position'])
        wrist_rot = torch.tensor(data['past_wrist_rotation'])
        past_head_pos = torch.tensor(data['past_head_position'])
        past_head_rot = torch.tensor(data['past_head_rotation'])
        acc = torch.tensor(data['acc'])
        gyro = torch.tensor(data['gyro'])
        head_pos = torch.tensor(data['head_position'])
        head_rot = torch.tensor(data['head_rotation'])
        ekf_output = torch.tensor(data['kalman_filter_output'])
        pos = torch.tensor(data['wrist_position'])
        rot = torch.tensor(data['wrist_rotation'])
        x = torch.cat([wrist_pos, wrist_rot, past_head_pos, past_head_rot], dim = 1)
        y = torch.transpose(torch.cat([acc, gyro, head_pos, head_rot, ekf_output], dim = 1), 0, 1)
        return x, y, pos, rot