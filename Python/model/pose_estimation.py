import torch
import torch.nn as nn

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout: float = 0.1, max_len: int = 5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PoseEstimationNetwork(nn.Module):
    def __init__(self, embed=256, heads=8, init_weights: bool = True):
        super(PoseEstimationNetwork, self).__init__()
        self.rnn = nn.RNN(input_size=18, hidden_size=256, num_layers=2, dropout=0.1)
        self.positional_encoding = PositionalEncoding(d_model=embed)

        self.conv1 = nn.Sequential(
            nn.Conv1d(18, 42, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(42),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(42, 42, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(42),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(42, 84, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(84),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(84, 84, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(84),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(84, 168, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(168),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(168, 168, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(168),
            nn.LeakyReLU(negative_slope=0.2),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )

        self.dense = nn.Sequential(
            nn.Linear(168, 256)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed, nhead=heads, dim_feedforward=2048, dropout=0.1, batch_first=True)
        self.tranformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.wrist_position_decoder_network = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.wrist_rotation_decoder_network = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x, y):
        x_rnn, _ = self.rnn(x, None)
        x_rnn = x_rnn[:, -1, :].unsqueeze(1)

        x_conv = self.conv1(y)
        x_conv = self.conv2(x_conv)
        x_conv = self.conv3(x_conv)
        x_conv = torch.transpose(x_conv, 1, 2)
        x_dense = self.dense(x_conv)
        
        input_transformer = torch.cat((x_rnn, x_dense), dim=1)
        input_transformer = self.positional_encoding(input_transformer)
        output_transformer = self.tranformer_encoder(input_transformer)
        wrist_input = torch.chunk(output_transformer, chunks=output_transformer.size(1), dim=1)[-1].view(-1, 256)
        wrist_position = self.wrist_position_decoder_network(wrist_input)
        wrist_rotation = self.wrist_rotation_decoder_network(wrist_input)
        return wrist_position, wrist_rotation

from filters import *

class PoseEstimator(nn.Module):
    def __init__(self):
        super(PoseEstimator, self).__init__()
        self.network = PoseEstimationNetwork()
        
    def forward(self, x, y):
        x_rnn, _ = self.rnn(x, None)
        x_rnn = x_rnn[:, -1, :].unsqueeze(1)

        x_conv = self.conv1(y)
        x_conv = self.conv2(x_conv)
        x_conv = self.conv3(x_conv)
        x_conv = torch.transpose(x_conv, 1, 2)
        x_dense = self.dense(x_conv)
        
        input_transformer = torch.cat((x_rnn, x_dense), dim=1)
        input_transformer = self.positional_encoding(input_transformer)
        output_transformer = self.tranformer_encoder(input_transformer)
        wrist_input = torch.chunk(output_transformer, chunks=output_transformer.size(1), dim=1)[-1].view(-1, 256)
        wrist_position = self.wrist_position_decoder_network(wrist_input)
        wrist_rotation = self.wrist_rotation_decoder_network(wrist_input)
        return wrist_position, wrist_rotation
