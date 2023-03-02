"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder

torch.manual_seed(0)
np.random.seed(0)

####################################
# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number
# src = (S, N, E)
# tgt = (T, N, E)
####################################


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model, device='cuda:0')
        position = torch.arange(0, max_len, dtype=torch.float, device='cuda:0').unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device='cuda:0').float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):

    def __init__(self, sequence_len, feature_size=2, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size*(sequence_len), 16)
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 1)

        # self.decoder = nn.Sequential(
        #     nn.Conv1d(in_channels=8, out_channels=16, kernel_size=2),
        #     nn.BatchNorm1d(16),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.ReLU(True),
        #     nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2),
        #     nn.BatchNorm1d(32),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.ReLU(True),
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=2),
        #     nn.BatchNorm1d(64),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.ReLU(True),
        #     # nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3),
        #     # nn.BatchNorm1d(2048),
        #     # nn.MaxPool1d(kernel_size=2),
        #     # nn.ReLU(True)
        # )
        #
        # self.prediction1 = nn.Sequential()
        # self.prediction1.add_module('c_fc1', nn.Linear(64, 32))
        # self.prediction1.add_module('c_bn1', nn.BatchNorm1d(32))
        # self.prediction1.add_module('c_relu1', nn.ReLU(True))
        # self.prediction1.add_module('c_drop1', nn.Dropout2d())
        # self.prediction1.add_module('c_fc2', nn.Linear(32, 16))
        # self.prediction1.add_module('c_bn2', nn.BatchNorm1d(16))
        # self.prediction1.add_module('c_relu2', nn.ReLU(True))
        #
        # self.prediction2 = nn.Sequential()
        # self.prediction2.add_module('c_fc3', nn.Linear(16, 10))
        # self.prediction2.add_module('c_bn3', nn.BatchNorm1d(10))
        # self.prediction2.add_module('c_relu3', nn.ReLU(True))
        # self.prediction2.add_module('c_fc4', nn.Linear(10, 1))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

        # # 初始化decoder参数
        # for layer in self.decoder:
        #     if isinstance(layer, nn.Conv1d):
        #         nn.init.kaiming_normal(layer.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(layer, nn.BatchNorm1d):
        #         nn.init.constant_(layer.weight, 1)
        #         # nn.init.constant_(layer.bais, 0)
        #
        # # 初始化prediction1参数
        # for layer in self.prediction1:
        #     if isinstance(layer, nn.Linear):
        #         param_shape = layer.weight.shape
        #         layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)).float()
        #     elif isinstance(layer, nn.BatchNorm1d):
        #         nn.init.constant_(layer.weight, 1)
        #         # nn.init.constant_(layer.bais, 0)
        #
        # # 初始化prediction2参数
        # for layer in self.prediction2:
        #     if isinstance(layer, nn.Linear):
        #         param_shape = layer.weight.shape
        #         layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)).float()
        #     elif isinstance(layer, nn.BatchNorm1d):
        #         nn.init.constant_(layer.weight, 1)
        #         # nn.init.constant_(layer.bais, 0)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        encode = self.transformer_encoder(src, self.src_mask)
        encode = encode.permute(1, 2, 0)
        encode = encode.reshape(encode.shape[0], -1)

        # fc1层
        decode = self.decoder(encode)
        # print('decode:', decode.shape)
        # print('decode:', type(decode))
        encode1 = self.fc1(decode)

        # fc2层
        output = self.fc2(encode1)

        # print('output:', output.shape)

        return encode, encode1, output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        # mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        mask = mask.float().masked_fill(mask == 0, float(1.0)).masked_fill(mask == 1, float(1.0))
        return mask


# class Domain_Discriminator(nn.Module):
#     """
#     域迁移判别器
#     """
#     def __init__(self):
#         super(Domain_Discriminator, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3),
#             nn.MaxPool1d(kernel_size=2)
#         )
#
#         self.conv2 = nn.Sequential(
#             nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3),
#             nn.MaxPool1d(kernel_size=2)
#         )
#
#         self.fc = nn.Sequential(
#             nn.Linear(448, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, indata):
#         indata = indata.reshape(indata.shape[0], 1, -1)
#         x = self.conv1(indata)
#         x = self.conv2(x)
#         x = x.view(x.size(0), -1)
#         out = self.fc(x)
#         return out.view(-1)










