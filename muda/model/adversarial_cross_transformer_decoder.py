"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
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


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class TransAm(nn.Module):

    def __init__(self, sequence_len, feature_size=2, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size*(sequence_len), 256)

        # self.prediction = nn.Sequential(
        #     nn.Linear(16, 8),
        #     nn.Linear(8, 1)
        # )

        self.discriminator = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(True),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2),
            nn.ReLU(True)
        )

        self.fc = nn.Sequential(
            nn.Linear(896, 1),
            nn.Sigmoid()
        )

        self.prediction = nn.Sequential()
        self.prediction.add_module('c_fc1', nn.Linear(256, 100))
        self.prediction.add_module('c_bn1', nn.BatchNorm1d(100))
        self.prediction.add_module('c_relu1', nn.ReLU(True))
        self.prediction.add_module('c_drop1', nn.Dropout2d())
        self.prediction.add_module('c_fc2', nn.Linear(100, 100))
        self.prediction.add_module('c_bn2', nn.BatchNorm1d(100))
        self.prediction.add_module('c_relu2', nn.ReLU(True))
        self.prediction.add_module('c_fc3', nn.Linear(100, 10))
        self.prediction.add_module('c_bn3', nn.BatchNorm1d(10))
        self.prediction.add_module('c_relu3', nn.ReLU(True))
        self.prediction.add_module('c_fc4', nn.Linear(10, 1))

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, alpha):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)

        # 使用transformer提取特征
        encode = self.transformer_encoder(src, self.src_mask)
        encode = encode.permute(1, 0, 2)
        encode = encode.reshape(encode.shape[0], -1)

        encode = self.decoder(encode)

        # 将特征送入预测器，得到预测结果
        pred_output = self.prediction(encode)

        # 将特征送入判别器，得到判别结果
        reverse_encode = ReverseLayerF.apply(encode, alpha)
        input_dis = reverse_encode.reshape(reverse_encode.shape[0], 1, -1)
        dis_output = self.discriminator(input_dis)
        # print('dis output:', dis_output.shape)
        dis_output = dis_output.view(dis_output.size(0), -1)
        # print('dis output:', dis_output.shape)
        dis_output = self.fc(dis_output)

        return pred_output, dis_output.view(-1), encode

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask













