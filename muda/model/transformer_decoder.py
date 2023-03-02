"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))

import torch
import torch.nn as nn
import numpy as np
import time
import math
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoderLayer, TransformerEncoder

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
        self.encoder_layer = TransformerEncoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size*(sequence_len), 1)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=4, dropout=dropout)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        # self.last = nn.Linear(feature_size*sequence_len, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.permute(1, 0, 2)
        output = output.reshape(output.shape[0], -1)

        output = self.decoder(output)
        # output = self.last(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0))
        return mask

