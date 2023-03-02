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

# load local package
from home.poac.lhr.Li_prediction.utils.mmd import mix_rbf_mmd2

torch.manual_seed(0)
np.random.seed(0)


class ADADR(nn.Module):
    def __init__(self, opt, input_size, hidden_size, num_layers=2):
        super(ADADR, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.opt = opt

        self.feature_extractor_lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)

        self.feature_extractor_conv = nn.Sequential(
            nn.Conv1d(9, 32, kernel_size=2),
            nn.BatchNorm1d(32),
            nn.Conv1d(32, 4, kernel_size=2),
            nn.BatchNorm1d(4)
        )

        self.feature_extractor_linear = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.domain_adaptor = nn.Sequential(
            nn.Linear(32, 64),
            nn.Sigmoid(),
            nn.BatchNorm1d(64)
        )

        self.feature_predictor = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features, _ = self.feature_extractor_lstm(x)
        features = self.feature_extractor_linear(features[:, -1, :])
        # elif self.opt.net == 'conv1d':
        #     features = self.feature_extractor_conv(x)
        #     features = features.reshape(x.shape[0], -1)

        domain_features = self.domain_adaptor(features)
        prediction = self.feature_predictor(features)
        return features, domain_features, prediction

