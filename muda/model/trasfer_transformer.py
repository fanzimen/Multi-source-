"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *

# load local packages
from home.poac.lhr.Li_prediction.baselines.Pytorch_Transfomer.utils import *


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn , n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)

        return x

class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads = 1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)

        a = self.attn2(x, kv = enc)
        x = self.norm2(a + x)

        a = self.fc1(F.elu(self.fc2(x)))

        x = self.norm3(x + a)
        return x

class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers = 1, n_encoder_layers = 1, n_heads = 1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len

        #Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads))

        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads))

        self.pos = PositionalEncoding(dim_val)

        #Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)

    def forward(self, x):
        #encoder
        e = self.encs[0](self.pos(self.enc_input_fc(x)))
        for enc in self.encs[1:]:
            e = enc(e)
        encoder_e = e

        #decoder
        d = self.decs[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), e)
        decoder_e = d
        for dec in self.decs[1:]:
            d = dec(d, e)

        #output
        x = self.out_fc(d.flatten(start_dim=1))

        return x, encoder_e, decoder_e, d.flatten(start_dim=1)



## test
if __name__ == '__main__':

    ## 参数
    enc_seq_len = 10
    dec_seq_len = 2
    output_sequence_len = 1
    input_size = 1

    dim_val = 10
    dim_attn = 5
    lr = 0.002
    epochs = 20

    n_heads = 3

    n_decoder_layers = 3
    n_encoder_layers = 3

    batch_size = 15

    ## 初始化网络
    t = Transformer(dim_val, dim_attn, input_size, dec_seq_len, output_sequence_len, n_decoder_layers, n_encoder_layers, n_heads)
    optimizer = torch.optim.Adam(t.parameters(), lr=lr)

    ## 产生数据 [batch_size, sequence_len, dimension], [batch_size, output_sequence_len]
    ## x shape:torch.Size([15, 10, 1]), y shape:torch.Size([15, 1])
    x, y = get_data(batch_size, enc_seq_len, output_sequence_len)
    print('x shape:{}, y shape:{}'.format(x.shape, y.shape))

    ## 将数据送入模型
    out = t(x)
    print('out shape:', out.shape)