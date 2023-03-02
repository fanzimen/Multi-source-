"#_*_ coding:utf-8 _*_"
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/..")))
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import math
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score
import parser
import torch.utils.data as Data
import random

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [seq_len, batch_size, d_model]
        :return:
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_atten_pad_mask(seq_q, seq_k):
    """
    :param seq_q: [batch_size, seq_len]
    :param seq_k: [batch_size, seq_len]
    :return:
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_atten_mask = seq_k.data.eq(0).unsqueeze(1)
    return pad_atten_mask.expand(batch_size, len_q, len_k)


def get_atten_subsequence_mask(seq):
    """
    :param seq: [batch_size, tgt_len]
    :return:
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.W_Q = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_K = nn.Linear(self.d_model, self.d_k * self.n_heads, bias=False)
        self.W_V = nn.Linear(self.d_model, self.d_v * self.n_heads, bias=False)
        self.fc = nn.Linear(self.n_heads * self.d_v, self.d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                           2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1,
                                                  1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(self.d_model).cuda()(output + residual), attn


class PosiwiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PosiwiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.fc = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(self.d_ff, self.d_model, bias=False)
        )

    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        """
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output+residual)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.enc_self_attn = MultiHeadAttention(self.d_model, self.d_k, self.d_v, self.n_heads)
        self.pos_ffn = PosiwiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        :return:
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.dec_self_attn = MultiHeadAttention(self.d_model, self.d_k, self.d_v, self.n_heads)
        self.dec_enc_attn = MultiHeadAttention(self.d_model, self.d_k, self.d_v, self.n_heads)
        self.pos_fnn = PosiwiseFeedForwardNet(self.d_model, self.d_ff)

    def forward(self, dec_inputs, enc_outpus, dec_self_attn_mask, dec_enc_attn_mask):
        """
        :param dec_inputs: [batch_size, tgt_len, d_model]
        :param enc_outpus:[batch_size, src_len, d_model]
        :param dec_self_attn_mask:[batch_size, tgt_len, tgt_len]
        :param dec_enc_attn_mask:[batch_size, tgt_len, src_len]
        :return:
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outpus, enc_outpus, dec_enc_attn_mask)
        dec_outputs = self.pos_fnn(dec_outputs)
        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super(Encoder, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.src_emb = nn.Embedding(self.src_vocab_size, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model)
        self.layers = nn.ModuleList(EncoderLayer(self.d_model, self.d_ff, self.d_k, self.d_v, self.n_heads) for _ in range(self.n_layers))

    def forward(self, enc_inputs):
        """
        :param enc_inputs: [batch_size, src_len]
        :return:
        """
        enc_outputs = self.src_emb(enc_inputs)
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)
        enc_self_attn_mask = get_atten_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, d_ff, d_k, d_v, n_heads, n_layers):
        super(Decoder, self).__init__()
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        self.tgt_emb = nn.Embedding(self.tgt_vocab_size, self.d_model)
        self.pos_emb = PositionalEncoding(self.d_model)
        self.layers = nn.ModuleList([DecoderLayer(self.d_model, self.d_ff, self.d_k, self.d_v, self.n_heads) for _ in range(self.n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_intpus: [batch_size, src_len]
        enc_outputs: [batsh_size, src_len, d_model]
        """
        dec_outputs = self.tgt_emb(dec_inputs)  # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).cuda()  # [batch_size, tgt_len, d_model]
        dec_self_attn_pad_mask = get_atten_pad_mask(dec_inputs, dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_subsequence_mask = get_atten_subsequence_mask(dec_inputs).cuda()  # [batch_size, tgt_len, tgt_len]
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).cuda()  # [batch_size, tgt_len, tgt_len]

        dec_enc_attn_mask = get_atten_pad_mask(dec_inputs, enc_inputs)  # [batc_size, tgt_len, src_len]

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, d_model, tgt_vocab_size, src_vocab_size, n_layers, d_ff, d_v, d_k, n_heads):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.tgt_vocab_size = tgt_vocab_size
        self.src_vocab_size = src_vocab_size
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.d_v = d_v
        self.d_k = d_k
        self.n_heads = n_heads

        self.encoder = Encoder(self.src_vocab_size, self.d_model, self.d_ff, self.d_k, self.d_v, self.n_heads, self.n_layers).cuda()
        self.decoder = Decoder(self.tgt_vocab_size, self.d_model, self.d_ff, self.d_k, self.d_v, self.n_heads, self.n_layers).cuda()
        self.projection = nn.Linear(self.d_model, self.tgt_vocab_size, bias=False).cuda()

    def forward(self, enc_inputs, dec_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        '''
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outpus: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)  # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns



## test
if __name__ == '__main__':

    ## 参数
    d_model = 512
    d_ff = 2048
    d_k = d_v = 64
    n_heads = 8
    n_layers = 6
    src_len = 10
    tgt_len = 1
    tgt_vocab_size = 1
    src_vocab_size = 8

    ## 定义模型和优化器
    model = Transformer(d_model, tgt_vocab_size, src_vocab_size, n_layers, d_ff, d_v, d_k, n_heads)
    model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    ## 定义输入和输出
    input = np.random.rand(10, 8)
    input = torch.from_numpy(input).long().cuda()
    print('input:', input.shape)

    output = np.random.rand(1, 1)
    output = torch.from_numpy(output).cuda()
    print('output:', output.shape)

    ## 数据送入模型
    pred, _, _, _ = model(input, input)
    print('pred:', pred.shape)







