import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
from functools import partial, reduce


class PositionalEncoder(nn.Module):
    # sin(position / 10000 ^ (2 * i / d_model))
    # cos(position / 10000 ^ (2 * i / d_model))
    def __init__(self, max_seq_length, d_model):
        super(PositionalEncoder, self).__init__()
        self.max_seq_length = max_seq_length
        self.d_model = d_model

        self.pe = self.get_angle(max_seq_length, d_model)

        self.pe[:, ::2] = torch.sin(self.pe[:, ::2])
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        p, batch, d = x.size()
        x += self.pe[:p, batch, 0:d] + x
        return x

    def get_angle(self, position, d_model):
        p = torch.arange(position, dtype=torch.float).view((position, 1))
        d = torch.arange(d_model, dtype=torch.float).view((1, d_model))
        angle = p / torch.pow(10000.0, (2 * (d // 2) / d_model))
        return angle

    def to(self, *args, **kwargs):
        self = super(PositionalEncoder, self).to(*args, **kwargs)
        self.pe = self.pe.to(*args, **kwargs)
        return self


class FeedForward(nn.Module):
    def __init__(self, d_model, dff, dropout=0.1):
        '''
            Args: Feed foward
              d_model: The number of feature
              dff: Dim for linear layer
              dropout: dropout percentage default 0.1

        '''
        super(FeedForward, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.linear1 = nn.Linear(d_model, dff)
        self.linear2 = nn.Linear(dff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AddNorm(nn.Module):
    def __init__(self, d_model, dropout):
        super(AddNorm, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, x, pre_x):
        x = self.dropout(x) + pre_x
        x = self.layernorm(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout

        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, dff, dropout)
        self.an1 = AddNorm(d_model, dropout)
        self.an2 = AddNorm(d_model, dropout)

    def forward(self, inp, key_padding_mask=None, attn_mask=None):
        pre_x = inp
        x = self.mha(inp, inp, inp, key_padding_mask=key_padding_mask,
                     attn_mask=attn_mask)[0]
        x = self.an1(x, pre_x)
        pre_x = x
        x = self.ff(x)
        x = self.an2(x, pre_x)
        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout = dropout

        self.mhas1 = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.mhas2 = nn.MultiheadAttention(d_model, num_heads, dropout)
        self.an1 = AddNorm(d_model, dropout)
        self.an2 = AddNorm(d_model, dropout)
        self.an3 = AddNorm(d_model, dropout)
        self.ff = FeedForward(d_model, dff, dropout)

    def forward(self, tar, mem, tar_key_padding_mask=None, mem_key_padding_mask=None, tar_attn_mask=None, mem_attn_mask=None):
        pre_x = tar
        x = self.mhas1(
            tar, tar, tar, key_padding_mask=tar_key_padding_mask, attn_mask=tar_attn_mask)[0]
        x = self.an1(x, pre_x)
        pre_x = x
        x = self.mhas2(
            x, mem, mem, key_padding_mask=mem_key_padding_mask, attn_mask=mem_attn_mask)[0]
        x = self.an2(x, pre_x)
        pre_x = x
        x = self.ff(x)
        x = self.an3(x, pre_x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout = dropout

        self.encoder_layers = nn.ModuleList()
        for i in range(num_layers):
            encoder_layer = TransformerEncoderLayer(
                d_model, num_heads, dff, dropout)
            self.encoder_layers.add_module(
                'encoder_layer{}'.format(i + 1), encoder_layer)

    def forward(self, inp, key_padding_mask=None, attn_mask=None):
        x = inp

        for encoder_layer in self.encoder_layers:
            x = encoder_layer(
                inp, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout = dropout

        self.decoder_layers = nn.ModuleList()
        for i in range(num_layers):
            decoder_layer = TransformerDecoderLayer(
                d_model, num_heads, dff, dropout)
            self.decoder_layers.add_module(
                'decoder_layer{}'.format(i + 1), decoder_layer)

    def forward(self, tar, mem, tar_key_padding_mask=None, mem_key_padding_mask=None, tar_attn_mask=None, mem_attn_mask=None):
        x = tar

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mem, tar_key_padding_mask=tar_key_padding_mask,
                              mem_key_padding_mask=mem_key_padding_mask, tar_attn_mask=tar_attn_mask, mem_attn_mask=mem_attn_mask)

        return x


class Transformer(nn.Module):
    def __init__(self, tar_vocab_size, d_model, num_heads, num_layers, dff, dropout=0.1, max_seq_length=512):
        super(Transformer, self).__init__()
        self.tar_vocab_size = tar_vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout = dropout
        self.max_seq_length = max_seq_length

        self.pe = PositionalEncoder(max_seq_length, d_model)
        self.encoder = TransformerEncoder(
            d_model, num_heads, num_layers, dff, dropout)
        self.word_embedding = nn.Embedding(tar_vocab_size, d_model)
        self.decoder = TransformerDecoder(
            d_model, num_heads, num_layers, dff, dropout)
        self.emb2voc = nn.Linear(d_model, tar_vocab_size)

    def forward(self, inp, tar, inp_key_padding_mask=None, tar_key_padding_mask=None, mem_key_padding_mask=None, inp_attn_mask=None, tar_attn_mask=None, mem_attn_mask=None):
        inp = self.pe(inp)
        mem = self.encoder(
            inp, key_padding_mask=inp_key_padding_mask, attn_mask=inp_attn_mask)

        tar = self.word_embedding(tar)
        tar = self.pe(tar)
        x = self.decoder(tar, mem, tar_key_padding_mask=tar_key_padding_mask,
                         mem_key_padding_mask=mem_key_padding_mask, tar_attn_mask=tar_attn_mask, mem_attn_mask=mem_attn_mask)
        x = self.emb2voc(x)

        return x

    def to(self, *args, **kwargs):
        self = super(Transformer, self).to(*args, **kwargs)
        self.pe.pe = self.pe.pe.to(*args, **kwargs)
        return self


def create_padding_mask_from_size(size, real_size):
    '''
        Args:
          size: sequense length (int)
          real_size: real sequence lenght (int)
        Returns:
          mask: Mask Tensor (seq)
    assert size > real_size
    '''
    mask = torch.ones((size))
    mask[:real_size] = 0
    return mask


def create_padding_mask_from_data(data):
    '''
        Args:
          data: Tensor (seq)
        Returns:
          mask: Mask Tensor (seq)
    '''
    mask = (data == 0).type(torch.uint8)
    return mask


def create_look_ahead_mask(size):
    '''
        Args:
          size: Sequence(time) size of data
        Returns:
          mask: Upper triangular matrix with -inf (size, size)
    '''
    mask = torch.triu(torch.ones(size, size), 1)
    mask = mask.float().masked_fill(mask == 1, float(
        '-1e9')).masked_fill(mask == 0, float(0.0))
    return mask
