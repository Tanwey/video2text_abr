import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset_extracted import ExtractedFeatureDataset

from models.transformer import Transformer, create_look_ahead_mask
from models.optimizer import ScheduledOptim

import transforms
import sentencepiece as spm

import numpy as np
import time
import os
import datetime

tar_vocab_size = 5000
d_model = 1024
num_heads = 8
num_layers = 6
dff = 2048
dropout = 0.1
max_seq_length = 512  # For positional encoding
inp_max_seq_length = 256
tar_max_seq_length = 360
cuda_device = 'cuda:1'
feature_path = '../MVAD/I3D_rgb/train'
corpus_file = '../MVAD/corpus_M-VAD_train.txt'
tokenizer_file = 'tokenizer.model'
checkpoint = os.path.join(
    '../checkpoint', datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
BATCH = 16
EPOCH = 30
beta1 = 0.9
beta2 = 0.98
lr = 0.000316
warmup_step = 4000


def accuracy_metrics(predict, target):
    matched_matrix = (torch.max(predict, dim=-1)[1] == target)
    mask = (target != 0)
    matched_matrix = matched_matrix * mask
    acc = matched_matrix.type(torch.float).sum() / mask.type(torch.float).sum()
    return acc * 100.0


def train_step(sample, model):
    pass


def main():
    # Cuda
    device = torch.device(cuda_device)

    # Model
    model = Transformer(tar_vocab_size, d_model, num_heads,
                        num_layers, dff, dropout, max_seq_length)
    model.to(device)
    model.train()

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_file)

    # Transform
    feature_transform = transforms.Compose([
        transforms.FeaturePadding(inp_max_seq_length)
    ])
    caption_transform = transforms.Compose([
        transforms.CaptionPadding(tar_max_seq_length, sp.PieceToId('<PAD>'))
    ])

    # Dataset
    train_dataset = ExtractedFeatureDataset(
        feature_path, corpus_file, inp_max_seq_length, tar_max_seq_length, sp, feature_transform, caption_transform)

    # Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH, shuffle=True, num_workers=2)

    # Optimizer
    optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=lr, betas=(
        beta1, beta2), eps=1e-9), d_model, warmup_step)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Check point
    # torch.utils.ch

    start_time = time.time()
    for epoch in range(EPOCH):
        train_loss = 0
        train_accuracy = 0
        for batch, sample in enumerate(train_loader):
            # Load
            feature = sample[0].to(device)
            # (inp_max_seq_length, batch, d_model)
            feature = feature.transpose(0, 1)
            assert BATCH == feature.size(1)
            caption = sample[1]  # (batch, tar_max_seq_length)
            assert BATCH == caption.size(0)
            inp_key_padding_mask = sample[3].to(device)
            tar_key_padding_mask = sample[4].to(device)
            mem_key_padding_mask = sample[5].to(device)
            caption_inp = caption[:, :-1].to(device)
            caption_tar = caption[:, 1:].to(device)
            tar_attn_mask = create_look_ahead_mask(
                caption_inp.size(1)).to(device)

            prediction = model(feature, caption_inp, inp_key_padding_mask,
                               tar_key_padding_mask, mem_key_padding_mask, tar_attn_mask=tar_attn_mask)

            assert BATCH == prediction.size(0)
            loss = criterion(prediction.transpose(1, 2), caption_tar)
            train_accuracy += accuracy_metrics(prediction, caption_tar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()

            train_loss += loss.item()

            if (batch + 1) % 100 == 0:
                current_time = time.time()
                print('Epoch {} Batch {}  {:.1f}s - train_loss: {:6f} train_accuracy: {:4f}%'.format(epoch + 1, batch + 1,
                                                                                                     current_time - start_time, train_loss / (batch + 1), train_accuracy / batch))

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / batch,
        }, checkpoint)
        current_time = time.time()
        print('Epoch {}  {:.1f}s - train_loss: {:6f} train_accuracy: {:4f}%'.format(epoch + 1,
                                                                                    current_time - start_time, train_loss / (batch + 1), train_accuracy / batch))


if __name__ == '__main__':
    main()
