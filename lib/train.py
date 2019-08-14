import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
d_model = 512
num_heads = 8
encoder_num_layers = 4
decoder_num_layers = 6
dff = 2048
dropout = 0.1
max_seq_length = 80  # For positional encoding
inp_max_seq_length = 50
tar_max_seq_length = 50
cuda_device = 'cuda:0'

#feature_path = '../MVAD/I3D_rgb_kinetics/train'
feature_path = None
#feature_files = None
train_feature_path = '../MVAD/I3D_rgb_kinetics/train'
with open('../MVAD/train_fine') as f:
    files = f.readlines()
    feature_files = list(map(lambda file: os.path.join(train_feature_path, str.strip(file) + '.npy'), files))
    feature_files.extend(list(map(lambda file: os.path.join(train_feature_path + '_fliped', str.strip(file) + '.npy'), files)))
                        
corpus_file = '../MVAD/corpus_M-VAD_train.txt'
tokenizer_file = 'tokenizer.model'
start_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
checkpoint = os.path.join('../checkpoint', start_datetime)
os.mkdir(checkpoint)
tensorboard_dir = os.path.join('../logs', start_datetime)
BATCH = 64
EPOCH = 50
beta1 = 0.9
beta2 = 0.98
lr = 0.0001
#warmup_step = 4000


def accuracy_metrics(predict, target):
    matched_matrix = (torch.max(predict, dim=-1)[1] == target)
    mask = (target != 0)
    matched_matrix = matched_matrix * mask
    acc = matched_matrix.type(torch.float).sum() / mask.type(torch.float).sum()
    return acc * 100.0


def main():
    # Cuda
    device = torch.device(cuda_device)

    # Model
    model = Transformer(tar_vocab_size, d_model, num_heads,
                        encoder_num_layers, decoder_num_layers, dff, dropout, max_seq_length)
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
        feature_path, corpus_file, inp_max_seq_length, tar_max_seq_length, sp, feature_transform, caption_transform, feature_files)

    # Dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH, shuffle=True, num_workers=2)

    # Optimizer
    #optimizer = ScheduledOptim(optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=1e-9), d_model, warmup_step)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           betas=(beta1, beta2), eps=1e-9)

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Tensorboard
    writer = SummaryWriter(tensorboard_dir)

    start_time = time.time()
    for epoch in range(1, EPOCH + 1):
        train_loss = 0
        train_accuracy = 0
        for batch, sample in enumerate(train_loader, start=1):
            # Load
            feature = sample[0].to(device)
            # (inp_max_seq_length, batch, d_model)
            feature = feature.transpose(0, 1)
            caption = sample[1]  # (batch, tar_max_seq_length)
            inp_key_padding_mask = sample[3].to(device)
            tar_key_padding_mask = sample[4].to(device)
            mem_key_padding_mask = sample[5].to(device)
            caption_inp = caption[:, :-1].to(device)
            caption_tar = caption[:, 1:].to(device)
            tar_attn_mask = create_look_ahead_mask(
                caption_inp.size(1)).to(device)

            prediction = model(feature, caption_inp, inp_key_padding_mask,
                               tar_key_padding_mask, mem_key_padding_mask, tar_attn_mask=tar_attn_mask)

            loss = criterion(prediction.transpose(1, 2), caption_tar)
            train_accuracy += accuracy_metrics(prediction, caption_tar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch % 100 == 0:
                current_time = time.time()
                print('Epoch {} Batch {}  {:.1f}s - train_loss: {:6f} train_accuracy: {:4f}%'.format(epoch, batch,
                                                                                                     current_time - start_time, train_loss / batch, train_accuracy / batch))

        # Tensorboard Graph
        if epoch == 1:
            with torch.no_grad():
                writer.add_graph(model, (feature, caption_inp))

        # Checkpoints
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss / batch,
        }, os.path.join(checkpoint, str(epoch)))

        # Logs
        writer.add_scalar('loss/train', train_loss / batch, epoch)
        writer.add_scalar('accuracy/train', train_accuracy / batch, epoch)

        current_time = time.time()
        print('Epoch {}  {:.1f}s - train_loss: {:6f} train_accuracy: {:4f}%'.format(epoch,
                                                                                    current_time - start_time, train_loss / batch, train_accuracy / batch))


if __name__ == '__main__':
    main()
