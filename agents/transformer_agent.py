import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm
import datetime
import os

from agents.base import BaseNNAgent
from graph.models.transformer import Transformer
from utils.mask import create_look_ahead_mask
from datasets.MVADdataset import MVADFeatureDataset


class TransformerAgent(BaseAgent):
    def __init__(self, config):
        super(TransformerAgent, self).__init__(config)
        # Variable
        self.current_epoch = 1
        self.start_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.device = torch.device(self.config.device)
        self.model = Transformer(
            tar_vocab_size=self.config.tar_vocab_size,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            encoder_num_layers=self.config.encoder_num_layers,
            decoder_num_layers=self.config.decoder_num_layers,
            dff=self.config.dff,
            dropout=self.config.dropout,
            max_seq_length=self.config.max_seq_length)

        # Dataset
        # TODO: Dataset
        train_dataset = MVADFeatureDataset(feature_files, corpus_file)
        train_dataloader = torch.utils.DataLoader(
            train_dataset, batch_size=self.config.batch_size, suffle=True)

        # Loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.config.pad_id)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate,
                                    betas=(self.config.beta1, self.config.beta2), eps=1e-9)

        # Tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.config.tokenizer_file)

        # Tensorboard
        if self.config.summary_writer_dir is not None:
            self.writer = SummaryWriter(self.config.summary_writer_dir)

        # Load Check point
        if self.config.load_checkpoint is not None:
            self.load_checkpoint(self.config.load_checkpoint)

    def save_checkpoint(self):
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.save_checkpoint, self.start_datetime, str(self.epoch)))

    def load_checkpoint(self):
        checkpoint = torch.load(self.config.load_checkpoint)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def run(self):
        if self.train_dataloader is not None:
            train()

        if self.test_dataloader is not None:
            test()

    def train(self):
        for epoch in range(self.current_epoch, self.config.epoch):
            self.current_epoch = epoch
            train_one_epoch()

            if epoch % self.config.val_interval == 0 and self.val_dataloader is not None:
                validate()

    def train_one_epoch(self):
        for batch, sample in enumerate(self.train_dataloader, start=1):
            # Predict
            feature = sample['feature'].transpose(
                0, 1).to(self.device)  # (seq, batch, 1024)
            token = sample['token']
            token_inp = token[:, :-1].to(self.device)
            token_tar = token[:, 1:].to(self.device)
            inp_key_padding_mask = sample['inp_key_padding_mask'].to(
                self.device)
            tar_key_padding_mask = sample['tar_key_padding_mask'].to(
                self.device)
            mem_key_padding_mask = sample['mem_key_padding_mask'].to(
                self.device)
            tar_attn_mask = create_look_ahead_mask(
                token_inp.size(1)).to(self.device)

            prediction = self.model(feature, token_inp, inp_key_padding_mask,
                                    tar_key_padding_mask, mem_key_padding_mask, tar_attn_mask=tar_attn_mask)

            loss = self.criterion(prediction.transpose(1, 2), token_tar)

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Eval

    def validate(self):
        pass

    def test(self):
        pass

    def finalize(self):
        pass
