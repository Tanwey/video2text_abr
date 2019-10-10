import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import sentencepiece as spm
import datetime
import os

from agents.base import BaseAgent
from graph.models.transformer import Transformer
from utils.mask import create_look_ahead_mask
from datasets.MVADdataset import MVADFeatureDataset
from utils.beam_search import BeamSearch
from utils.metrics import accuracy_batch


class TransformerAgent(BaseAgent):
    def __init__(self, config):
        super(TransformerAgent, self).__init__(config)
        # Variable
        self.current_epoch = 1
        self.start_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.device = torch.device(self.config.device)
        print(config)
        self.model = Transformer(
            tar_vocab_size=self.config.graph.token.tar_vocab_size,
            encoder_d_model=self.config.graph.model.encoder_d_model,
            decoder_d_model=self.config.graph.model.decoder_d_model,
            num_heads=self.config.graph.model.num_heads,
            encoder_num_layers=self.config.graph.model.encoder_num_layers,
            decoder_num_layers=self.config.graph.model.decoder_num_layers,
            dff=self.config.graph.model.dff,
            dropout=self.config.graph.model.dropout,
            max_seq_length=self.config.graph.model.max_seq_length).to(self.device)

        # Tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(self.config.graph.token.tokenizer_file)

        # Dataset
        if self.config.train.train_feature_list_file is not None:
            with open(self.config.train.train_feature_list_file, 'r') as f:
                train_feature_files = f.readlines()
                train_feature_files = [feature_file.strip(
                ) + '.npy' for feature_file in train_feature_files]

            train_dataset = MVADFeatureDataset(
                train_feature_files, self.config.train.train_corpus_file, self.config.graph.sequence.inp_max_sequence_size, self.config.graph.sequence.tar_max_sequence_size, self.sp, self.config.graph.sequence.cut_sequence)
            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.config.train.train_batch_size, shuffle=True)

        if self.config.val.val_feature_list_file is not None:
            with open(self.config.val.val_feature_list_file, 'r') as f:
                val_feature_files = f.readlines()
                val_feature_files = [feature_file.strip(
                ) + '.npy' for feature_file in val_feature_files]

            val_dataset = MVADFeatureDataset(
                val_feature_files, self.config.val.val_corpus_file, self.config.graph.sequence.inp_max_sequence_size, self.config.graph.sequence.tar_max_sequence_size, self.sp, self.config.graph.sequence.cut_sequence)
            self.val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.config.val.val_batch_size, shuffle=False)

        if self.config.test.test_feature_list_file is not None:
            with open(self.config.test.test_feature_list_file, 'r') as f:
                test_feature_files = f.readlines()
                test_feature_files = [feature_file.strip(
                ) + '.npy' for feature_file in test_feature_files]
            test_dataset = MVADFeatureDataset(
                test_feature_files, self.config.test.test_corpus_file, self.config.graph.sequence.inp_max_sequence_size, self.config.graph.sequence.tar_max_sequence_size, self.sp, self.config.graph.sequence.cut_sequence)
            self.test_dataloader = torch.utils.data.DataLoader(
                test_dataset, batch_size=self.config.test.test_batch_size, shuffle=False)

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=self.config.graph.token.pad_id)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.graph.optimizer.learning_rate,
                                    betas=(self.config.graph.optimizer.beta1, self.config.graph.optimizer.beta2), eps=1e-9)

        # Tensorboard
        if self.config.summary_writer_dir is not None:
            if os.path.exists(self.config.summary_writer_dir) is False:
                os.makedirs(self.config.summary_writer_dir)
            self.summary_writer = SummaryWriter(self.config.summary_writer_dir)

        # Load Check point
        if self.config.load_checkpoint is not None:
            self.load_checkpoint(self.config.load_checkpoint)

        # Save Check point
        if self.config.save_checkpoint is not None:
            if os.path.exists(self.config.save_checkpoint) is False:
                os.makedirs(self.config.save_checkpoint)

    def save_checkpoint(self):
        torch.save({
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.config.save_checkpoint, self.start_datetime, str(self.current_epoch)))

    def load_checkpoint(self):
        checkpoint = torch.load(self.config.load_checkpoint)
        self.current_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def run(self):
        """If train_dataloader is not provided only test runs
        train() contains train loop and validation loop
        """
        if self.train_dataloader is not None:
            self.train()

        if self.test_dataloader is not None:
            self.test()

    def train(self):
        """If val_dataloader is not provided only train runs"""
        for epoch in range(self.current_epoch, self.config.train.epoch):
            self.current_epoch = epoch
            self.train_one_epoch()

            if epoch % self.config.val.val_interval == 0 and self.val_dataloader is not None:
                self.validate()

    def train_one_epoch(self):
        train_loss = 0
        train_accuracy = 0
        self.model.train()
        for step, sample in enumerate(self.train_dataloader, start=1):
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

            # Eval
            loss = self.criterion(prediction.transpose(1, 2), token_tar)
            train_loss += loss.item()

            train_accuracy += accuracy_batch(
                torch.max(prediction, dim=-1)[1], token_tar)
            if step % 10 == 0:
                print('{} EPOCH {} / {} - Loss: {}, Accuracy: {}'.format(self.current_epoch, step,
                                                                         len(self.train_dataloader), train_loss / step, train_accuracy / step))

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # summary
        if self.config.save_checkpoint is not None:
            self.save_checkpoint()
        if self.config.summary_writer_dir is not None:
            self.summary_writer.add_scalar(
                'loss/train', train_loss / step, self.current_epoch)
            self.summary_writer.add_scalar(
                'accuracy/train', train_accuracy / step, self.current_epoch)
        print('{} EPOCH - Loss {}, Accuracy: {}'.format(self.current_epoch,
                                                        train_loss, train_accuracy))

    def validate(self):
        self.model.eval()
        val_accuracy = 0
        for step, sample in enumerate(self.val_dataloader, start=1):
            feature = sample['feature'].transpose(
                0, 1).to(self.device)  # (seq, batch, 1024)
            token = sample['token']
            inp_key_padding_mask = sample['inp_key_padding_mask'].to(
                self.device)
            mem_key_padding_mask = sample['mem_key_padding_mask'].to(
                self.device)

            output = torch.Tensor(token.size(0),
                                  1).fill_(self.config.graph.token.start_id).to(self.device)
            for i in range(1, self.tar_max_seq_length):
                tar_attn_mask = create_look_ahead_mask(i).to(self.device)
                prediction = self.model(feature, output, inp_key_padding_mask,
                                        mem_key_padding_mask=mem_key_padding_mask, tar_attn_mask=tar_attn_mask)
                prediction = prediction[:, -1:, :]
                predicted_id = torch.max(prediction, dim=-1)[1]
                output = torch.cat((output, predicted_id), dim=1)

            # Eval
            val_accuracy += accuracy_batch(
                prediction[:, 1:], token[:, 1:])
        # TODO: Blue, Meteor
        # self.summary_writer.add_scalar(
        #     'loss/val', val_loss / batch, self.current_epoch)
        self.summary_writer.add_scalar(
            'accuracy/val', val_accuracy / step, self.current_epoch)
        print('{} EPOCH - Accuracy: {}'.format(self.current_epoch, val_accuracy / step))

    def test(self):
        #beam_search = BeamSearch(self.config.beam_k, self.model, self.config.start_id, self.config.end_id, self.config.tar_max_seq_length, self.config.beam_min_length, self.config.beam_num_required)
        self.model.eval()
        # TODO: Beamsearch
        # TODO: Blue, Meteor
        # TODO: Summary writer
        pass

    def finalize(self):
        pass
