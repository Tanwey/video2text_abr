import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
import re
import yaml

from datasets.feature_dataset import FeatureDatasetFromDir, BaseFeatureDataset
from utils.token_transforms import TokenPadding
from utils.mask import create_padding_mask_from_size


# Max sequence Length of MVAD
# feature train 222
# feature test 165
# caption train 335
# caption test 246

class TransformerCaption:
    def __init__(self, corpus_file, transform=None):
        """Corpus_file formated with YAML or JSON
        Args:
            corpus_file (str): corpus YAML or JSON file 
            transform (Compose, default: None): Compose of transforms for caption
        """
        self.corpus_file = corpus_file
        self.transform = transform
        with open(corpus_file, 'r') as f:
            self.corpus = yaml.load(f, yaml.Loader)

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, file):
        caption = self.corpus[file]
        if self.transform is not None:
            caption = self.transform(caption)
        return caption


class TransformerFeatureDataset(Dataset):
    def __init__(self, feature_files=None, corpus_file=None, inp_max_sequence_size=None, tar_max_sequence_size=None, sp_processor=None, cut_sequence=False, feature_transform=None, caption_transform=None):
        """
        Args:
            feature_files (List[str], default: None): List of files
            corpus_file (str, default: None): Corpus YAML or JSON file 
            inp_max_sequence_size (int, default: None)
            tar_max_sequence_size (int, default: None)
            sp_processor ()
            cut_sequence (bool, default: False): If True, cut sequences longer than max_sequence_size
            feature_transform (Compose, default: None): Compose of transform for feature
            caption_transform (Compose, default: None): Compose of transforms for caption
        __getitem__:
            sample (Dict['feature', 'feature_file', 'caption', 'token', inp_key_padding_mask', 'tar_key_padding_mask', 'mem_key_padding_mask'])
        """
        # Feature dataset
        if feature_files is not None:
            self.feature_dataset = BaseFeatureDataset(
                feature_files, max_sequence_size=inp_max_sequence_size, cut_sequence=cut_sequence, transform=feature_transform)
            feature_files = self.feature_dataset.feature_files
        else:
            assert (feature_files is not None)
        self.feature_files = feature_files
        self.inp_max_sequence_size = inp_max_sequence_size

        # Caption
        if corpus_file is not None:
            self.caption_dataset = TransformerCaption(
                corpus_file, transform=caption_transform)
        else:
            print('corpus_file not provided')
            exit(-1)

        # Check sp_processor
        if sp_processor is not None:
            self.sp_processor = sp_processor
            self.start_id = self.sp_processor.PieceToId('<s>')
            self.end_id = self.sp_processor.PieceToId('</s>')
            self.pad_id = sp_processor.PieceToId('<pad>')
        else:
            print('sp_processor not provided')
            exit(-1)

        # Caption padding
        if tar_max_sequence_size is not None:
            self.token_padding = TokenPadding(
                tar_max_sequence_size, pad_id=self.pad_id, cut_sequence=cut_sequence)

        self.tar_max_sequence_size = tar_max_sequence_size

    def __getitem__(self, index):
        sample = self.feature_dataset[index]

        # inp, mem key padding mask
        if self.inp_max_sequence_size is not None:
            key_padding_mask = sample.pop('padding_mask')
            sample['inp_key_padding_mask'] = key_padding_mask
            sample['mem_key_padding_mask'] = key_padding_mask

        # Caption
        feature_file = sample['feature_file']
        original_file = re.sub('\.npy$', '', os.path.split(feature_file)[-1])
        caption = self.caption_dataset[original_file]
        sample['caption'] = caption

        # Token
        token = [self.start_id,
                 *list(self.sp_processor.EncodeAsIds(caption)), self.end_id]
        token = torch.LongTensor(token)

        # Token padding
        if self.tar_max_sequence_size is not None:
            tar_key_padding_mask = create_padding_mask_from_size(
                self.tar_max_sequence_size - 1, token.size(0))
            sample['tar_key_padding_mask'] = tar_key_padding_mask
            token = self.token_padding(token)

        sample['token'] = token

        return sample

    def __len__(self):
        return len(self.feature_files)
