import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from datasets.MVADvideo_dataset import MVADCorpusReader
import re
from functools import reduce

from datasets.feature_dataset import FeatureDatasetFromDir, BaseFeatureDataset
from utils.token_transforms import TokenPadding
from utils.mask import create_padding_mask_from_size


# Max sequence Length
# feature train 222
# feature test 165
# caption train 335
# caption test 246

class MVADCaption:
    def __init__(self, corpus_file, transform=None):
        """
        Args:
            corpus_file (str): MVAD corpus file
            transform (Compose, default: None): Compose of transforms for caption

        get_caption:
            Args:
                video_file (str)
            Returs:
                caption (str)
        """
        self.corpus_file = corpus_file
        self.transform = transform
        with open(corpus_file, 'r') as f:
            lines = f.readlines()
            self.corpus = reduce(self._line_2_dict, lines, {})

    def __len__(self):
        return len(self.corpus)

    def _line_2_dict(self, corpus_dict, line):
        file_name, caption = list(map(str.strip, line.split('\t')))
        corpus_dict.update({file_name: caption})
        return corpus_dict

    def get_caption(self, video_file):
        caption = self.corpus[video_file]
        if self.transform is not None:
            caption = self.transform(caption)
        return caption


class MVADFeatureDataset(Dataset):
    def __init__(self, feature_dir=None, feature_files=None, corpus_file=None, inp_max_sequence_size=None, tar_max_sequence_size=None, sp_processor=None, cut_sequence=False, feature_transform=None, caption_transform=None):
        # Feature dataset
        if feature_dir is not None:
            self.feature_dataset = FeatureDatasetFromDir(
                feature_dir, max_sequence_size=inp_max_sequence_size, cut_sequence=cut_sequence, transform=feature_transform)
        elif feature_files is not None:
            self.feature_dataset = BaseFeatureDataset(
                feature_files, max_sequence_size=inp_max_sequence_size, cut_sequence=cut_sequence, transform=feature_transform)
            feature_files = self.feature_dataset.feature_files
        else:
            assert (feature_files is not None) or (feature_dir is not None)
        self.feature_files = feature_files
        self.inp_max_sequence_size = inp_max_sequence_size

        # Caption
        if corpus_file is not None:
            self.caption_dataset = MVADCaption(
                corpus_file, transform=caption_transform)
        else:
            assert corpus_file is not None

        # Check sp_processor
        if sp_processor is not None:
            self.sp_processor = sp_processor
            self.start_id = self.sp_processor.PieceToId('<s>')
            self.end_id = self.sp_processor.PieceToId('</s>')
            self.pad_id = sp_processor.PieceToId('<pad>')
        else:
            assert sp_processor is not None

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
        video_file = re.sub('\.npy$', '', os.path.split(feature_file)[-1])
        caption = self.caption_dataset.get_caption(video_file)
        sample['caption'] = caption

        # Token
        token = [self.start_id,
                 *list(self.sp_processor.EncodeAsIds(caption)), self.end_id]
        token = torch.LongTensor(token)

        # Token padding
        if self.tar_max_sequence_size is not None:
            tar_key_padding_mask = create_padding_mask_from_size(
                self.tar_max_sequence_size, token.size(0))
            sample['tar_key_padding_mask'] = tar_key_padding_mask
            token = self.token_padding(token)

        sample['token'] = token

        return sample

    def __len__(self):
        return len(self.feature_files)


class ExtractedFeatureDataset(Dataset):
    def __init__(self, feature_path=None, feature_files=None, corpus_file=None, inp_max_sequence_size=None, tar_max_sequence_size=None, sp_processor=None, feature_transform=None, caption_transform=None):
        '''Load Extracted Feature of Video, tokenized caption, metadata, and masks
            Args:
              feature_path: If None, feature_file should be given
              corpus_path:
              inp_max_sequence:
              tar_max_sequence:
              sp_processor:
              feature_transform:
              caption_transform:
              feature_files: If featuer_path is None, get from feature_files
            Getitem:
              feature:
              caption:
              video_file:
              inp_key_padding_mask:
              tar_key_padding_mask:
              mem_key_padding_mask:
        '''
        if feature_path is not None:
            self.feature_path = feature_path
            self.feature_files = glob.glob(os.path.join(feature_path, '*'))
        else:
            self.feature_files = feature_files

        self.corpus_path = corpus_path
        self.inp_max_sequence_size = inp_max_sequence_size
        self.tar_max_sequence_size = tar_max_sequence_size
        self.sp_processor = sp_processor
        self.feature_transform = feature_transform
        self.caption_transform = caption_transform
        self.corpus_reader = MVADCorpusReader(corpus_path)

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, index):
        feature_file = self.feature_files[index]
        video_file = re.sub('\.npy$', '', os.path.split(feature_file)[-1])
        feature = torch.from_numpy(np.load(feature_file))  # (seq, d_model)
        if feature.size(0) > self.inp_max_sequence_size:
            return
        caption = self.corpus_reader.get_corpus()[video_file]  # string
        caption = [self.sp_processor.PieceToId('<s>')] + self.sp_processor.EncodeAsIds(caption) + \
            [self.sp_processor.PieceToId('</s>')]  # list[int]
        if len(caption) > self.tar_max_sequence_size:
            return

        inp_key_padding_mask = create_padding_mask_from_size(
            self.inp_max_sequence_size, feature.shape[0])
        mem_key_padding_mask = inp_key_padding_mask

        caption = torch.IntTensor(caption)

        if self.feature_transform is not None:
            feature = self.feature_transform(feature)

        if self.caption_transform is not None:
            caption = self.caption_transform(caption).type(torch.int64)

        tar_key_padding_mask = create_padding_mask_from_data(caption[:-1])

        return feature, caption, video_file, inp_key_padding_mask, tar_key_padding_mask, mem_key_padding_mask
