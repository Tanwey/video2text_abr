import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from datasetMVAD import MVADCorpusReader
import re
from models.transformer import create_padding_mask_from_size, create_padding_mask_from_data


# Max sequence Length
# feature train 222
# feature test 165
# caption train 335
# caption test 246
class ExtractedFeatureDataset(Dataset):
    def __init__(self, feature_path, corpus_path, inp_max_sequence_size, tar_max_sequence_size, sp_processor, feature_transform=None, caption_transform=None, feature_files=None):
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

    
class ExtractedFeatureDatasetOnlyFeature(Dataset):
    def __init__(self, feature_path, feature_files=None, transform=None):
        self.feature_path = feature_path
        if feature_path is None:
            self.feature_files = feature_files
        else:
            self.feature_files = glob.glob(os.path.join(feature_path, '*'))
        self.transform = transform
        
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, index):
        feature_file = self.feature_files[index]
        feature = torch.from_numpy(np.load(feature_file))
        
        if self.transform is not None:
            feature = self.transform(feature)
        return feature
        