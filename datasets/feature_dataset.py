import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os

from utils.feature_transform import FeaturePadding
from utils.mask import create_padding_mask_from_size


class BaseFeatureDataset(Dataset):
    def __init__(self, feature_files, max_sequence_size=None, pad_value=0, cut_sequence=False, transform=None):
        """Base wrapper of pytorch Dataset for feature
        Args:
            feature_files (List[str]): List of feature files 
            max_sequence_size (int, default: None): Maximum sequence length to pad. If None no padding
            pad_value (int, default: 0): Value for padding
            cut_sequence (bool, default: False): If True, when feature sequence is longer than max_sequence_size, cut the feature.
                If False, when feature sequence is longer than max_sequence_size, alert error.
            transform (Compose, default: None): Compose object of transforms

        __getitem__:
            Returns Dict['feature', 'feature_file']
            If max_sequence_size is not None, Dict['feature', 'feature_file', 'padding_mask']
        """
        super(BaseFeatureDataset, self).__init__()
        self.feature_files = [feature_file.strip()
                              for feature_file in feature_files]
        self.max_sequence_size = max_sequence_size
        # Padding
        if max_sequence_size is not None:
            self.pad_value = pad_value
            self.cut_sequence = cut_sequence
            self.feature_padding = FeaturePadding(
                max_sequence_size, pad_value=pad_value, cut_sequence=cut_sequence)
        self.transform = transform

    def __getitem__(self, index):
        feature_file = self.feature_files[index]
        feature = np.load(feature_file)
        feature = torch.from_numpy(feature)

        if self.transform is not None:
            feature = self.transform(feature)
        # Padding
        if self.max_sequence_size is not None:
            padding_mask = create_padding_mask_from_size(
                self.max_sequence_size, feature.size(0))
            feature = self.feature_padding(feature)

        sample = {'feature': feature, 'feature_file': feature_file}
        # Padding Mask
        if self.max_sequence_size is not None:
            sample['padding_mask'] = padding_mask
        return sample

    def __len__(self):
        return len(self.feature_files)


class FeatureDatasetFromDir(BaseFeatureDataset):
    def __init__(self, feature_dir, recursive=False, max_sequence_size=None, pad_value=0, cut_sequence=False, transform=None):
        """Base wrapper of pytorch Dataset for feature
        Args:
            feature_dir (str): Feature directory
            recursive (bool, default: False): If True recursively search files
            max_sequence_size (int, default: None): Maximum sequence length to pad. If None no padding
            pad_value (int, default: 0): Value for padding
            cut_sequence (bool): If True, when feature sequence is longer than max_sequence, cut the feature.
                If False, when feature sequence is longer than max_sequence, alert error.
            transform (Compose): Compose object of transforms

        __getitem__:
            Returns Dict['feature', 'feature_file']
            If max_sequence_size is not None, Dict['feature', 'feature_file', 'padding_mask']
        """
        self.feature_dir = feature_dir
        if recursive is False:
            paths = glob(os.path.join(feature_dir, '*'))
            feature_files = list(
                filter(lambda path: os.path.isfile(path), paths))
        elif recursive is True:
            paths = glob(os.path.join(feature_dir, '**', '*'), recursive=True)
            feature_files = list(
                filter(lambda path: os.path.isfile(path), paths))

        super(FeatureDatasetFromDir, self).__init__(feature_files,
                                                    max_sequence_size, pad_value, cut_sequence, transform)
