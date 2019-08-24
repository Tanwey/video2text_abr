import torch
from torch.utils.data import Dataset
from easydict import EasyDict
from glob import glob
from utils.video import read_video


class ActivityNetDataset(torch.utils.data.Dataset):
    """
    """

    def __init__(self, config):
        super(ActivityNetDataset, self).__init__()
        if type(config) == EasyDict:
            config = EasyDict(config)

        self.video
        self.caption_file = config.caption_file
        self.transforms = config.transforms

    def __getitem__(self, index):
        return

    def __len__(self):
        return


class ActivityNetVideoReader:
    def __init__(self, files):
        self.files = files

    def get_video(index):
        read_video()
