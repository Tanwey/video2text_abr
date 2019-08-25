import torch
from glob import glob
import os
import json

from utils.video import read_video


class ActivityNetDataset(torch.utils.data.Dataset):
    """
    """

    def __init__(self, video_dir, caption_file, transform):
        super(ActivityNetDataset, self).__init__()

        self.video_dir = video_dir
        self.video_files = glob(os.path.join(video_dir, '*.mp4'))
        self.caption_file = caption_file
        self.transforms = transforms

    def __getitem__(self, index):
        self.video_files
        self.transforms()
        return

    def __len__(self):
        return


class ActivityNetVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir=None, video_files=None, transform=None):
        super(ActivityNetVideoDataset, self).__init__()
        if video_files is None:
            self.video_dir = video_dir
            video_files = glob(os.path.join(video_dir, '*'))
        self.video_files = video_files
        self.transform = transform

    def __getitem__(self, index):
        video_file = self.video_files[index]
        video = read_video(video_file)
        if self.transform is not None:
            video = self.transform(video)
        return video

    def __len__(self):
        return len(video_files)
