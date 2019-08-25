import torch
from torch.utils.data import Dataset
from glob import glob
import os
from utils.video import read_video, video_info


class BaseVideoDataset(Dataset):
    def __init__(self, video_files, transform=None):
        """Base wrapper of pytorch Dataset for video
        Args:
            video_files (List[str]): List of video files 
            transform (Compose): Compose object of transforms

        __getitem__:
            returns Dict['video', 'video_file', 'video_info']
                'video_info': Dict['count', 'height', 'width', 'fps']
        """
        super(BaseVideoDataset, self).__init__()
        self.video_files = video_files
        self.transform = transform

    def __getitem__(self, index):
        video_file = self.video_files[index]
        video = read_video(video_file)
        if self.transform is not None:
            video = self.transform(video)
        return {'video': video, 'video_file': video_file, 'video_info': video_info(video_file)}

    def __len__(self):
        return len(self.video_files)


class VideoDatasetFromDir(BaseVideoDataset):
    def __init__(self, video_dir, recursive=False, transform=None):
        """Wrapper of pytorch Dataset for video
        Get videos from directory.
        Args:
            video_dir (str): Video directory
            recursive (bool): If True recursively search files
            transform (Compose): Compose object of transforms

        __getitem__:
            returns Dict['video', 'video_file', 'video_info']
            'video_info': Dict['count', 'height', 'width', 'fps']
        """
        self.video_dir = video_dir
        if recursive is False:
            paths = glob(os.path.join(video_dir, '*'))
            video_files = list(
                filter(lambda path: os.path.isfile(path), paths))
        elif recursive is True:
            paths = glob(os.path.join(video_dir, '**', '*'), recursive=True)
            video_files = list(
                filter(lambda path: os.path.isfile(path), paths))

        super(VideoDatasetFromDir, self).__init__(video_files, transform)


class VideoDatasetFromDirs(BaseVideoDataset):
    def __init__(self, video_dirs, recursive=False, transform=None):
        """Wrapper of pytorch Dataset for video
        Get videos from directories.
        Args:
            video_dirs (List[str]): List of video directories
            recursive (bool): If True recursively search files
            transform (Compose): Compose object of transforms

        __getitem__:
            returns Dict['video', 'video_file']
            'video_info': Dict['count', 'height', 'width', 'fps']
        """
        self.video_dirs = video_dirs
        video_files = []
        if recursive is False:
            for video_dir in video_dirs:
                paths = glob(os.path.join(video_dir, '*'))
                video_files_tmp = filter(
                    lambda path: os.path.isfile(path), paths)
                video_files.extend(video_files_tmp)
        elif recursive is True:
            for video_dir in video_dirs:
                paths = glob(os.path.join(
                    video_dir, '**', '*'), recursive=True)
                video_files_tmp = filter(
                    lambda path: os.path.isfile(path), paths)
                video_files.extend(video_files_tmp)

        super(VideoDatasetFromDirs, self).__init__(video_files, transform)
