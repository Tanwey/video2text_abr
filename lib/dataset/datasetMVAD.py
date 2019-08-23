from torch.utils.data import Dataset

import cv2
import numpy as np
import os
import glob

from functools import reduce
from typing import List, Dict


class MVADCorpusReader(object):
    def __init__(self, MVAD_corpus_path: str):
        '''Make Dict[file, caption] from MVAD corpus file
            input:
              MVAD_corpus_path: path to corpus of MVAD
        '''
        self.MVAD_corpus_path = MVAD_corpus_path
        with open(MVAD_corpus_path, 'r') as f:
            lines: List[str] = f.readlines()
            self.corpus = reduce(self._line_2_dict, lines, {})
            
    def _line_2_dict(self, corpus_dict: Dict[str, str], line: str) -> Dict[str, str]:
        file_name, caption = list(map(str.strip, line.split('\t')))
        corpus_dict.update({file_name: caption})
        return corpus_dict
    
    def get_corpus(self) -> Dict[str, str]:
        '''
            output:
              corpus: Dict[file, caption]
        '''
        return self.corpus
    
class MVADVideoReader:
    def __init__(self, videos_dir: str, train: bool, val: bool = None):
        self.videos_dir = videos_dir
        self.train = train
        self.val = val
        flag = 'train' if train is True else 'val' if val is True else 'test'
        self.videos_files = glob.glob(os.path.join(videos_dir, flag, '*'))
        
    def _read_video(self, file: str):
        '''Read from video file return video numpy 4dim array
            input:
              file: video file
            output: 4 dim numpy array video
        '''
        cap = cv2.VideoCapture(file)
        frames = []
        for i in range(int(cap.get(7))):
            ret, frame = cap.read()
            if ret is True:
                frames.append(np.expand_dims(frame[:, :, ::-1], axis=0))
            else:
                #print('{}frame cannot read')
                pass
        video = np.concatenate(frames)
        return video
        
    def get_video(self, index):
        video_file = self.videos_files[index]
        video = self._read_video(video_file)
        return video, os.path.split(video_file)[-1]
    
class MVADDataset(Dataset):
    def __init__(self, videos_dir: str, corpus_file: str, train: bool = False, val: bool = False, transform=None):
        '''MVAD dataset
            when __getitem__ call return {'video': video(4 dim numpy array), 'caption': caption(str)}
            input:
              videos_dir: MVAD directory
              corpus_file: MVAD corpus file
              train: bool type. True for train, False for test
              val: bool type. True for val, False for test (Train should be False)
              transform: default None
        '''
        if train is True and val is True:
            print('Train and val cannot be True at once. Dataset would be trainset')
        self.video_reader = MVADVideoReader(videos_dir, train, val)
        self.corpus_reader = MVADCorpusReader(corpus_file)
        self.train = train
        self.val = val
        self.transform = transform
        
    def __len__(self):
        return len(self.video_reader.videos_files)
    
    def __getitem__(self, index):
        video, video_file = self.video_reader.get_video(index)
        caption = self.corpus_reader.get_corpus()[os.path.split(video_file)[-1]]
        sample = {'video': video, 'caption': caption, 'video_file': video_file}
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample