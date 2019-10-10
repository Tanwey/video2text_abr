import torch
import numpy as np
import time
import os
import logging
from glob import glob
from easydict import EasyDict

from utils.video import read_video, save_video
from datasets.video_dataset import VideoDatasetFromDir


class BaseFeatureExtractor:
    def __init__(self, model, save_dir):
        """
        Args:
            model (subclass torch.nn.Module): Model for extraction
            save_dir (str): Directory to save
        """
        self.model = model.eval()

        self.save_dir = save_dir
        if os.path.exists(save_dir) is False:
            os.makedirs(save_dir)

    def extract_feature(self, video, save_name):
        """Extract feature from video and save as save_path/save_name.npy
        Args:
            video (Tensor): Video Tensor input to model
            save_name (str): File name to save
        """
        path = os.path.join(self.save_path, save_name + '.npy')
        # Check the file already exist
        if os.path.exists(path) is True:
            print('{} already exist'.format(path))
            return False

        # Extract feature
        try:
            extracted_feature = self.model(video)
        except:
            print('ERROR MODEL CANNOT EXTRACT VIDEO'.format(path))

        # Save as .npy
        try:
            numpy_feature = extracted_feature.to('cpu').numpy()
            np.save(path, numpy_feature)
            return True
        except:
            print('ERROR NUMPY_NOT_SAVED: ',
                  numpy_feature.shape, path, numpy_feature)
            return False

    def extract_feature_from_dataloader(self, dataloader):
        """
        Args:
            dataloader (torch.utils.data.Dataloader): 
                Dataloader must return {'video': video, 'caption': caption, 'video_file': video_file}
        Returns:
            error_videos (List[str]): Video file name that cannot be extracted
        """
        start_time = time.time()

        error_videos = []
        for batch, sample in enumerate(dataloader):
            video, caption, video_file = sample['video'].to(
                self.gpu_device), sample['caption'], sample['video_file']
            ret = self.extract_feature(video, video_file[0])

            if ret is False:
                error_videos.append(video_file)

            if (batch + 1) % 100 == 0:
                current_time = time.time()
                print('{} have extracted - {}s'.format(batch + 1,
                                                       int(current_time - start_time)))

        print('Extraction End')
        print('Error Videos: ', len(error_videos))
        for i, video in enumerate(error_videos):
            print('{}. {}'.format(i, video))
        return error_videos


class BaseAgent:
    def __init__(self, config):
        if isinstance(config, EasyDict) is False:
            config = EasyDict(config)
        self.config = config

    def run(self):
        """
        The main operator
        """
        raise NotImplementedError

    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        """
        raise NotImplementedError
