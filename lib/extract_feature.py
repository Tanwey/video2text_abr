import torch
import numpy as np
import time
import os
from pytorch_i3d.pytorch_i3d import InceptionI3d


class Extractor:
    def __init__(self, model, path, gpu_device=None):
        self.model = model
        self.path = path
        if os.path.exists(path) is False:
            os.makedirs(path)

        self.gpu_device = gpu_device
        if gpu_device is None:
            self.gpu_device = torch.device('cuda')

    def extract_feature(self, video, video_file):
        path = os.path.join(self.path, video_file + '.npy')
        if os.path.exists(path) is True:
            return
        extracted_feature = self.model.extract_features(
            video)
        try:
            extracted_feature = extracted_feature.permute(1, 0)
        except:
            print('ERROR:', video_file, extracted_feature.size())
            return False

        try:
            numpy_feature = extracted_feature.to('cpu').numpy()
            np.save(path, numpy_feature)
            return True
        except:
            print('ERROR NUMPY_NOT_SAVED: ',
                  numpy_feature.shape, path, numpy_feature)
            return False

    def extract_feature_from_loader(self, loader):
        '''
            Args:
              loader: torch Dataloader
                  torch.utils.data.Dataloader {'video': video, 'caption': caption, 'video_file': video_file}
            Returns:
              error_videos: errored videos
        '''
        start_time = time.time()

        error_videos = []
        with torch.no_grad():
            for batch, sample in enumerate(loader):
                video, caption, video_file = sample['video'].to(
                    self.gpu_device), sample['caption'], sample['video_file']
                flag = self.extract_feature(video, video_file[0])

                if flag is False:
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
