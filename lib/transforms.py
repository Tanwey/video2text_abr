import torch
from torch import Tensor
import cv2
from typing import List, Dict
import numpy as np
import random
from models.transformer import create_padding_mask_from_size


class Compose:
    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)

        return sample


class VideoCompose(Compose):
    pass


class VideoResize:
    def __init__(self, output_size):
        '''
            input:
              output_size: List[int] - [height, width]
        '''
        self.output_size = output_size

    def __call__(self, sample):
        '''
            input:
              sample: Dict['video', 'caption'] - video (Time, Height, Width, Channel)
            return:
              sample: Dict['video', 'caption'] - video (Time, output_H, output_W, Channel)
        '''
        video, caption, video_file = sample['video'], sample['caption'], sample['video_file']
        clips = [np.expand_dims(cv2.resize(
            clip, self.output_size[::-1], interpolation=cv2.INTER_AREA), axis=0) for clip in video]
        video = np.concatenate(clips)
        return {'video': video, 'caption': caption, 'video_file': video_file}


class VideoToTensor:
    def __init__(self):
        pass

    def __call__(self, sample):
        '''
            input:
              sample: Dict['video', 'caption'] - video (Time, Height, Width, Channel)
            return:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
        '''
        video, caption, video_file = sample['video'], sample['caption'], sample['video_file']
        video = torch.from_numpy(video)
        return {'video': video, 'caption': caption, 'video_file': video_file}


class VideoRandomHorizontalFlip:
    def __init__(self):
        pass

    def call(self, sample):
        '''
            input:
              sample: Dict['video', 'caption'] - video (Time, Height, Width, Channel)
            return:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
        '''
        video, caption, video_file = sample['video'], sample['caption'], sample['video_file']
        flip = True if random.random > 0.5 else False

        if flip is True:
            video = video[:, :, :, ::-1]

        return {'video': video, 'caption': caption, 'video_file': video_file}


class VideoRandomCrop:
    def __init__(self, size: List[int]):
        '''
            Args:
              size: (h, w)
        '''
        self.size = size

    def __call__(self, sample):
        '''
            input:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
            return:
              sample: Dict['video', 'caption'] - videoTensor (Time, Croped Height, Croped Width, Channel)
        '''
        video, caption, video_file = sample['video'], sample['caption'], sample['video_file']
        t, h, w, c = video.size()
        dh, dw = self.size[0], self.size[1]
        sh = int(random.random() * (h - dh))
        sw = int(random.random() * (w - dw))

        video = video[:, sh:sh + dh, sw:sw + dw]
        return {'video': video, 'caption': caption, 'video_file': video_file}


class VideoCenterCrop:
    def __init__(self, size: List[int]):
        '''
            Args:
              size: (h, w)
        '''
        self.size = size

    def __call__(self, sample):
        '''
            input:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
            return:
              sample: Dict['video', 'caption'] - videoTensor (Time, Croped Height, Croped Width, Channel)
        '''
        video, caption, video_file = sample['video'], sample['caption'], sample['video_file']
        t, h, w, c = video.size()
        dh, dw = self.size[0], self.size[1]
        sh = round(h / 2) - round(dh / 2)
        sw = round(w / 2) - round(dw / 2)
        video = video[:, sh:sh + dh, sw:sw + dw, :]
        return {'video': video, 'caption': caption, 'video_file': video_file}


class VideoToFloat:
    def __init__(self):
        pass

    def __call__(self, sample):
        '''
            input:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
            return:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
        '''
        video, caption, video_file = sample['video'], sample['caption'], sample['video_file']
        video = video.float().div_(255.0)
        return {'video': video, 'caption': caption, 'video_file': video_file}


class VideoNormalize:
    def __init__(self, mean, std):
        '''
            inputs:
              mean: int
              std: int
        '''
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        '''
            input:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
            return:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
        '''
        video, caption, video_file = sample['video'], sample['caption'], sample['video_file']
        video = video.sub_(self.mean).div_(self.std)
        return {'video': video, 'caption': caption, 'video_file': video_file}


class VideoTranspose:
    def __init__(self):
        pass

    def __call__(self, sample):
        '''
            input:
              sample: Dict['video', 'caption'] - videoTensor (Time, Height, Width, Channel)
            return:
              sample: Dict['video', 'caption'] - videoTensor (Channel, Time, Height, Width)
        '''
        video, caption, video_file = sample['video'], sample['caption'], sample['video_file']
        video = video.permute([3, 0, 1, 2])
        return {'video': video, 'caption': caption, 'video_file': video_file}


def mvad_basic_transform():
    transform = VideoCompose([
        VideoResize((256, 256)),
        VideoToTensor(),
        VideoCenterCrop((224, 224)),
        VideoToFloat(),
        VideoNormalize(0.5, 0.5),
        VideoTranspose(),
    ])
    return transform


class FeaturePadding:
    def __init__(self, max_sequence_size: int):
        self.max_sequence_size = max_sequence_size

    def __call__(self, feature):
        seq_size, d_model = feature.size()
        padded_feature = torch.zeros((self.max_sequence_size, d_model))
        padded_feature[:seq_size] = feature

        return padded_feature


class CaptionPadding:
    def __init__(self, max_sequence_size: int, padding_id: int):
        self.max_sequence_size = max_sequence_size
        self.padding_id = padding_id

    def __call__(self, caption):
        new_caption = torch.zeros((self.max_sequence_size), dtype=torch.int)
        new_caption[:caption.size(0)] = caption
        return new_caption
