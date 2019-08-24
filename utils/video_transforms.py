import torch
import torch
import cv2
import numpy as np
import random


class Compose:
    def __init__(self, transforms):
        """
        Args: 
            transforms (List[transform]): List of transform
        """
        self.transforms = transforms

    def __call__(self, video):
        for transform in self.transforms:
            video = transform(video)

        return video


class VideoCompose(Compose):
    pass


class VideoResize:
    def __init__(self, size):
        """Resize video
        When reduce size, it will apply INTER_AREA interpolation,
        and expand size, it will apply INTER_CUBIC interpolation
        Args:
            size (List[int]): Size to resize
        """
        self.size = size
        self.output_area = size[0] * size[1]

    def __call__(self, video):
        """
        Args:
            video (Tensor(Channel, Time, Height, Width))
        Returns:
            video (Tensor(Channel, Time, Height, Width))
        """
        area = video.size(2) * video.size(3)
        video = video.numpy().transpose(1, 2, 3, 0)
        if self.output_area < area:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC

        clips = [np.expand_dims(cv2.resize(
            clip, self.size[::-1], interpolation=interpolation), axis=0) for clip in video]
        print(clips[0].shape)
        video = np.concatenate(clips, axis=0).transpose(3, 0, 1, 2)
        print('concated')
        video = torch.from_numpy(video)
        return video


class VideoResizePreserve:
    def __init__(self, min_size:int = 256):
        """Resize video preserving ratio
        When reduce size, it will apply INTER_AREA interpolation,
        and expand size, it will apply INTER_CUBIC interpolation
        Args:
            min_size (int): 
        """
        self.min_size = min_size

    def __call__(self, video):
        """
        Args:
            video (Tensor(Channel, Time, Heigth, Width))
        Returns:
            video (Tensor(Channel, Time, Resized_Heigth, Resized_Width))
        """
        h = video.size(2)
        w = video.size(3)
        ratio = self.min_size / min(h, w)
        if ratio < 1.0:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_CUBIC
        video = video.numpy().transpose(1, 2, 3, 0)
        clips = [np.expand_dims(cv2.resize(
            clip, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=interpolation), axis=0) for clip in video]
        video = np.concatenate(clips, axis=0).transpose(3, 0, 1, 2)
        video = torch.from_numpy(video)
        return video


class VideoToTensor:
    def __init__(self):
        """Numpy array to Tensor + Transpose from (Time, Height, Width, Channel) to (Channel, Time, Height, Width)"""
        pass

    def __call__(self, video):
        """
        Args:
            video (ndarray[Time, Height, Width, Channel])
        Return:
            video (Tensor[Channel, Time, Height, Width])
        """
        video = torch.from_numpy(video)
        video = video.permute([3, 0, 1, 2])
        return video
    

class VideoHorizontalFlip:
    def __init__(self):
        """Flip video Horizontally"""
        pass

    def __call__(self, video):
        """
        Args:
            video (Tensor(Channel, Time, Height, Width))
        Return:
            video (Tensor(Channel, Time, Height, Width)) Flipped video
        """

        video = video.flip(3)

        return video




class VideoRandomHorizontalFlip(VideoHorizontalFlip):
    def __init__(self, ratio=0.5):
        """Randomly Flip video Horizontally
        Args:
            ratio (float, default=0.5): Ratio of Flipping
        """
        self.ratio = ratio

    def __call__(self, video):
        """
        Args:
            video (Tensor(Channel, Time, Height, Width))
        Return:
            video (Tensor(Channel, Time, Height, Width)) Randomly Flipped video
        """
        flip = True if random.random() < self.ratio else False

        if flip is True:
            video = super(VideoRandomHorizontalFlip, self).__call__(video)

        return video

class VideoRandomCrop:
    def __init__(self, size):
        """Randomly crop the video
        Args:
            size (List[int]): Size to crop (height, width)
        """
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (Tensor(Channel, Time, Height, Width))
        Return:
            video (Tensor(Channel, Time, Height, Width)): Randomly cropped video
        """
        c, t, h, w = video.size()
        dh, dw = self.size[0], self.size[1]
        sh = int(random.random() * (h - dh))
        sw = int(random.random() * (w - dw))

        video = video[:, :, sh:sh + dh, sw:sw + dw]
        return video


class VideoCenterCrop:
    def __init__(self, size):
        """Center crop the video
        Args:
            size (List[int]): Size to crop (height, width)
        """
        self.size = size

    def __call__(self, video):
        """
        Args:
            video (Tensor(Channel, Time, Height, Width))
        Returns:
            video (Tensor(Channel, Time, Height, Width)): Cropped video
        """
        c, t, h, w = video.size()
        dh, dw = self.size[0], self.size[1]
        sh = round(h / 2) - round(dh / 2)
        sw = round(w / 2) - round(dw / 2)
        video = video[:, :, sh:sh + dh, sw:sw + dw]
        return video


class VideoToFloat:
    def __init__(self):
        pass

    def __call__(self, video):
        """
        Args:
            video (Tensor(Time, Height, Width, Channel)): Video with dtype uint8 [0, 255]
        Returns:
            video (Tensor(Time, Height, Width, Channel)): Video with dtype float [0, 1)
        """
        video = video.float().div_(255.0)
        return video


class VideoNormalize:
    def __init__(self, mean, std):
        """
        Args:
            mean (int)
            std (int)
        """
        self.mean = mean
        self.std = std

    def __call__(self, video):
        """
        Args:
            video (Tensor)
        Returns:
            video (Tensor): (video - mean) / std
        """
        video = video.sub_(self.mean).div_(self.std)
        return video


class VideoTranspose:
    def __init__(self):
        pass

    def __call__(self, video):
        """
        Args:
            video (Tensor(Time, Height, Width, Channel))
        Returns:
            video (Tensor(Channel, Time, Height, Width))
        """
        video = video.permute([3, 0, 1, 2])
        return video


def mvad_basic_transform():
    transform = VideoCompose([
        VideoResizePreserve(256),
        VideoToTensor(),
        VideoCenterCrop((224, 224)),
        VideoToFloat(),
        VideoNormalize(0.5, 0.5),
        VideoTranspose(),
    ])
    return transform


def mvad_augment_transform():
    transform = VideoCompose([
        VideoResizePreserve(256),
        VideoToTensor(),
        VideoRandomHorizontalFlip(),
        VideoCenterCrop((224, 224)),
        #VideoRandomCrop((224, 224)),
        VideoToFloat(),
        VideoNormalize(0.5, 0.5),
        VideoTranspose(),
    ])
    return transform
