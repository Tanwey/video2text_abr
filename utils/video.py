import cv2
import numpy as np
from easydict import EasyDict
import os


def read_video(video_file, rgb=True):
    """
    Args:
        video_file (str):
        rgb (bool, default: True): If False, returns BGR type video (usually used in opencv)
    Returns:
        video (ndarray(Time, Height, Width, Channel))
    """
    cap = cv2.VideoCapture(video_file)
    video_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            if rgb is True:
                video_frames.append(np.expand_dims(frame[:, :, ::-1], axis=0))
            elif rgb is False:
                video_frames.append(np.expand_dims(frame, axis=0))
        else:
            break

    video = np.concatenate(video_frames, axis=0)
    cap.release()

    return video


def video_info(video_file, verbose=False):
    """
    Args:
        video_file (str)
        verbose (bool, default: False): If True, print info
    Returns:
        info (EasyDict['count', 'height', 'width', 'fps']): Information about video
            Keys:
                count: count of frames
                height: height of video
                width: width of video
                fps: frame per second of video
    """
    cap = cv2.VideoCapture(video_file)
    info = EasyDict({
        'count': cap.get(cv2.CAP_PROP_FRAME_COUNT),
        'height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'fps': cap.get(cv2.CAP_PROP_FPS)
    })
    if verbose is True:
        print('Frame Count: {}'.format(info.count))
        print('Frame Height: {}, Width: {}'.format(info.height, info.width))
        print('Frame FPS: {}'.format(info.fps))
    cap.release()
    return info


def save_video(video, save_path, fps=29.983304595341707, overwrite=False):
    """
    Args:
        video (ndarray[Time, Height, Width, Channel])
        save_path (str)
        fps (float): Frames per second of video
        overwrite (bool, default: False): If True, Overwrite file, although file exists.
    """
    if overwrite is False:
        if os.path.exists(save_path) is True:
            print('File already exists')
            return
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps,
                          (video.shape[2], video.shape[1]))

    for frame in video[:, :, :, ::-1]:
        out.write(frame)
    out.release()
