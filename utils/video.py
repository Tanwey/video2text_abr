import cv2
import numpy as np


def read_video(video_file, rgb=True):
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

    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == len(video_frames):
        print('All read')
    video = np.concatenate(video_frames, axis=0)
    cap.release()

    return video


def video_info(video_file):
    cap = cv2.VideoCapture(video_file)
    print('Frame Count: {}'.format(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print('Frame Height: {}, Width: {}'.format(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print('Frame Count: {}'.format(cap.get(cv2.CAP_PROP_FPS)))
    cap.release()

