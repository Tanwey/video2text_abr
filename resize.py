from agents.i3d_extractor import I3dResize
from datasets.video_dataset import VideoDatasetFromDir
from utils.video import save_video
import os
dataset = VideoDatasetFromDir('data/ActivityNet200/video')
i3d_resizer = I3dResize()
# 0, 10000
# 10000, 19970
for i in range(1730, 10000):
    try:
        sample = dataset[i]
        video = sample['video']
        resized_video = i3d_resizer(video)
        
        file_name = os.path.split(sample['video_file'])[-1]
        save_path = os.path.join('data/ActivityNet200/video_resize', file_name)

        fps = sample['video_info']['fps']

        save_video(resized_video, save_path, fps)
    except MemoryError:
        print('Memory Error in {}'.format(i))
       
    except:
        print('Other Error in {}'.format(i))
    
    if (i + 1) % 10 == 0:
        print('{} resized'.format(i + 1))