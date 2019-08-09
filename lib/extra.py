import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms, utils, models

import matplotlib.pyplot as plt
import numpy as np

from functools import partial, reduce
from typing import List
from pytorch_i3d.pytorch_i3d import InceptionI3d
from datasetMVAD import MVADDataset
import os
from transforms import mvad_basic_transform
import time
from extract_feature import Extractor


if torch.cuda.is_available():
    device = torch.device('cuda:0')
    
i3d = InceptionI3d(400, end_module='Mixed_5c')
i3d.load_state_dict(torch.load('./pytorch_i3d/models/rgb_kinetics.pth'), strict=True)
i3d.to(device)

MVAD_PATH = '../MVAD'
MVAD_CORPUS_PATH = os.path.join(MVAD_PATH, 'corpus_M-VAD_train.txt')
MVAD_VIDEOS_PATH = os.path.join(MVAD_PATH, 'video')
transform = mvad_basic_transform()
train_dataset = MVADDataset(MVAD_VIDEOS_PATH, MVAD_CORPUS_PATH, train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
MVAD_TRAIN_SAVE_PATH = os.path.join(MVAD_PATH, 'I3D_rgb', 'train')
train_extractor = Extractor(i3d, MVAD_TRAIN_SAVE_PATH, gpu_device=device)
error_files = train_extractor.extract_feature_from_loader(train_loader)
print(error_files)