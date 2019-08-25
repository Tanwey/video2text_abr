import torch
import numpy as np

from graph.models.i3d import InceptionI3d
import utils.video_transforms as video_transforms


class I3dResizeAndCrop:
    def __init__(self, return_tensor=False):
        """Resizing and Cropping video
        I3D use video that resize to 256 preserving ratio and crop to (224, 224)
        Args:
            return_tensor (bool, default: False): If False it returns ndarray[Time, Height, Width, Channel]
            and if True it returns Tensor[Channel, Time, Height, Width]
        """
        self.return_tensor = return_tensor
        transforms = [
            video_transforms.VideoToTensor(),
            video_transforms.VideoResizePreserve(256),
            video_transforms.VideoCenterCrop((224, 224)),
        ]
        if return_tensor is False:
            transforms.append(video_transforms.VideoToNumpy())
        self.transform = video_transforms.VideoCompose(transforms)

    def __call__(self, video):
        """
        Args:
            video (ndarray[Time, Height, Width, Channel])
        Returns:
            video (ndarray[Time, Height, Width, Channel])
            # If return_tensor is True (Tensor[Channel, Time, Height, Width])
        """
        video = self.transform(video)
        return video


class I3dTV_L1:
    def __init__(self):
        pass


class I3dRGBExtractor:
    def __init__(self, pretrained_weights_file, device=None):
        """RGB Feature Extractor with pretrained I3D
        Use extract method for extraction from rgb video
        Args:
            pretrained_weights_file (str): File path of I3D rgb weights
                (imagenet, kinetics, charades)
            device (torhc.device, default: None): Device for model
        """

        if device is None:
            device = torch.device('cuda')
        self.device = device

        weights = torch.load(pretrained_weights_file)
        self.model = InceptionI3d(final_endpoint='Mixed_5c', in_channels=3)
        self.model.load_state_dict(weights, strict=False)
        self.model = self.model.to(device).eval()

    def extract(self, video):
        """
        Args:
            video (Tensor[1, Channel, Time, Height, Width] or Tensor[Channel, Time, Height, Width]):
                Video to extract. Height and Width must be 224 and Channel must be 3
        Returns:
            rgb_feature (Tensor[seq, 1024]): Extracted RGB feature. Seq would be (Time // 8)
        """
        if video.dim() == 4:
            video = video.unsqueeze(0)
        assert video.size(0) == 1
        assert video.size(1) == 3
        assert video.size(3) == 224
        assert video.size(4) == 224
        video = video.to(self.device)
        rgb_feature = self.model.extract_features(video).squeeze(0).transpose(0, 1)
        return rgb_feature


class I3dFlowExtractor:
    def __init__(self, pretrained_weights_file, device=None):
        """Flow Feature Extractor with pretrained I3D
        Use extract method for extraction from optical flow
        Args:
            pretrained_weights_file (str): File path of I3D flow weights
                (imagenet, kinetics, charades)
            device (torhc.device, default: None): Device for model
        """
        if device is None:
            device = torch.device('cuda')
        self.device = device

        weights = torch.load(pretrained_weights_file)
        self.model = InceptionI3d(final_endpoint='Mixed_5c', in_channels=2)
        self.model.load_state_dict(weights, strict=False)
        self.model = self.model.to(device).eval()

    def extract(self, flow):
        """
        Args:
            video (Tensor[1, Channel, Time, Height, Width] or Tensor[Channel, Time, Height, Width]):
                Video to extract. Height and Width must be 224 and Channel must be 2.
        Returns:
            flow_feature (Tensor[seq, 1024]): Extracted RGB feature. Seq would be (Time // 8)
        """
        if flow.dim() == 4:
            flow = flow.unsqueeze(0)
        assert flow.size(0) == 1
        assert flow.size(1) == 2
        assert flow.size(3) == 224
        assert flow.size(4) == 224
        flow = flow.to(self.device)
        flow_feature = self.model.extract_features(flow).squeeze(0).transpose(0, 1)
        return flow_feature
