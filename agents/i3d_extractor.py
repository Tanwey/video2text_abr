from agents.base import BaseFeatureExtractor
from graph.models.i3d import InceptionI3d


class InceptionI3dFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        model = InceptionI3d()
        config.
        super(InceptionI3dFeatureExtractor, self).__init__()
