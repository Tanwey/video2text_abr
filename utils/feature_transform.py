import torch


class FeaturePadding:
    def __init__(self, max_sequence_size):
        """Pad to feature
        Args:
            max_sequence_size (int): Sequence size after padding
        """
        self.max_sequence_size = max_sequence_size

    def __call__(self, feature):
        """
        Args:
            feature (Tensor[seq, d_model])
        Returns:
            padded_feature (Tensor[max_sequence_size, d_model])
        """
        seq_size, d_model = feature.size()
        padded_feature = torch.zeros((self.max_sequence_size, d_model))
        padded_feature[:seq_size] = feature

        return padded_feature
