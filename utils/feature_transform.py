import torch


class FeaturePadding:
    def __init__(self, max_sequence_size, pad_value=0, cut_sequence=False):
        """Pad to feature
        Args:
            max_sequence_size (int): Sequence size after padding
            cut_sequence (bool): If True, when feature sequence is longer than max_sequence, cut the feature.
                If False, when feature sequence is longer than max_sequence, alert error.
        """
        self.max_sequence_size = max_sequence_size
        self.pad_value = pad_value
        self.cut_sequence = cut_sequence

    def __call__(self, feature):
        """
        Args:
            feature (Tensor[seq, d_model])
        Returns:
            padded_feature (Tensor[max_sequence_size, d_model])
        """
        seq_size, d_model = feature.size()
        if self.cut_sequence is True:
            if seq_size > self.max_sequence_size:
                seq_size = self.max_sequence_size
        else:
            assert seq_size <= self.max_sequence_size

        padded_feature = torch.full(
            (self.max_sequence_size, d_model), self.pad_value)
        padded_feature[:seq_size] = feature

        return padded_feature
