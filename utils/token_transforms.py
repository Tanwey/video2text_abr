import torch


class TokenPadding:
    def __init__(self, max_sequence_size, pad_id=0, cut_sequence=False):
        """Pad to token
        Args:
            max_sequence_size (int): Sequence size after padding
            pad_id (int, default=0): Token id of <PAD>
            cut_sequence (bool): If True, when feature sequence is longer than max_sequence_size, cut the feature.
                If False, when feature sequence is longer than max_sequence_size, alert error.
        """
        self.max_sequence_size = max_sequence_size
        self.pad_id = pad_id
        self.cut_sequence = cut_sequence

    def __call__(self, token):
        """
        Args:
            token (Tensor[seq])
        Returns:
            padded_token (Tensor[max_sequence_size])
        """
        seq_size = token.size(0)
        if self.cut_sequence is True:
            if seq_size > self.max_sequence_size:
                seq_size = self.max_sequence_size
        else:
            assert seq_size <= self.max_sequence_size

        padded_token = torch.full([self.max_sequence_size],
                                  self.pad_id, dtype=torch.int64)
        padded_token[:seq_size] = token[:seq_size]
        return padded_token
