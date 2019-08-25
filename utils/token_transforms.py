import torch


class TokenPadding:
    def __init__(self, max_sequence_size, padding_token=0):
        """Pad to token
        Args:
            max_sequence_size (int): Sequence size after padding
            padding_token (int, default=0): Token id of <PAD>
        """
        self.max_sequence_size = max_sequence_size
        self.padding_token = padding_token

    def __call__(self, token):
        """
        Args:
            token (Tensor[seq, d_model])
        Returns:
            padded_token (Tensor[max_sequence_size, d_model])
        """
        new_token = torch.full((self.max_sequence_size, token.size(1)),
                               self.padding_token, dtype=torch.int64)
        new_token[:token.size(0)] = token
        return new_token
