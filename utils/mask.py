import torch


def create_padding_mask_from_size(size, real_size):
    '''
    Args:
        size: sequense length (int)
        real_size: real sequence length (int)
    Returns:
          mask: Mask Tensor (seq)
    assert size > real_size
    '''
    mask = torch.ones((size)).type(torch.bool)
    mask[:real_size] = 0
    return mask


def create_padding_mask_from_data(data):
    '''
    Args:
        data: Tensor (seq)
    Returns:
        mask: Mask Tensor (seq)
    '''
    mask = (data == 0).type(torch.bool)
    return mask


def create_look_ahead_mask(size):
    '''
    Args:
        size: Sequence(time) size of data
    Returns:
        mask: Upper triangular matrix with -1e9 (size, size)
    '''
    mask = torch.triu(torch.ones(size, size), 1)
    mask = mask.float().masked_fill(mask == 1, float(
        '-1e9')).masked_fill(mask == 0, float(0.0))
    return mask
