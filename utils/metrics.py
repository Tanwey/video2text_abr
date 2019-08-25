import torch


def accuracy_metrics(predict, target, pad_id):
    """
    Args:
        predict (Tensor[batch, seq, vocab_size])
        target (Tensor[batch, seq]): Sparse tensor
    Returns:
        accuracy (int): Accuracy from 0 to 100 (unit: % percentage)
    """
    matched_matrix = (torch.max(predict, dim=-1)[1] == target)
    mask = (target != pad_id) | (predict != pad_id)
    matched_matrix = matched_matrix * mask
    acc = matched_matrix.type(torch.float).sum() / mask.type(torch.float).sum()
    return acc * 100.0


def bleu_metrics(predict, target):
    """
    """
