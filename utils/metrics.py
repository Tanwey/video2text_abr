import torch
from nltk.translate import meteor, bleu


def accuracy_batch(predict, target, pad_id=0):
    """
    Args:
        predict (Tensor[batch, seq])
        target (Tensor[batch, seq])
    Returns:
        accuracy (int): Accuracy from 0 to 100 (unit: % percentage)
    """
    matched_matrix = (predict == target)
    mask = (target != pad_id) | (predict != pad_id)
    matched_matrix = matched_matrix * mask
    acc = matched_matrix.type(torch.float).sum() / mask.type(torch.float).sum()
    return acc * 100.0


def bleu_batch(references_batch, hypothesis_batch):
    """Bilingual Evaluation Understudy(BLEU)
    Args:
        references_batch (list[list[list[str]]])
        hypothesis_batch (list[list[str]])
    Returns:
        bleu_score (double)
    """
    count = len(references_batch)
    assert count == len(hypothesis_batch)
    bleu_sum = 0
    for i in range(count):
        bleu_sum += bleu(references_batch[i], hypothesis_batch[i])
    return bleu_sum / count


def meteor_batch(references_batch, hypothesis_batch):
    """Metric for Evaluation of Translation with Explicit Ordering(METEOR)
    Args:
        references_batch (list[list[list[str]]])
        hypothesis_batch (list[list[str]])
    Returns:
        meteor_score (double)
    """
    count = len(references_batch)
    assert count == len(hypothesis_batch)
    meteor_sum = 0
    for i in range(count):
        meteor_sum += meteor(references_batch[i], hypothesis_batch[i])
    return meteor_sum / count


def rouge_metrics(predict, target):
    """Recall Oriented Understudy for Gisting Evaluation(ROUGE)
    """


def cider_metrics(predict, target):
    """Consensus based Image Description Evaluation(CIDEr)
    """
    pass


def wmd_metrics(predict, target):
    """Word Mover's Distance(WMD)
    """


def spice_metrics(predict, target):
    """Semantic Propositional Image Captioning Evaluation(SPICE)
    """
    pass
