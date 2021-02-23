import torch.nn as nn
import torch.nn.functional as F


class DenseCrossEntropy(nn.Module):
    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        log_probs = F.log_softmax(logits, dim=-1)
        loss = -labels * log_probs
        loss = loss.sum(-1)
        return loss.mean()


def get_loss_fn():
    return DenseCrossEntropy()
