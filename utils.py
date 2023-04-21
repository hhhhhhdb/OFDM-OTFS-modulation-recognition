import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def ensure_path(path):
    """The function to make log path.
    Args:
      path: the generated saving path.
    """
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)


class Averager():
    """The class to calculate the average."""
    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    """The function to calculate the .
    Args:
      logits: input logits.
      label: ground truth labels.
    Return:
      The output accuracy.
    """
    pred = F.softmax(logits, dim=1).argmax(dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return (pred == label).type(torch.FloatTensor).mean().item()



def compute_confidence_interval(data):
    """The function to calculate the .
    Args:
      data: input records
      label: ground truth labels.
    Return:
      m: mean value
      pm: confidence interval.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm
