import random
import torch
import numpy as np


def setup_random(seed = 428):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_bernoulli(p):
    """
      Generates a single Bernoulli random number.

      Args:
        p: The probability of success (1).

      Returns:
        0 or 1, representing failure or success, respectively.
      """
    return np.random.binomial(n=1, p=p)
