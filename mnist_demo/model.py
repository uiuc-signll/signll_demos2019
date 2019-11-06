import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 16)
        
        # TODO: add more linear layers

    def forward(self, x):
        """
        @param x    a tensor with shape (28, 28) representing the image
        @return     a 1D tensor of length 10, where the ith element is the predicted probability that
                    the image depicts the digit i. Note that since the elements are probabilities,
                    all elements should be positive and they should sum to 1.
        """

        # some useful functions:
        # torch.flatten     https://pytorch.org/docs/stable/torch.html#torch.flatten
        # F.relu            https://pytorch.org/docs/stable/nn.functional.html#relu
        # torch.exp         https://pytorch.org/docs/stable/torch.html#torch.exp
        # torch.sum         https://pytorch.org/docs/stable/torch.html#torch.sum

        pass