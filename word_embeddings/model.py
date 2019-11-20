import torch
import torch.nn as nn
import torch.nn.functional as F

class YelpNet(nn.Module):
    def __init__(self, document_dim, embedding_dim):
        super(YelpNet, self).__init__()
        
        # TODO: add linear layers!

        raise NotImplementedError()

    def forward(self, x):
        """
        @param x    a tensor with shape (document_dim, embedding_dim) representing the review
        @return     a 1D tensor of length 1, element measures how positive or negative the review's
                    sentiment is.
        """

        # some useful functions:
        # torch.flatten     https://pytorch.org/docs/stable/torch.html#torch.flatten
        # F.relu            https://pytorch.org/docs/stable/nn.functional.html#relu
        # torch.exp         https://pytorch.org/docs/stable/torch.html#torch.exp
        # torch.sum         https://pytorch.org/docs/stable/torch.html#torch.sum

        raise NotImplementedError()