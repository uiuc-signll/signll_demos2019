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
        @return     a 1D tensor of length 2, where the 1st element is the predicted probability that
                    the review is positive, and the 2nd element is the predicted probability that
                    the  review is negative. Note that since the elements are probabilities,
                    all elements should be positive and they should sum to 1.
        """

        # some useful functions:
        # torch.flatten     https://pytorch.org/docs/stable/torch.html#torch.flatten
        # F.relu            https://pytorch.org/docs/stable/nn.functional.html#relu
        # torch.exp         https://pytorch.org/docs/stable/torch.html#torch.exp
        # torch.sum         https://pytorch.org/docs/stable/torch.html#torch.sum

        raise NotImplementedError()