import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.linear1 = nn.Linear(28 * 28, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 10)

    def forward(self, x):
        """
        @param x    a tensor with shape (28, 28) representing the image
        @return     a 1D tensor of length 10, where the ith element is the predicted probability that
                    the image depicts the digit i. Note that since the elements are probabilities,
                    all elements should be positive and they should sum to 1.
        """
        x = x.flatten()
        hidden1 = F.relu(self.linear1(x))
        hidden2 = F.relu(self.linear2(hidden1))
        lastlayer = self.linear3(hidden2)

        allPositiveNums = torch.exp(lastlayer)
        probabilities = allPositiveNums / torch.sum(allPositiveNums)
        return probabilities