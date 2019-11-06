import torch
import utils
from model import DigitNet

net = DigitNet()

optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

for imageMatrix, correctLabel in utils.trainloader:
    imageMatrix = imageMatrix.reshape(28, 28)

    probabilities = net.forward(imageMatrix)
    
    # assertions for sanity checking the output
    for prob in probabilities:
        assert prob >= 0
    assert 0.9999 < torch.sum(probabilities) < 1.0001

    print(probabilities[correctLabel])
    # if this is close to 1, our neural net did a good job
    # if it is close to 0, our neural net should be punished!

    loss = 0 # TODO: what should the loss be?

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
            
torch.save(net, "trained_net.pb")