import torch

X = torch.tensor([
    [+1, +1],
    [-1, +1],
    [+1, -1],
    [-1, -1]
], dtype=torch.float)

Y = [1, -1, -1, 1]

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        
        self.weight_vec = torch.nn.Parameter(torch.randn(2))
        self.bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.weight_vec @ x + self.bias

classifier = NeuralNet()

# the optimizer can take care of gradient descent for us (we set its learning rate to 0.01)
# SGD stands for stochastic gradient descent
optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)

for epoch in range(1000):
    for i in range(len(X)):
        prediction = classifier.forward(X[i])

        if Y[i] == 1:
            loss = torch.log(1 + torch.exp(-prediction))
        else:
            loss = torch.log(1 + torch.exp(prediction))

        # Sanity Check:
        # The loss should approach 0 as the predictor becomes more confident in the correct label
        # and approach infinity as the predictor becomes more confident in the wrong label

        loss.backward()         # calculate the gradients
        optimizer.step()        # take step down the gradient
        optimizer.zero_grad()   # reset the gradients to zero

for i in range(len(X)):
    prediction = classifier.forward(X[i])
    print("Data: ", X[i], "\tPrediction: ", prediction, "\tActual: ", Y[i])