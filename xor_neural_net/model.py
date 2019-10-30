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
        
        self.layer1weight0 = torch.nn.Parameter(torch.randn(2))
        self.layer1weight1 = torch.nn.Parameter(torch.randn(2))
        self.layer2weight = torch.nn.Parameter(torch.randn(2))

        self.layer1bias0 = torch.nn.Parameter(torch.randn(1))
        self.layer1bias1 = torch.nn.Parameter(torch.randn(1))
        self.layer2bias = torch.nn.Parameter(torch.randn(1))

    def forward(self, x):
        h = torch.zeros(2)

        h[0] = x @ self.layer1weight0 + self.layer1bias0
        h[1] = x @ self.layer1weight1 + self.layer1bias1

        if h[0] < 0:
            h[0] *= 0.01
        if h[1] < 0:
            h[1] *= 0.01
        
        return h @ self.layer2weight + self.layer2bias

class NeuralNetUsingBuiltInModules(torch.nn.Module):
    def __init__(self):
        super(NeuralNetUsingBuiltInModules, self).__init__()

        self.l1 = torch.nn.Linear(2, 2)     # can easily increase number of nodes in hidden layer
        self.l2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        from torch.nn.functional import relu
        return self.l2(relu(self.l1(x)))

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

        if epoch % 143 == 0:
            print(loss)

for i in range(len(X)):
    prediction = classifier.forward(X[i])
    print("Data: ", X[i], "\tPrediction: ", prediction, "\tActual: ", Y[i])