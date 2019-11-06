import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import itertools


trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                    transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)


testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                    transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)
plt.style.use('grayscale')
