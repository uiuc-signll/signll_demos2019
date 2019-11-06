import torch
import matplotlib.pyplot as plt
import utils

net = torch.load("trained_net.pb")

correctCount = 0
wrongCount = 0

for imageMatrix, correctLabel in utils.testloader:
    imageMatrix = imageMatrix.reshape(28, 28)
    probabilities = net.forward(imageMatrix)
    prediction = torch.argmax(probabilities)

    if prediction == correctLabel:
        correctCount += 1
    else:
        wrongCount += 1

    # code below can actually display images so you can see
    # the predictions on individual images!
    
    # print("Prediction", prediction)
    # plt.imshow(imageMatrix)
    # plt.show()

print(correctCount, wrongCount)