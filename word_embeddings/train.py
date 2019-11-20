from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from model import YelpNet

# load the word embeddings into the model from file
filename = 'word2vecSmall.bin.gz'
model = KeyedVectors.load_word2vec_format(filename, binary=True)

# load csv
df = pd.read_csv('../yelp_demo/yelp_reviews.csv')

# tokenize each review
df['tokenized'] = df['text'].apply(lambda x: x.split())

# filter out words that aren't in the word2vec model
df['tokenized'] = df['tokenized'].apply(lambda x: [word for word in x if word in model])
df['word_count'] = df['tokenized'].apply(lambda x: len(x))

# The dimensions to each input vector for our array need to match, so we'll use the 
# average review length as the dimension for the length of each review
#
# Each review of length greater than this average will be cut short, and each review
# that's shorter than this average will have padding added onto it.
avg_review_length = int(df['word_count'].mean())

print('average review length is ' + str(avg_review_length))

# split up our dataset into train and test sets using a handy sklearn function
X_train, X_test, y_train, y_test = train_test_split(df['tokenized'].values, df['pos_neg'].values, test_size=0.2)

# convert scalar y values into arrays where first element represents the probability that the review is positive,
# and the second element represents the probability of the review being negative.
y_train = np.array([np.array([1.0 if x > 0 else 0.0, 1.0 if x <= 0 else 0.0]) for x in y_train])
y_test = np.array([np.array([1.0 if x > 0 else 0.0, 1.0 if x <= 0 else 0.0]) for x in y_test])

# a function that returns an array of word embeddings for the given input list of reviews
# each list of embeddings in output list must contain review_length embeddings
def get_embeddings(reviews, review_length):
    review_embeddings = []

    for review in reviews:
        # Checkpoint: how do we get embeddings from the model?
        # What do we do if the length of the review is less than review_length?
        if len(review) > review_length:
            pass
        elif len(review) < review_length:
            # some useful functions:
            # np.zeroes(shape)
            # np.concatenate((a1, a2, ...))
            pass
    
    return review_embeddings

# get embeddings for train and test set
X_train_embeddings = get_embeddings(X_train, avg_review_length)
X_test_embeddings = get_embeddings(X_test, avg_review_length)

# Checkpoint: how do we initialize our neural network?
# Go to model.py to finish implementing YelpNet!
net = YelpNet()

review = Variable(torch.Tensor(X_test_embeddings[0]))
prediction = net.forward(review)

# Sanity check - what should shape of prediction be?
print(prediction)

optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

for i in range(len(X_train_embeddings)):
    review = Variable(torch.Tensor(X_train_embeddings[i]))
    correct_label = 0 if y_train[i][0] == 1 else 1
    
    prediction = net.forward(review)

    # Checkpoint: what should our loss function be?
    # hint: check the mnist_demo from a couple weeks ago!
    loss = None

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save model
torch.save(net, "trained_net.pb")

correct_count = 0
wrong_count = 0

for i in range(len(X_test_embeddings)):
    review = Variable(torch.Tensor(X_test_embeddings[i]))
    # Checkpoint: what is the correct label for X_test_embeddings[i]
    correct_label = None
    predictions = net.forward(review)

    # the model's prediction is the category with the max probability
    value, prediction = predictions.max(0)

    if prediction == correct_label:
        correct_count += 1
    else:
        wrong_count += 1

print(correct_count, wrong_count)
print(correct_count / (correct_count + wrong_count))