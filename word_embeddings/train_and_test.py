from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# a function gets word embeddings for a given review
def get_embeddings(review, review_length):
    # Checkpoint: how do we get embeddings from the model?
    # What do we do if the length of the review is greater than review_length? 
    # Less than review_length?
    if len(review) > review_length:
        pass
    elif len(review) < review_length:
        # some useful functions:
        # np.zeroes(shape) https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
        # np.concatenate((a1, a2, ...)) https://docs.scipy.org/doc/numpy/reference/generated/numpy.concatenate.html
        pass

# create a column in our dataframe for word embeddings
df['embeddings'] = df['tokenized'].apply(lambda review: get_embeddings(review,  avg_review_length))

# split up our dataset into train and test sets using a handy sklearn function
X_train, X_test, y_train, y_test = train_test_split(df['embeddings'].values, df['pos_neg'].values, test_size=0.2)

# Checkpoint: how do we initialize our neural network?
# Go to model.py to finish implementing YelpNet!
net = YelpNet()

review = Variable(torch.Tensor(X_test[0]))
prediction = net.forward(review)

# Sanity check - what should shape of prediction be?
print(prediction)

optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

for i in range(len(X_train)):
    review = Variable(torch.Tensor(X_train[i]))
    correct_label = y_train[i]
    
    prediction = net.forward(review)

    # Checkpoint: what should our loss function be?
    loss = None

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Save model
torch.save(net, "trained_net.pb")

correct_count = 0
wrong_count = 0

for i in range(len(X_test)):
    review = torch.Tensor(X_test[i])
    correct_label = y_test[i]
    prediction = net.forward(review)

    # Checkpoint: how do we check whether the prediction is right?

print(correct_count, wrong_count)
print(correct_count / (correct_count + wrong_count))