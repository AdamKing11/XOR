from __future__ import print_function

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

def random_grow(d, rows_to_grow=1000):
	"""
	not necessary for the ML, just takes in a pandas data frame and copies
	data in it randomly to make a bigger data set of more or less the same
	distribution of data
	"""
	l = d.shape[0]	# get original number of rows
	for i in range(rows_to_grow):
		r = np.random.randint(0,l)	# random int between 0 and l
		d = d.append(d[r:r+1], ignore_index=True)	# copy row l
	return d

# specify a random seed for reproducability
#np.random.seed(1234)

df = pd.read_csv("data_for_xor.csv", sep='\t')
df = random_grow(df, 100)		# take our original data file and make it bigger

# change the type of the data in the data frame to a numpy array
# tip: use as SMALL of a data type as you need; big data can use up memory real
# quick so you want to be as efficient as possible
df = df.as_matrix().astype('int8')	

# randomize the order of the data
np.random.shuffle(df)

# we specify how we want to break up the data into testing and training
# specify here the test/train ratio
ttr = .8
ttr = int(ttr * len(df))

# we need to build our training and test data, as well as split it into
# inputs (X) and responses (y)
train_X = df[:ttr,:-1]
train_y = df[:ttr,-1]
test_X = df[ttr:,:-1]
test_y = df[ttr:,-1]


hidden_size = 8
print("Training size:", len(train_X), "\n\tTesting size:", len(test_X))
print("Number of hidden states:", hidden_size)
# now we build the model!
# we first define the type of model, in this case sequential
# which means each layer feeds into the others sequentially (duh)
model = Sequential()

# this is our first layer
# it's a fully connected layer with 4 hidden nodes
# for the first layer you NEED to specify the input dimensions
# in this case, a 2-long vector for XOR
model.add(Dense(hidden_size, input_dim = 2, activation="sigmoid"))

# second and final layer, because we are predicting a 1-long vector
# the number of nodes is 1 - which is the same dimensions as our
# y
model.add(Dense(1, activation="sigmoid"))

# now we compile it, specify a loss function and an optimizer
model.compile(loss="mean_squared_error",optimizer="adam", metrics=["accuracy"])
# now the model is complete! time to train

# time to train!
# we specify the X and the y for training as the first 2 arguments
# then we tell it how many epochs to do and the batch size
model.fit(train_X, train_y, nb_epoch=2000, batch_size = 10,verbose = 1)

# now we feed in our test data and see how well the model did
score = model.evaluate(test_X, test_y)
print("\n\nloss, accuracy:", score[0], score[1])

# we'll now see the EXACT predictions of the model on some of the test data
# because our outputs are either 0 or 1, this will generate a float between 0
# and 1. the model should be pretty accurate so we'll get values VERY close
# to exactly 0 or 1
for i in range(5):
	guess_X = test_X[i:i+1,:]
	guess_Y = model.predict_proba(guess_X, verbose = 0)
	print(guess_X, "\t", guess_Y)

# cool thing is we can try it will non integer values
# because the inputs won't be exactly like the training the data, the model
# will be a little less sure of the response
nonint_guess = np.array([[.8,.1]])
print(nonint_guess, model.predict_proba((nonint_guess), verbose = 0))