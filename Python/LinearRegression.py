'''
Implement Linear Regression from Scrach 

Author: Jianhong Chen
Date: 06.24.2021

'''

# %matplotlib inline


import random
from mxnet import autograd, np, npx
from d2l import mxnet as d2l

npx.set_np() # numpy set up

# -------------------------------------- Helper Functions ----------------------------


## -- Generating the Dataset:
def synthetic_data(w, b, num_examples): #@save
	""" Generate y = Xw + b + noise. """
	X = np.random.normal(0, 1, (num_examples, len(w)))
	y = np.dot(X, w) + b
	y += np.random.normal(0, 0.01, y.shape) # add noise to the output
	return X, y.reshape((-1,  1))

## -- Sampling batch of data
def data_iter(batch_size, features, labels):
	num_examples = len(features)
	indices = list(range(num_examples))
	# shuffle the original indices to make the sampling random:
	random.shuffle(indices)
	for i in range(0, num_examples, batch_size):
		batch_indices = np.array(indices[i:min(i+batch_size, num_examples)])
		# yield will return a generator type object
		yield features[batch_indices], labels[batch_indices]

## -- Defining the Model
def linreg(X, w, b): 
	""" Linear Regression Equation """
	return np.dot(X, w) + b

## -- Define the Loss Function
def squared_loss(y_hat, y):
	''' Squared loss. '''
	return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

## Define the Optimization Algorithm: SGD
def sgd(params, lr, batch_size):
	''' Minibatch stochastic gradient descent. '''
	for param in params:
		param[:] = param - lr/batch_size * param.grad 



## -- define true parameters 
true_w = np.array([2, -3.4])
true_b = 4.2

## -- Initiate data
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

# VIZ the synthetic data
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].asnumpy(), labels.asnumpy(), 1)
# d2l.plt.show()

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
	print(X, '\n', y)
	break

## -- Initializaing Model Parameters:
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()
# hyper-parameters:
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

# train the model
for epoch in range(num_epochs):
	for X, y in data_iter(batch_size, features, labels):
		with autograd.record():
			l = loss(net(X, w, b), y) # Minibatch loss in 'X' and 'y'
		l.backward()
		sgd([w, b], lr, batch_size) # update parameters using their gradient
	train_l = loss(net(features, w, b), labels)

	print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'error in estimating w: {true_w - w.reshape(true_w.shape)}')
print(f'error in estimating b: {true_b - b}')


