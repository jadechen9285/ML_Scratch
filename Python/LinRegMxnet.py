'''
Implement Linear Regression using Mxnet and Gluon Framework

Author: Jianhong Chen
Date: 06.28.2021

'''

# %matplotlib inline

from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon
from d2l import mxnet as d2l


npx.set_np() # numpy set up

# -------------------------------------- Helper Functions ----------------------------

def load_array(data_arrays, batch_size, is_train = True):
	""" Construct a Gluon data iterator. """
	dataset = gluon.data.ArrayDataset(*data_arrays)
	return gluon.data.DataLoader(dataset, batch_size, shuffle = is_train)


## -- define true parameters 
true_w = np.array([2, -3.4])
true_b = 4.2

## -- Initiate data
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


## Data iteration
batch_size = 10
data_iter = load_array((features, labels), batch_size)
next(iter(data_iter))


## -- Construct the model in gluon framework:
net = nn.Sequential()
net.add(nn.Dense(1)) # note 1-layer for Linear regression

## -- Initializaing Model Parameters
net.initialize(init.Normal(sigma = 0.01))

## Define the Loss Function
loss = gluon.loss.L2Loss()

## Optimization Algorithm:
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

## -- training the model
num_epochs = 3

for epoch in range(num_epochs):
	for X, y in data_iter:
		with autograd.record():
			l = loss(net(X), y)
		l.backward()
		trainer.step(batch_size)
	l = loss(net(features), labels)
	print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')

## -- Collect the final parameters: weights
w = net[0].weight.data()
b = net[0].bias.data()
print(w)
print(b)

