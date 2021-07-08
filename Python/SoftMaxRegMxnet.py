'''
Implement Softmax Regression using Mxnet and Gluon Framework

Author: Jianhong Chen
Date: 07.06.2021

'''

# %matplotlib inline

from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
from mxnet import init
from mxnet import gluon
from d2l import mxnet as d2l


npx.set_np() # numpy set up

# Initiate model parameters:
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# Build the model
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma = 0.01))

loss = gluon.loss.SoftmaxCrossEntropyLoss()

# Optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})


## Training the model:
num_epochs = 10
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)