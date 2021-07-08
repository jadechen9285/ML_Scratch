# +++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Implementing Softmax Regression from Scratch          #
# Prediction                                            #                     
# Author: Jianhong Chen                                 #
# Date: 07.08.2021                                      #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

library(tidyverse)
library(matrixStats)
library(tensorflow)
library(keras)

library(RColorBrewer)
library(ComplexHeatmap)


# ---------------------------- Load the Data -------------------------------
# Digit dataset
# mnist = dataset_mnist()
# X_train = mnist$train$x
# Y_train = mnist$train$y
# X_test = mnist$test$x
# Y_test = mnist$test$y

# Fashion Mnist:
fashion_mnist = dataset_fashion_mnist()
X_train  = fashion_mnist$train$x
Y_train = fashion_mnist$train$y
X_test = fashion_mnist$test$x
Y_test = fashion_mnist$test$y

fashion_key = c('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 
                'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

# VIZ the image:
PlotImage = function(image_array) {
  digit_color = colorRampPalette(rev(brewer.pal(n=9, 
                                                name = "Greys")))(100)
  Heatmap(image_array, name= 'pixels',
          col = digit_color,
          cluster_rows = F, cluster_columns = F,
          border = F, column_names_rot = 90
  )
}

for (i in 1:10) {
  image = X_train[i, , ]
  print(PlotImage(image))
  
}

# ---------------------------- Data Processing -------------------------------
## Flatten the 2D image into 1D features:
X_train_mat = matrix(0,  nrow = nrow(X_train), ncol = 28*28 )
for (i in 1:nrow(X_train)){
  # also normalize the pixel value by dividing 255
  X_train_mat[i, ] = matrix(X_train[i, , ], ncol = 28*28, byrow = TRUE) / 255
}

X_test_mat = matrix(0, nrow = nrow(X_test), ncol = 28*28 )
for (i in 1:nrow(X_test)){
  # also normalize the pixel value by dividing 255
  X_test_mat[i, ] = matrix(X_test[i, , ], ncol = 28*28, byrow = TRUE) / 255
}

## Add dummy col of value 1 for X for embedding the bias term:
X_train_mat = cbind(rep(1, nrow(X_train_mat)), X_train_mat)
X_test_mat = cbind(rep(1, nrow(X_test_mat)), X_test_mat)

## One-hot encoding for the Label:
Y_train_oh = to_categorical(Y_train)
Y_test_oh = to_categorical(Y_test)


# ---------------------------- Building the Models -------------------------------
Hypothesis = function(X, w) {
  ## Hypothesis function: Z = XW + b
  # Notice: b is already embedded within W as the first column
  return(X %*% w )
}

Softmax = function(Z){
  ## Softmax activation: exp(Z) / Row Summation (exp(Z)) 
  return( exp(Z) / rowSums(exp(Z)) )
}

CrossEntropy = function(Y_hat, Y_oh){
  ## Loss function: l(yhat, y) = -summation(yj log yhatj)
  # NOTEL Groud Truth Y need be one-hot encoded.
  # using the Ground Truth target as index to only select the nonzero class
  # [using normal element wise multiplication instead of matrix multiplication]
  return(-rowSums(log(Y_hat) * Y_oh))
}

ComputeAccuracy = function(Y_hat, Y){
  ## Compute the Accuracy
  # need to subtract 1 here because the target starts with 0!
  pred = apply(Y_hat, MARGIN = 1, FUN = function(x) which.max(x) - 1)
  acc = sum(pred == Y) / length(Y)
  return(acc)
}

SGD = function(X, Y_oh, w, lr, batch_size){
  ## minibatch stochastic gradient descent
  
  # Minibatch sampling:
  batch_ind = sample(nrow(X), batch_size, replace = FALSE)
  X_batch = X[batch_ind, ]
  Y_batch = Y_oh[batch_ind, ]
  # Compute gradient for the batch sample:
  Z_batch = Hypothesis(X_batch, w)
  Y_hat_batch = Softmax(Z_batch)
  
  # transpose X for correct dimension
  w_grad = t(X_batch) %*% (Y_hat_batch - Y_batch) 
  
  # Now update w
  w = w - lr/batch_size * w_grad
  
  return(w)
  
}


# ------------------------- Initiate Models Parameters ---------------------------
n = nrow(X_train_mat) # num of training data points
m = ncol(X_train_mat) # num of features
k = ncol(Y_train_oh) # num of target classes

# Dim(w) = num of features * num of Target Classes
# note: b is also embedded within W as the first column
w = matrix(rnorm(m*k, mean = 0, sd = 0.01), 
           nrow = m, ncol = k, byrow = TRUE )
#
batch_size = 512
lr = 0.3
num_epoch = 50
#
net = Hypothesis
active = Softmax 
loss = CrossEntropy
#
w_list = list()
l_list = list()
acc_list = list()

# ------------------------------ Train the models --------------------------------
for (epoch in 1:num_epoch ){
  # update parameters w:
  w = SGD(X_train_mat, Y_train_oh, w, lr, batch_size)
  # w = SGD_res$w
  # b = SGD_res$b
  Y_hat = active(net(X_train_mat, w))
  l = loss(Y_hat, Y_train_oh)
  acc = ComputeAccuracy(Y_hat, Y_train)
  
  ## Accumulate parameters, loss, and accuracy:
  w_list[[epoch]] = w
  l_list[[epoch]] = l
  acc_list[[epoch]] = acc
  
  print(str_c('epoch ', epoch, ' Loss ', mean(l), ' Accuracy ', acc ))
  
}

## separate b out: acc = 0.8686667


# ------------------------- Test the data on Keras ---------------------------
nn = keras_model_sequential()
nn %>%
  layer_dense(units = 10, input_shape = ncol(X_train_mat),
              activation = 'softmax')

nn %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'sgd',
  metrics = c('accuracy')
)

history = nn %>% fit(
  X_train_mat, Y_train_oh, 
  epochs = 50, batch_size = 512,
  validation_split = 0.2
)

## keras nn: 0.8986



