# +++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Implementing Linear Regression from Scratch           #
# Prediction                                            #                     
# Author: Jianhong Chen                                 #
# Date: 06.24.2021                                      #
#++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

library(tidyverse)
library(matrixStats)


# ---------------------------- Model Modules -------------------------------

SyntheticData = function(w, num_examples){
  ## Generate y = Xw + b + noise 
  ## The intercept param, b,  is embedded into w as the first parameter!!!
  
  # NOTE set row-wise here to match with Python rowwise notation
  X = matrix(rnorm(num_examples * length(w), mean = 0, sd = 1),
             nrow = num_examples, ncol = length(w), byrow = TRUE)
  # set the first column into 1 for the intercept parameter
  X[, 1] = 1

  y = X %*% w  # matrix multiplication 
  y = y + rnorm(nrow(y), mean = 0, sd = 0.01) # add some noise
  
  ret = data.frame(y, X)
  
  return(ret)
  
}

LinReg = function(X, w){
  ## Linear Regression equation
  ## The intercept param, b,  is embedded into w as the first parameter!!!
  ret = X %*% w
  return(ret)
}

SquaredLoss = function(y_hat, y){
  ## Squared Loss of each iteration
  ret = (y_hat - y ) ^2 / 2
  return(ret)
}

SGD = function(X, y, w, lr, batch_size){
  ## minibatch stochastic gradient descent
  
  # Minibatch sampling:
  batch_ind = sample(nrow(X), batch_size, replace = FALSE)
  X_batch = X[batch_ind, ]
  y_batch = y[batch_ind, ]
  # Compute gradient for the batch sample:
  w_grad = matrix( (LinReg(X_batch, w) - y_batch), nrow = 1 ) # reshape for matrix operation 
  w_grad = matrix(w_grad %*% X_batch, ncol = 1) # reshape to col-vector

  # Now update w & b
  w = w - lr/batch_size * w_grad
  
  return(w)
  
}


# ---------------------------- Implementing the Model -------------------------------

## True parameters:
# note: the first parameter in w is the intercept
true_w = matrix(c(4.2, 2, -3.4), ncol = 1)
num_examples = 1000

dat = SyntheticData(true_w,  num_examples)
# VIZ synthetic data
dat %>% 
  ggplot(aes(x = X3, y = y)) +
  geom_point()

# initiate model parameters:
X = as.matrix(dat[, -1]) 
y = as.matrix(dat[, 1])

b = 0 
w = matrix(c(b, rnorm(2, mean = 0, sd = 0.01) ), ncol = 1)

batch_size = 125
lr = 0.5

num_epoch = 20
loss = SquaredLoss
net = LinReg

## -- Train the model:
for (epoch in 1:num_epoch ){
  # update parameters w:
  w = SGD(X, y, w, lr, batch_size)
  y_hat = net(X, w)
  l = loss(y_hat, y)
  
  print(str_c('epoch ', epoch, ' Loss ', mean(l) ))
  
}

print( str_c('True W: ', true_w, ' Optimized W: ', w))

# ------------------------------- Prediction -----------------------------------
y_pred = net(X, w)
# compute MAE and RMSE:
( mae = mean(abs(y - y_pred)) )
( rmse =  sqrt(mean((y-y_pred)^2)) )

data.frame(y, y_pred) %>%
  ggplot(aes(x = y, y = y_pred)) +
  geom_point() + 
  geom_abline(intercept = 0, slope = 1, lty = 2, color = 'red')


# ----------------------- Compare Built-in Function -------------------------------

mod_df = dat %>%
  dplyr::select(-X1) # remove the dummpy 1 column

mod = lm(y ~., data = mod_df)

print(mod$coefficients)
print(w)




