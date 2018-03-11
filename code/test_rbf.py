import numpy as np
from plotutil import *

X_orig = np.loadtxt('X.txt')
y_orig = np.loadtxt('y.txt')

plot_data(X_orig, y_orig)

# Hyperparameters
num_of_iterations = 1000
learning_rate = 0.002
gauss_std = 0.1

# Train/Test set split
trainset_frac = 0.8
trainset_size = int(X_orig.shape[0] * trainset_frac)
X_train_orig = X_orig[ : trainset_size , : ]
X_test_orig = X_orig[ trainset_size : , : ]
X_train = data_transform(X_train_orig, 'rbf', std = gauss_std, target = X_train_orig)
X_test = data_transform(X_test_orig, 'rbf', std = gauss_std, target = X_train_orig)
y_train = y_orig[ : trainset_size ]
y_test  = y_orig[ trainset_size : ]

# Parameter vector - extra bias term at last entry
w = np.zeros(trainset_size + 1)

# Run Gradient Ascent
ll = np.zeros(num_of_iterations) # log-likelihood recording over epochs

for i in range(num_of_iterations):
    dw = np.dot(X_train.T, y_train - logistic(np.dot(X_train, w)))
    w += learning_rate * dw
    ll[i] = compute_average_ll(X_train, y_train, w)

# plot log-likelihood curve
plot_ll(ll)

# show predictions on test set
plot_predictive_distribution(X_orig, y_orig, w, 'pd-rbf.png', 'rbf', std = gauss_std, target = X_train_orig)

# final Train/Test set average log-likelihood
print ()
print ('Train set log-likelihood:', compute_average_ll(X_train, y_train, w))
print ('Test set log-likelihood:', compute_average_ll(X_test, y_test, w))

# evaluate accuracy of our model
y_test_hat = predict(X_test, w) > 0.5
y_merged = y_test + y_test_hat
true_neg_frac = np.count_nonzero(y_merged == 0) / np.count_nonzero(y_test == 0)
true_pos_frac = np.count_nonzero(y_merged == 2) / np.count_nonzero(y_test == 1)

# print confusion table
print ()
print ('Confusion Table:')
print ('   %6d%8d' % (0, 1))
print ('0  %6.4f%8.4f' % (true_neg_frac, 1 - true_neg_frac))
print ('1  %6.4f%8.4f' % (1 - true_pos_frac, true_pos_frac))

# print overall train/test accuracy

y_train_hat = predict(X_train, w) > 0.5
y_test_hat = predict(X_test, w) > 0.5
train_accuracy = np.count_nonzero(y_train_hat - y_train == 0) / y_train.shape[0]
test_accuracy = np.count_nonzero(y_test_hat - y_test == 0) / y_test.shape[0]
print ()
print ('Training accuracy = %.2f%%' % (train_accuracy * 100))
print ('Test accuracy = %.2f%%' % (test_accuracy * 100))