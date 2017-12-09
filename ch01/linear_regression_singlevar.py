# -*- coding: utf-8 -*-
__author__ = 'Yaicky'

import sys, os
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

filename = os.path.join(os.path.dirname(__file__), "data_singlevar.txt")
print(filename)
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        y.append(yt)

num_training = int(0.8 * len(X))
num_test = len(X) - num_training

#训练数据
X_train = np.array(X[: num_training]).reshape((num_training, 1))
y_train = np.array(y[: num_training])
print('X_train')
print('X_train_shape', X_train.shape)
print('y_train')

#测试数据
X_test = np.array(X[num_training:]).reshape((num_test, 1))
y_test = np.array(y[num_training:])

#创建线性回归对象
linear_regressor = linear_model.LinearRegression()

#用训练数据集训练数据
linear_regressor.fit(X_train, y_train)

y_train_pred = linear_regressor.predict(X_train)
plt.figure()
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, y_train_pred, color='black', linewidth=4)
plt.title('Training data')
plt.show()

y_test_pred = linear_regressor.predict(X_test)
plt.figure()
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.title('Test data')
plt.show()