# -*- coding: utf-8 -*-
__author__ = 'Yaicky'

import sys, os
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import pickle

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

#计算回归准确性
'''平均绝对误差(mean absolute error)
给定数据集的所有数据点的绝对误差平均值
'''
print('Mean absolute error =', round(sm.mean_absolute_error(y_test, y_test_pred), 2))
# Mean absolute error = 0.54
'''均方误差(mean squard error)
给定数据集的所有数据点的绝对误差的平方的平均值，最流行的指标之一
'''
print('Mean squared error =', round(sm.mean_squared_error(y_test, y_test_pred), 2))
# Mean squared error = 0.38
'''中位数绝对误差(median absolute error)
给定数据集的所有数据点的的误差的中位数，优点是可以消除异常值(outlier)的干扰
'''
print('Median absolute error =', round(sm.median_absolute_error(y_test, y_test_pred), 2))
# Median absolute error = 0.54
'''解释方差分(explained variance score)
用于衡量模型对数据波动的解释能力，得分1.0，表示完美
'''
print('Explained variance score =', round(sm.explained_variance_score(y_test, y_test_pred), 2))
# Explained variance score = 0.68
'''R方得分(R2 score)
是指确定性相关系数，用于衡量模型对未知样本预测的效果，最好得分为1.0
'''
print('R2 score =', round(sm.r2_score(y_test, y_test_pred), 2))
# R2 score = 0.68

#保存数据模型
output_model_file = os.path.join(os.path.dirname(__file__), 'saved_model.pkl')
with open(output_model_file, 'wb+') as f:
    pickle.dump(linear_regressor, f)

with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print ('\nNew mean absolute error =', round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))