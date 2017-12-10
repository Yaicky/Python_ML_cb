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
# X_train = np.array(X[: num_training])
y_train = np.array(y[: num_training])
print('X_train')
print('X_train_shape', X_train.shape)
print('y_train')

#测试数据
X_test = np.array(X[num_training:]).reshape((num_test, 1))
# X_test = np.array(X[num_training:])
y_test = np.array(y[num_training:])

#创建线性回归对象和岭回归对象
linear_regressor = linear_model.LinearRegression()
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)

#用训练数据集训练数据
linear_regressor.fit(X_train, y_train)
ridge_regressor.fit(X_train, y_train)

#预测结果
y_test_pred = linear_regressor.predict(X_test)
y_test_pred_ridge = ridge_regressor.predict(X_test)

print ("LINEAR:")
print ("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2) )
print ("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2) )
print ("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2) )
print ("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2) )
print ("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

print ("\nRIDGE:")
print ("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2) )
print ("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2) )
print ("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2) )
print ("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2) )
print ("R2 score =", round(sm.r2_score(y_test, y_test_pred_ridge), 2))


#保存数据模型
output_model_file = os.path.join(os.path.dirname(__file__), 'saved_model_ridge.pkl')
with open(output_model_file, 'wb+') as f:
    pickle.dump(ridge_regressor, f)

with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)
