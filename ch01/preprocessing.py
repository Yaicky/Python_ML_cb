# -*- coding: utf-8 -*-
__author__ = 'Yaicky'

import numpy as np
from sklearn import preprocessing

data = np.array([[ 3, -1.5,  2, -5.4],
                 [ 0,  4,  -0.3, 2.1],
                 [ 1,  3.3, -1.9, -4.3]])


'''均值移除 z-score标准化

z=(x-μ)/σ。其中x为某一具体分数，μ为平均数，σ为标准差。(每个属性，每列进行求）

'''
data_standardized = preprocessing.scale(data)
print(data_standardized)
print('\nMean =', data_standardized.mean(axis=0))
print('Std deviation =', data_standardized.std(axis=0))

#output：
# [[ 1.33630621 -1.40451644  1.29110641 -0.86687558]
#  [-1.06904497  0.84543708 -0.14577008  1.40111286]
#  [-0.26726124  0.55907936 -1.14533633 -0.53423728]]
#
# Mean = [  5.55111512e-17  -1.11022302e-16  -7.40148683e-17  -7.40148683e-17]   ≈ [0, 0, 0, 0]
# Std deviation = [ 1.  1.  1.  1.]

'''范围缩放 最小最大值标准化

每列max~1 min~0, x_std = x-x.min(axis=0) / x.max(axis=0)-x.min(axis=0)

每列x的值-该列最小值，除以该列最大-最小

'''
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data_scaled = data_scaler.fit_transform(data)
print("\nMin max scaled data =", data_scaled)
#output
# Min max scaled data = [[ 1.          0.          1.          0.        ]
#  [ 0.          1.          0.41025641  1.        ]
#  [ 0.33333333  0.87272727  0.          0.14666667]]

'''归一化 正则化

 p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
 
 l1 x / x.sum(axis=1)
 
 每个元素除以该行的上述p范数计算公式
 
'''
data_normalized = preprocessing.normalize(data, norm='l1')
print('\nL1 normalized data =', data_normalized)
#output
# L1 normalized data = [[ 0.25210084 -0.12605042  0.16806723 -0.45378151]
#  [ 0.          0.625      -0.046875    0.328125  ]
#  [ 0.0952381   0.31428571 -0.18095238 -0.40952381]]

'''二值化

if x>threshold x=1 else x=0

'''
data_binarized = preprocessing.Binarizer(threshold=1.4).transform(data)
print('\nBinarized data =', data_binarized)
#output
# Binarized data = [[ 1.  0.  1.  0.]
#  [ 0.  1.  0.  1.]
#  [ 0.  1.  0.  0.]]

'''独热编码 one-hot-encoding

example：性别：["male"，"female"]
地区：["Europe"，"US"，"Asia"]
浏览器：["Firefox"，"Chrome"，"Safari"，"Internet Explorer"]

["male"，"US"，"Internet Explorer"]，male则对应着[1，0]，同理US对应着[0，1，0]，Internet Explorer对应着[0,0,0,1]。则完整的特征数字化的结果为：[1,0,0,1,0,0,0,0,1]。

'''
encoder = preprocessing.OneHotEncoder()
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]])
encoder_vector = encoder.transform([[2, 3, 5, 3]]).toarray()
print('\nEncoded vector =', encoder_vector)
#output
# Encoded vector = [[ 0.  0.  1.  0.  1.  0.  0.  0.  1.  1.  0.]]
# 第一特征(列):[0, 1, 2, 1] = [0, 1, 2]
# 第二特征(列):[2, 3, 3, 2] = [2, 3]
# 第三特征(列):[1, 5, 2, 4] = [1, 2, 4, 5]
# 第四特征(列):[12, 3, 12, 3] = [3, 12]
# [2, 3, 5, 3] = [0, 0, 1|(2), 0, 1|(3), 0, 0, 0, 1|(5), 1, 0|(3)]
