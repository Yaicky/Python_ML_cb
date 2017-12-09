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

