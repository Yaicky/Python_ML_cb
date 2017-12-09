# -*- coding: utf-8 -*-
__author__ = 'Yaicky'
import numpy as np
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
input_classes = ['audi', 'ford', 'audi', 'toyota', 'ford', 'bmw']
label_encoder.fit(input_classes)

print('\nClass mapping:')
for i, item in enumerate(label_encoder.classes_):
    print(item, '-->', i)
#output
# Class mapping:
# audi --> 0
# bmw --> 1
# ford --> 2
# toyota --> 3

labels = ['toyota', 'ford', 'audi']
encoded_labels = label_encoder.transform(labels)
print('\nLabels =', labels)
print('\nEncoded labels =', list(encoded_labels))
#output
# Labels = ['toyota', 'ford', 'audi']
# Encoded labels = [3, 2, 0]

encoded_labels = [2, 1, 0, 3, 1]
decoded_labels = label_encoder.inverse_transform(encoded_labels)
print('\nEncoded labels =', encoded_labels)
print('\nDecoded labels =', decoded_labels)
#output
# Encoded labels = [2, 1, 0, 3, 1]
# Decoded labels = ['ford' 'bmw' 'audi' 'toyota' 'bmw']