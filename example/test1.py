'''
入门练习1：波士顿房价预测
'''
import tensorflow as tf
from tensorflow import keras
import numpy as np
(x_train,y_train),(x_test,y_test) = keras.datasets.boston_load_data()
