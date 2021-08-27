'''
数据的预处理功能
'''
import numpy as np


__all__ = ['StandardScaler']
def StandardScaler(data):
    data = np.array(data)
    mean_ = data.mean()
    scale_ = data.std()
    var_ = data.var()
    return (data - mean_)/scale_, mean_,scale_,var_


