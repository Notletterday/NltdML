'''
数据的预处理功能
'''
import numpy as np


__all__ = ['StandardScaler']
def StandardScaler(data):
    data = np.array(data)
    mean_ = data.mean(axis=0)
    scale_ = data.std(axis=0)
    var_ = data.var(axis=0)
    return (data - mean_)/scale_, mean_,scale_,var_


