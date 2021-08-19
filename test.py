"""
第一次仿写，纯属娱乐，各位请不要用作商业用途。
作者：南巷旧梦
QQ：1305638814
"""
import numpy as np
from evaluation import train_test_split
import sklearn.model_selection
# 用于测试功能
if __name__ == '__main__':
    aa = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])
    print(type(aa))
    a, b = train_test_split.hold_out(aa)
    print(a, b)
    print(type(a))


