import os

import PySide2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

"""
导入数据集
"""
iris = datasets.load_iris()  # 最近spyder坏了很难过
x = iris.data[:, -3:-1]
y = iris.target
h = .02
print(x)
print(np.size(x), np.size(y))
"""
这数据因为是预先准备好的所以不需要预处理,只简单把训练集与测试集分出来就好
"""
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
