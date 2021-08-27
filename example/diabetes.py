import os

import PySide2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from DataProcessing.PerformanceMeasure import MeanSquarederror, R2Score

dirname = os.path.dirname(PySide2.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

'''
导入糖尿病的数据集
'''
diabetes = datasets.load_diabetes()
'''
把数据分成训练集与验证集
'''
x = diabetes.data[:, 3]
x = x.reshape(442, 1)
y = diabetes['target']
x_train = x[:-20]
x_test = x[-20:]
y_train = y[:-20]
y_test = y[-20:]
'''
开始建立模型，并得出结果
'''
reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)
'''
用模型评估的标准去评估一下这个模型
'''
"""
这是我之前自己写的方法
print("平均标准误差: %.2f" % MeanSquarederror(y_test, y_pred))  
print('决定系数: %.2f' % R2Score(y_test,y_pred))
"""
print("平均标准误差: %.2f" % mean_squared_error(y_test, y_pred))
print('决定系数: %.2f' % r2_score(y_test, y_pred))
print(np.size(x_test), np.size(y_test))
'''
画图
'''
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_pred, color='blue')
plt.show()
