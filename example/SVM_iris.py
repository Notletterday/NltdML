# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 19:04:46 2021

@author: 13056
"""
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
"""
data
DESCR
feature_names
filename
frame
target
target_names
"""
x = iris['data']
y = iris['target']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3)
"""
建立模型并使用
"""
clf = svm.SVC()
clf = clf.fit(x_train,y_train)
score = clf.score(x_test,y_test)
"""
查看支持向量
"""
support = clf.support_vectors_