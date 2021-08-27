'''
数据的分割功能
'''
import random
import numpy as np

__all__ = ['HoldOut', 'StratifiedSampling']


def HoldOut(data: np.ndarray, persent=0.3) -> np.ndarray:
    """
    留一法
    :param data: 要进行留出法分离集合的数据
    :param persent: 训练集保留的数量
    :return: np.narray的类型
    """

    n = data.shape[0] * persent
    n = np.ceil(n)

    data_train = data[0:n]
    data_test = data[n:]
    return data_train, data_test


def StratifiedSampling(data: np.ndarray, persent=0.3) -> np.ndarray:
    """
    分层抽样
    :param data:
    :param persent: 训练集保留的数量
    :return: np.narray的类型
    """
    label_data_unique = np.unique(data[:, -1])  # 定义分层值域
    sample_data = []
    test_data = []
    sample_dict = {}
    for label_data in label_data_unique:
        sample_list = []
        for data_tmp in data:
            if data_tmp[-1] == label_data:
                sample_list.append(data_tmp)
        num = int(np.ceil(len(sample_list) * persent))
        each_sample_data = random.sample(sample_list, num)
        sample_data.extend(each_sample_data)
        sample_dict[label_data] = len(each_sample_data)
    data_train = np.array(sample_data)
    for i in data_train:
        b = 0
        for j in data:
            if (i == j).all():
                data = np.delete(data, b, axis=0)
            b += 1
    data_test = data
    return data_train, data_test


# 开始转变思路，先实现再创造
from sklearn.model_selection import KFold


def KFold_cai(data, k=2):
    '''
     def custom_cv_2folds(X):
...     n = X.shape[0]
...     i = 1
...     while i <= 2:
...         idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
...         yield idx, idx
...         i += 1
    有时间就随机获取下前面的index /斜眼笑
    :param data:
    :param k:
    :return:
    '''
    train = []
    test = []
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(data):
        train.append(data[train_index])
        test.append(data[test_index])
    return train, test


from sklearn.model_selection import RepeatedKFold


def RepeatedKFold_cai(data, k=2, n=2):
    train = []
    test = []
    kf = RepeatedKFold(n_splits=k, n_repeats=n)
    for train_index, test_index in kf.split(data):
        train.append(data[train_index])
        test.append(data[test_index])
    return train, test


from sklearn.model_selection import LeaveOneOut


def LeaveOneOut_cai(data):
    train = []
    test = []
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(data):
        train.append(data[train_index])
        test.append(data[test_index])
    return train, test


from sklearn.model_selection import LeavePOut


def LeavePOut_cai(data, n=2):
    train = []
    test = []
    lpo = LeavePOut(p=n)
    for train_index, test_index in lpo.split(data):
        train.append(data[train_index])
        test.append(data[test_index])
    return train, test


from sklearn.model_selection import ShuffleSplit


def ShuffleSplit_cai(data, n=2):
    """
    随机交叉验证
    :param data:
    :param n:
    :return:
    """
    train = []
    test = []
    ss = ShuffleSplit(n_splits=n, test_size=0.25)
    for train_index, test_index in ss.split(data):
        train.append(data[train_index])
        test.append(data[test_index])
    return train, test


from sklearn.model_selection import cross_val_score


def cross_val_score_cai(X, Y, model, sort='f1_macro', v=5):
    scores = cross_val_score(model, X, Y, cv=v, scoring=sort)

    return scores


from sklearn.model_selection import cross_validate

def cross_validate_cai(X, Y, model, v=5, scort=['precision_macro', 'recall_macro']):
    """
    from sklearn.metrics.scorer import make_scorer
scoring = {'prec_macro': 'precision_macro',
...            'rec_macro': make_scorer(recall_score, average='macro')}
 scores = cross_validate(clf, iris.data, iris.target, scoring=scoring,
...                         cv=5, return_train_score=True)
sorted(scores.keys())
['fit_time', 'score_time', 'test_prec_macro', 'test_rec_macro',
 'train_prec_macro', 'train_rec_macro']
scores['train_rec_macro']
array([0.97..., 0.97..., 0.99..., 0.98..., 0.98...])
    :param X:
    :param Y:
    :param model:
    :param v:
    :param scort:
    :return:
    """
    scores = cross_validate(model, X, Y, cv=v, scoring=scort)
    sorted(scores.keys())

    return scores

