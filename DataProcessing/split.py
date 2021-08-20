'''
数据的分割功能
'''

import numpy as np

__all__ = ['HoldOut', 'StratifiedSampling']


def HoldOut(data: np.ndarray, persent=0.3) -> np.ndarray:
    """
    :param data: 要进行留出法分离集合的数据
    :param persent: 训练集保留的数量
    :return: np.narray的类型
    """

    n = data.shape[0] * persent
    n = np.ceil(n)

    data_train = data[0:n]
    data_test = data[n:]
    return data_train, data_test


def StratifiedSampling(data: np.ndarray, split_type=0, name=None, col=0, n=0, persent=0.3) -> np.ndarray:
    # 因为用的numpy暂时只对数字的范围进行分类
    if not (name != 0 & split_type == 1 & n != 0) | (col != 0 & split_type == 0 & n == 0):
        # 这里还需要改一下条件
        raise Exception("请确认参数是否正确")
    max_result = np.max(data, axis=col)
    num = np.ceil(max_result.max()/n)
    #这里可以取三分之一点的数，结果放了numpy里，以后要用每一列
