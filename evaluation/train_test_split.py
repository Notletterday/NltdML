import math
import numpy as np


def hold_out(data: np.ndarray, persent=0.3) -> np.ndarray:

    """
    :param data: 要进行留出法分离集合的数据
    :param persent: 训练集保留的数量
    :return: np.narray的类型
    """

    n = data.shape[0] * persent
    n = math.ceil(n)

    data_train = data[0:n]
    data_test = data[n:]
    return data_train, data_test
