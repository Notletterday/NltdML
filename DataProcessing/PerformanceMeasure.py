from sklearn.metrics import confusion_matrix


def MeanSquarederror(test, result):
    '''
    test与result一个是测试集本身的映射结果，一个是测试集经过模型的映射
    :param test:
    :param result:
    :return:
    '''
    sum = 0
    n = len(test)
    i = 0
    while i < n:
        sum += (test[i] - result[i]) * (test[i] - result[i])
        i += 1
    return sum / n


def ACC(test, result):
    """

    :param test:
    :param result:
    :return:
    """
    sum = 0
    n = len(test)
    i = 0
    while i < n:
        if test[i] == result[i]:
            sum += 1
        i += 1
    return sum / n


def REC(test, result):
    """

    :param test:
    :param result:
    :return:
    """
    sum = 0
    n = len(test)
    i = 0
    while i < n:
        if test[i] != result[i]:
            sum += 1
        i += 1
    return sum / n


def ConfusionMatrix(y_test, y_pre):
    """
    记得这是分类的任务，如果你当成回归的话很可能运行失败
    :param test:
    :param result:
    :return:
    """
    return confusion_matrix(y_test, y_pre)


def F1_Score(y_test, y_pre):
    """
     记得这是分类的任务，如果你当成回归的话很可能运行失败
    :param test:
    :param result:
    :return:
    """
    matrix = confusion_matrix(y_test, y_pre)
    n = len(y_test)
    TP = matrix[0, 0]
    TN = matrix[1, 1]
    return (2 * TP) / (n + TP - TN)


def Precision(y_test, y_pre):
    """
    :param y_test:
    :param y_pre:
    :return:
    """
    TP = confusion_matrix(y_test, y_pre)[0, 0]
    FP = confusion_matrix(y_test, y_pre)[1, 0]
    return TP / (TP + FP)


def Recall(y_test, y_pre):
    """
    :param y_test:
    :param y_pre:
    :return:
    """
    TP = confusion_matrix(y_test, y_pre)[0, 0]
    FN = confusion_matrix(y_test, y_pre)[1, 1]
    return TP / (TP + FN)


def Fβ_score(y_test, y_pre, beta=0.5):
    """
    :param y_test:
    :param y_pre:
    :param beta:
    :return:
    """
    R = Recall(y_test, y_pre)
    P = Precision(y_test, y_pre)
    b = beta * beta
    return ((1 + b) * P * R) / ((b * P) + R)


def TruePositiverate(y_test, y_pre):
    TP = confusion_matrix(y_test, y_pre)[0, 0]
    FN = confusion_matrix(y_test, y_pre)[0, 1]
    return TP / (TP + FN)


def FalsePositiverate(y_test, y_pre):
    FP = confusion_matrix(y_test, y_pre)[1, 0]
    FN = confusion_matrix(y_test, y_pre)[0, 1]
    return FP / (FP + FN)

def R2Score(y_test, y_pre):
    mse = MeanSquarederror(y_test,y_pre)
    _var = y_test.var()
    return 1-(mse/_var)
