
import numpy as np
from DataProcessing import split

# 用于测试功能
if __name__ == '__main__':
    x = np.array([[1,2],[2,3],[3,4],[2,8]])
    split.StratifiedSampling(data=x,col=1,n=3)





