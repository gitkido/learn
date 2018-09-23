# 関数

import numpy as np

""" 活性化関数 """

# 階段関数 (テンソルが使えるバージョン)
def stepf(x):
    y = x > 0
    return y.astype(np.int)

# シグモイド関数
# (数式をそのまま入れるだけやね)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ReLU関数
def ReLU(x):
    return np.maximum(0, x)

""" その他の関数 """

# 恒等関数
def identity_function(x):
    return x
