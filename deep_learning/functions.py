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

""" 出力層の活性化関数 """

# 恒等関数
def identity_function(x):
    return x

# ソフトマックス関数 (オーバーフロー対策付き)
def softmax(a):

    c = np.max(a)

    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

""" 損失関数 """

# ※yはニューラルネットワークの出力、tは教師データ

# 2乗和誤差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

# 交差エントロピー誤差
# 教師データはone-hot表現になっていることが前提
def cross_entropy_error(y, t):
    delta = 1e-7 # log(0) (= -inf) にならないように対策
    return -1.0 * np.sum(t * np.log(y + delta))