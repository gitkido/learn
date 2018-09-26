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
    return -np.sum(t * np.log(y + delta))

# 交差エントロピー誤差（バッチ対応版1）
# 教師データはone-hot表現になっていることが前提
def cross_entropy_error1(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    '''
    この部分の注意。
    yは、通常こんなのが来る想定で

    [[1, 2, 3, 4, 5, 6, 7, 8],
     [1, 2, 3, 4, 5, 6, 7, 8],
               ...
     [1, 2, 3, 4, 5, 6, 7, 8]]
    
    データが1つだけの場合は特殊形式になってしまうので、↑と合わせる。
    [1, 2, 3, 4, 5, 6, 7, 8]   こうではなく
    [[1, 2, 3, 4, 5, 6, 7, 8]] こう
    '''
    batch_size = y.shape[0] # 何件のデータを束ねたバッチか
    delta = 1e-7 # log(0) (= -inf) にならないように対策
    return -np.sum(t * np.log(y + delta)) / batch_size

# 交差エントロピー誤差（バッチ対応版2）
# 教師データは正解ラベルが格納された配列（[2, 7, 0, 9, 4]みたいな）の前提
def cross_entropy_error2(y, t):
    if y.ndim == 1:
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size
    ''' ココの注意。
    y[np.arange(5), t] ※5は例
    = y[[0, 1, 2, 3, 4], [2, 7, 0, 9, 4]]
    = [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]]
    →正解ラベルの値だけが抽出できる。
    （numpyの配列はこういう動きみたい。自然なような、恣意的なような・・・）
    '''
