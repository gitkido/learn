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

# 交差エントロピー誤差（バッチ処理対応版）
# 引数:
# y: ニューラルネットワークの出力（2次元配列。1入力に対する出力セット×入力数分。）
# t: 教師データ（yと同じ形）
def cross_entropy_error(y, t):
    # 入力が1セットのみの場合は2次元配列に整形する [a, b, c] => [[a, b, c]]
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size: # sizeは要素数（3×3なら9）
        # 1次元目（横軸）を基準に、正解ラベルのインデックスを抽出
        t = t.argmax(axis=1)
        
    batch_size = y.shape[0] # 出力データ数（＝投入データ数）
    
    # y[np.arange(batch_size), t] : yの成分のうち正解ラベルに該当するもの
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

""" OLD

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
"""