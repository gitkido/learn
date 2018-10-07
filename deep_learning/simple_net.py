# 1層バイアスなしニューラルネットワーク

import sys, os
sys.path.append(os.pardir)
import numpy as np
# ※自分が書き写したモジュールを使用
from functions import softmax, cross_entropy_error
from gradient import numerical_gradient # ★これ使われてなくない？

class simpleNet:
    # コンストラクタ
    def __init__(self):
        # 重み付けの初期値：2×3行列、ガウス分布
        self.W = np.random.randn(2,3)
        
    # ニューラルネットの出力を求める関数
    # 引数...
    # x: 入力（2要素×nセットの配列）
    def predict(self, x):
        # 入力値と重み付けの行列積を計算するだけ
        return np.dot(x, self.W)
        
    # 損失を求める関数
    # 引数...
    # x: 入力（2要素×nセットの配列）
    # t: 教師データ
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss