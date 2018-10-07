# 2層ニューラルネットワーク（重み・バイアス付き）
# 入力層→隠れ層→出力層

from functions import *
from gradient import numerical_gradient

class TwoLayerNet:

    # 重みとバイアスの初期化
    def __init__(self, input_size, hidden_size, output_size,
                 weight_init_std=0.01):
    
        # 重みはガウス分布、バイアスは0
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        
    # 認識（推論）
    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        # 規則にしたがい出力値を計算
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
        
    # 損失関数
    def loss(self, x, t):
        # まず推論させる。
        y = self.predict(x)
        
        # 交差エントロピー誤差を返却
        return cross_entropy_error(y, t)
        
    # 認識精度
    def accuracy(self, x, t):
        # まず推論させる。
        y = self.predict(x)
        
        # 最大値（最も確率が高い選択肢）を抽出
        y = np.argmax(y, axis=1)
        # ★これはone-hot表現であることが前提でしょうかね
        t = np.argmax(t, axis=1)
        
        # 出力のうち、正解ラベルと一致しているものが全体のうちどれだけあるか
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
        
    # 勾配
    def numerical_gradient(self, x, t):
        # 勾配を算出する対象は損失関数
        loss_W = lambda W: self.loss(x, t)
        
        # それぞれの重み・バイアスに対して勾配を算出
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        return grads
