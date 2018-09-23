# 3.4 3層ニューラルネットワークの実装
# 2入力→3ノード→2ノード→2出力 のニューラルネットワークを作る。

import numpy as np
import functions as f

# 重み付け、バイアスの定義
def init_network():

    # ディクショナリ
    network = {}
    
    # ディクショナリの中身を詰めていくよ
    network[ 'W1' ] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) # 2*3行列
    network[ 'W2' ] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]]) # 3*2行列
    network[ 'W3' ] = np.array([[0.1, 0.3], [0.2, 0.4]]) # 2*2行列
    network[ 'b1' ] = np.array([0.1, 0.2, 0.3]) # 3要素配列
    network[ 'b2' ] = np.array([0.1, 0.2]) # 3要素配列
    network[ 'b3' ] = np.array([0.1, 0.2]) # 2要素配列

    return network

# フォワード実行（入力→出力）
# xは2要素配列であることを前提とする
def forward(network, x):

    # 1層目: 2入力→3ノード
    a1 = network[ 'b1'] + np.dot(x, network[ 'W1' ])
    z1 = f.sigmoid(a1)
    
    # 2層目: 3ノード→2ノード
    a2 = network[ 'b2'] + np.dot(z1, network[ 'W2' ])
    z2 = f.sigmoid(a2)
    
    # 3層目: 2ノード→2出力
    a3 = network[ 'b3'] + np.dot(z2, network[ 'W3' ])
    y = f.identity_function(a3) # 最後のみ恒等関数を使う

    return y

# 実行
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)
