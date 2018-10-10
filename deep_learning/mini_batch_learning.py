# 4.5.2 ミニバッチ学習の実装

import datetime # ★進捗測定用
import numpy as np
from dataset.mnist import load_mnist # ★カレントディレクトリ下に配置しておくこと。
from two_layer_net import TwoLayerNet

# 学習用データ、教師データの読み込み
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

# ハイパーパラメータ
iters_num = 10000             # 学習回数（100データ×10000回）
train_size = x_train.shape[0] # 全訓練データ数
batch_size = 100              # 1回の学習で訓練するデータ数
learning_rate = 0.1           # 学習率

train_loss_list = [] # 学習回数ごとの損失関数の値を格納する。
train_acc_list = [] # 学習回数ごとの（訓練データに対する）認識精度を格納する。
test_acc_list = [] # 学習回数ごとの（テストデータに対する）認識精度を格納する。

# 1エポック（訓練データが1周する期間）は学習何回分か
iter_per_epoch = max(train_size / batch_size, 1)

# ニューラルネット初期化
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ★進捗測定用
    if i == 0:
        nowbfr = datetime.datetime.now()
    if i > 0:
        nowbfr = now
    
    now = datetime.datetime.now()
    timespan = now - nowbfr
    print("No. " + str(i) + " done. timespan: " + str(timespan))
    # ミニバッチの取得
    # 各データに対してランダムな要素を抽出する
    batch_mask = np.random.choice(train_size, batch_size) # ランダムなラベルの配列
    x_batch = x_train[batch_mask] # n回目の入力データ
    t_batch = t_train[batch_mask] # n回目の教師データ
    
    # 勾配の計算
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # パラメータの更新（勾配降下）
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    # 学習経過の記録
    loss = network.loss(x_batch, t_batch) # 損失関数の値
    train_loss_list.append(loss)
    
    # 1エポックごとに認識精度を計算
    # ※認識精度は、ニューラルネットワーク（重みとバイアスのセット）に対して
    #   評価される。
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train) # 訓練データの認識精度
        test_acc = network.accuracy(x_test, t_test) # テストデータの認識精度
        # リストに格納
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        # 標準出力に表示
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
