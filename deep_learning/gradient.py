# 勾配法。

import numpy as np

# ある点での勾配（点の各成分に対する関数の偏微分係数）を求める関数
#
# 引数...
# f : 関数
# x : 点（1次元配列）
def numerical_gradient(f, x):
    
    # ごく小さな変化量
    h = 1e-4
    # if x == [x0, x1, x2] then grad == [0, 0, 0]
    grad = np.zeros_like(x)
    
    # xの各要素について偏微分係数を計算
    for idx in range(x.size):
        
        tmp_val = x[idx]
        
        # 正方向の差分 : xの1成分を正方向にずらしたときの関数の値
        x[idx] = tmp_val + h
        dfdx_plus = f(x)
        
        # 負方向の差分 : xの1成分を負方向にずらしたときの関数の値
        x[idx] = tmp_val - h
        dfdx_minus = f(x)
        
        # 偏微分係数を計算して、返却値にセット
        # ★2*hを括弧でくくらないと結果が違ってくる。浮動小数点だからか。
        grad[idx] = (dfdx_plus - dfdx_minus) / (2*h)
        
        # tmp値を元に戻す
        x[idx] = tmp_val
    
    return grad
    
# 勾配降下を行う（その点の勾配にしたがい点を移動させる）
#
# 引数...
# f : 関数
# init_x : 降下開始の点
# lr : 学習率（人間がチューニングする）
# step_num : 何回降下するか（人間がチューニングする）
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    # x（動く点）を初期化
    x = init_x
    
    # ステップ数分降下を行う
    for i in range(step_num):
        
        grad = numerical_gradient(f, x)
        x -= lr * grad
        
        # ★ x -= lr * numerical_gradient(f, x) みたいにまとめると結果が違う。
        # これも浮動小数点計算の特性か？ 丸めタイミングとかが変わるのだろうか。
        
    return x

# テスト用関数
def function_2(x):
    return x[0]**2 + x[1]**2
