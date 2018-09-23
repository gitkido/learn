# パーセプトロンの考え方による論理ゲートの実現（一部）

import numpy as np

def AND(x1, x2):    # ANDゲートを実現するパーセプトロン
    x = np.array([x1, x2])   # 入力値
    w = np.array([0.5, 0.5]) # 重み
    b = -0.7
    
    # 計算
    rslt = np.sum(x * w) + b
    
    # 戻り値を判定
    if rslt <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):    # NANDゲートを実現するパーセプトロン
    x = np.array([x1, x2])   # 入力値
    w = np.array([-0.5, -0.5]) # 重み
    b = 0.7
    
    # 計算
    rslt = np.sum(x * w) + b
    
    # 戻り値を判定
    if rslt <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):    # ANDゲートを実現するパーセプトロン
    x = np.array([x1, x2])   # 入力値
    w = np.array([1, 1]) # 重み
    b = 0
    
    # 計算
    rslt = np.sum(x * w) + b
    
    # 戻り値を判定
    if rslt <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    # XORは他の単相パーセプトロンでは実現できないので、
    # 論理ゲートの組み合わせで作る。
    return AND(OR(x1, x2), NAND(x1, x2))