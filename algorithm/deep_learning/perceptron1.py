# パーセプトロンの実装
import numpy as np

# 簡単な実装(ANDゲート)
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1

# 重みとバイアスによる実装(ANDゲート)
def AND2(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 重みとバイアスによる実装(NANDゲート)
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5]) #重みとバイアスだけがANDと違う！
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 重みとバイアスによる実装(ORゲート)
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5]) #重みとバイアスだけがANDと違う！
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# XORゲート
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

# ANDゲート
print('AND(0, 0):', AND(0, 0))
print('AND(1, 0):', AND(1, 0))
print('AND(0, 1):', AND(0, 1))
print('AND(1, 1):', AND(1, 1))
# ANDゲート2
print('AND2(0, 0):', AND2(0, 0))
print('AND2(1, 0):', AND2(1, 0))
print('AND2(0, 1):', AND2(0, 1))
print('AND2(1, 1):', AND2(1, 1))
# NANDゲート
print('NAND(0, 0):', NAND(0, 0))
print('NAND(1, 0):', NAND(1, 0))
print('NAND(0, 1):', NAND(0, 1))
print('NAND(1, 1):', NAND(1, 1))
# ORゲート
print('OR(0, 0):', OR(0, 0))
print('OR(1, 0):', OR(1, 0))
print('OR(0, 1):', OR(0, 1))
print('OR(1, 1):', OR(1, 1))
# XORゲート
print('XOR(0, 0):', XOR(0, 0))
print('XOR(1, 0):', XOR(1, 0))
print('XOR(0, 1):', XOR(0, 1))
print('XOR(1, 1):', XOR(1, 1))
