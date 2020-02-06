# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def ReLU(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)
    
input_data = np.random.randn(1000, 100)  # 1000개의 데이터
node_num = 100  # 각 은닉층의 노드(뉴런) 수
hidden_layer_size = 5  # 은닉층이 5개
activations = {}  # 이곳에 활성화 결과를 저장

x = input_data

for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    # 초깃값을 다양하게 바꿔가며 실험해보자！
    w = np.random.randn(node_num, node_num) * 1
    # w = np.random.randn(node_num, node_num) * 0.01
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) # Xavier initialization
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num) # He initialization


    a = np.dot(x, w)


    # 활성화 함수도 바꿔가며 실험해보자！
    z = sigmoid(a)
    # z = ReLU(a)
    # z = tanh(a)

    activations[i] = z

# 히스토그램 그리기
for i, a in activations.items():
    plt.subplot(1, len(activations), i+1)
    plt.title(str(i+1) + "-layer")
    if i != 0: plt.yticks([], [])
    # plt.xlim(0.1, 1)
    # plt.ylim(0, 7000)
    plt.hist(a.flatten(), 30, range=(0,1))
plt.show()

"""
	활성화값이 0과 1에 치우쳐 있는 경우 → Gradient vanishing
	0.5 부근에 집중되어 있는 경우 → 표현력 관점에서 문제 (뉴런의 개수가 의미없어지는 경우)
	
	[대안] 
	1. Xavier initialization (초깃값의 표준편차가 root(1/n)이 되도록 설정)
		⇒ Activation function이 Linear함을 가정 (예: sigmoid, tanh)
		⇒ ReLU에서 사용할 경우 layer가 깊어질수록 Gradient vanishing이 일어날 가능성이 높아짐
	
	2. He initialization (초깃값의 표준편차가 root(2/n)이 되도록 설정)
		⇒ ReLU에 특화된 initialization
			:음의 영역이 0이기 때문에 더 넓게 분포시키기 위함		
			
	[정리]
	- Xavier init: sigmoid, tanh에서 사용
	- He init    : ReLU에서 사용
"""
