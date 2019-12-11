# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
	"""
	normalize = True 상태 (정규화로 전처리).
	실제 정규화 시에는 단순 255로 나누는 것 외에, 데이터 전체의 분포를 고려하여 전처리하는 경우가 많다.
	(전체 평균과 표준편차 이용)
	
	전체 데이터를 균일하게 분포시키는 데이터 백색화(whitening) 기법도 존재.
	"""
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
	"""
	return 값 예시: [0.1, 0.3, 0.2, ..., 0.04]
		→ label이 0일 확률 0.1, 1일 확률 0.3, 2일 확률 0.2, ...
	"""
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

	# x.shape == (784, )  → batch가 100인 경우에는 (100, 784)가 될 것임.
    a1 = np.dot(x, W1) + b1  # W1.shape == (784, 50)
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2 # W2.shape == (50, 100)
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3 # W3.shape == (100, 10)
    y = softmax(a3)

    return y


x, t = get_data() # x: (10000, 784) / t: (10000, )
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
