# coding: utf-8
import numpy as np


def identity_function(x):
	return x


def step_function(x):
	return np.array(x > 0, dtype=np.int)


def sigmoid(x):
	return 1 / (1 + np.exp(-x))    


def sigmoid_grad(x):
	return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
	return np.maximum(0, x)


def relu_grad(x):
	grad = np.zeros(x)
	grad[x>=0] = 1
	return grad


def softmax(x):
	if x.ndim == 2:
		x = x.T
		x = x - np.max(x, axis=0)
		y = np.exp(x) / np.sum(np.exp(x), axis=0)
		return y.T 

	x = x - np.max(x) # 오버플로 대책
	return np.exp(x) / np.sum(np.exp(x))

########################################################################################
## Q. 왜 정확도가 아닌 손실함수를 신경망 학습의 지표로 삼는가?                                      ##
## A. 정확도를 지표로 하면 매개변수를 미분했을 때 대부분의 경우 0이 되기 때문이다.                      ##
##    매개변수를 약간만 조정해서는 정확도가 개선되지 않고, 변화하더라도 불연속적인 값(discrete)으로 바뀐다.  ##
########################################################################################


def mean_squared_error(y, t):
	"""
	손실함수의 미분값을 구할 때 발생하는 미분계수(2)를 상쇄하기 위해 0.5를 곱한다.
	"""
	return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
	"""
	batch_size = 5일 때,
	np.arange(batch_size) == [0, 1, 2, 3, 4]
	t == [2, 7, 0, 9, 4] (label 예시)

	y[np.arange(batch_size), t] == [y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]]
		→ 각 데이터의 정답 레이블에 해당하는 신경망의 출력
	""" 
	if y.ndim == 1:
		t = t.reshape(1, t.size)
		y = y.reshape(1, y.size)

	# 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
	if t.size == y.size:
		t = t.argmax(axis=1)

	batch_size = y.shape[0]
	return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size # 마이너스 무한대 방지 위해 1e-7 더해줌.


def softmax_loss(X, t):
	y = softmax(X)
	return cross_entropy_error(y, t)
