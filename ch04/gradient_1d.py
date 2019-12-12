# coding: utf-8
import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
	"""
	일반적인 수치 미분(근사치 미분)에는 오차가 포함된다.
	오차를 줄이기 위해 x를 중심으로 그 전후의 차분을 계산한다 (중심 차분 or 중앙 차분).
	"""
	h = 1e-4 # 0.0001 (너무 작은 값을 사용하면 컴퓨터로 계산하는 데 문제가 발생)
	return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
	return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
	d = numerical_diff(f, x)
	print(d)
	y = f(x) - d*x
	return lambda t: d*t + y

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2)
plt.show()
