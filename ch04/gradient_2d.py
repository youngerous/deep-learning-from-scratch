# coding: utf-8
# cf.http://d.hatena.ne.jp/white_wheels/20100327/p3
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D


def _numerical_gradient_no_batch(f, x):
	"""
	gradient: 모든 변수의 편미분을 벡터로 정리한 것.
	"""
	h = 1e-4 # 0.0001
	grad = np.zeros_like(x) # x와 형상이 같은 배열을 생성

	for idx in range(x.size):
		tmp_val = x[idx]

		# f(x+h) 계산
		x[idx] = float(tmp_val) + h
		fxh1 = f(x)

		# f(x-h) 계산
		x[idx] = tmp_val - h 
		fxh2 = f(x) 

		grad[idx] = (fxh1 - fxh2) / (2*h)
		x[idx] = tmp_val # 값 복원

	return grad


def numerical_gradient(f, X):
	"""
	numpy array인 X의 각 원소에 대해 수치미분을 구한다.
	
	예)
		numerical_gradient(function_2, np.array([3.0, 4.0]))  →  array([6., 8.])
		numerical_gradient(function_2, np.array([0.0, 2.0]))  →  array([0., 4.])
		numerical_gradient(function_2, np.array([3.0, 0.0]))  →  array([6., 0.])
		
	수치미분은 단순하고 구현하기 쉽지만, 계산 시간이 오래 걸린다는 단점이 있다.
	"""
	if X.ndim == 1:
		return _numerical_gradient_no_batch(f, X)
	else:
		grad = np.zeros_like(X) 

		for idx, x in enumerate(X):
			grad[idx] = _numerical_gradient_no_batch(f, x)

		return grad


def function_2(x):
	"""
	그래프를 그리면 3차원으로 그려질 것이다. 
	"""
	if x.ndim == 1:
		return np.sum(x**2)
	else:
		return np.sum(x**2, axis=1)


def tangent_line(f, x):
	d = numerical_gradient(f, x)
	print(d)
	y = f(x) - d*x
	return lambda t: d*t + y

if __name__ == '__main__':
	x0 = np.arange(-2, 2.5, 0.25)
	x1 = np.arange(-2, 2.5, 0.25)
	X, Y = np.meshgrid(x0, x1)

	X = X.flatten()
	Y = Y.flatten()

	grad = numerical_gradient(function_2, np.array([X, Y]) )

	plt.figure()
	plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
	plt.xlim([-2, 2])
	plt.ylim([-2, 2])
	plt.xlabel('x0')
	plt.ylabel('x1')
	plt.grid()
	plt.legend()
	plt.draw()
	plt.show()
	
	## 기울기가 가리키는 방향 == 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향