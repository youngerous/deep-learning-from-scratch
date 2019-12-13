# coding: utf-8
import numpy as np
from common.functions import *
from common.util import im2col, col2im


class Relu:
	def __init__(self):
		self.mask = None 

	def forward(self, x):
		self.mask = (x <= 0) # forward input 값이 0 이하면 True. x와 shape이 같다.
		out = x.copy()
		out[self.mask] = 0

		return out

	def backward(self, dout):
		dout[self.mask] = 0 # forward에서 <= 0 이었던 위치 기억
		dx = dout

		return dx


class Sigmoid:
	def __init__(self):
		self.out = None

	def forward(self, x):
		out = sigmoid(x)
		self.out = out
		return out

	def backward(self, dout):
		dx = dout * (1.0 - self.out) * self.out

		return dx


class Affine:
	"""
	신경망의 순전파 때 수행하는 행렬의 곱을 기하학에서는 Affine transformation이라 한다.
	"""
	def __init__(self, W, b):
		self.W = W
		self.b = b

		self.x = None
		self.original_x_shape = None
		# 가중치와 편향 매개변수의 미분
		self.dW = None
		self.db = None

	def forward(self, x):
		# 텐서 대응
		self.original_x_shape = x.shape
		x = x.reshape(x.shape[0], -1)
		self.x = x

		out = np.dot(self.x, self.W) + self.b

		return out

	def backward(self, dout):
		dx = np.dot(dout, self.W.T)
		self.dW = np.dot(self.x.T, dout)
		self.db = np.sum(dout, axis=0)

		dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
		return dx


class SoftmaxWithLoss:
	"""
	신경망에서 수행하는 작업: 학습(train), 추론(inference)

	추론 시에는 softmax 계층을 사용하지 않는다.
	신경망에서 정규화하지 않은 출력 결과를 score라 하는데,
		정답을 하나만 내는 경우는 가장 높은 점수만 알면 되기 때문에 softmax를 쓸 필요가 없다.
	"""
	def __init__(self):
		self.loss = None # 손실함수
		self.y = None    # softmax의 출력
		self.t = None    # 정답 레이블(원-핫 인코딩 형태)

	def forward(self, x, t):
		self.t = t
		self.y = softmax(x)
		self.loss = cross_entropy_error(self.y, self.t)

		return self.loss

	def backward(self, dout=1):
		batch_size = self.t.shape[0]
		if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
			dx = (self.y - self.t) / batch_size
		else:
			dx = self.y.copy()
			dx[np.arange(batch_size), self.t] -= 1
			dx = dx / batch_size

		return dx


class Dropout:
	"""
	http://arxiv.org/abs/1207.0580
	"""
	def __init__(self, dropout_ratio=0.5):
		self.dropout_ratio = dropout_ratio
		self.mask = None

	def forward(self, x, train_flg=True):
		if train_flg:
			self.mask = np.random.rand(*x.shape) > self.dropout_ratio
			return x * self.mask
		else:
			return x * (1.0 - self.dropout_ratio)

	def backward(self, dout):
		return dout * self.mask


class BatchNormalization:
	"""
	http://arxiv.org/abs/1502.03167
	"""
	def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
		self.gamma = gamma
		self.beta = beta
		self.momentum = momentum
		self.input_shape = None # 합성곱 계층은 4차원, 완전연결 계층은 2차원  

		# 시험할 때 사용할 평균과 분산
		self.running_mean = running_mean
		self.running_var = running_var  

		# backward 시에 사용할 중간 데이터
		self.batch_size = None
		self.xc = None
		self.std = None
		self.dgamma = None
		self.dbeta = None

	def forward(self, x, train_flg=True):
		self.input_shape = x.shape
		if x.ndim != 2:
			N, C, H, W = x.shape
			x = x.reshape(N, -1)

		out = self.__forward(x, train_flg)

		return out.reshape(*self.input_shape)

	def __forward(self, x, train_flg):
		if self.running_mean is None:
			N, D = x.shape
			self.running_mean = np.zeros(D)
			self.running_var = np.zeros(D)

		if train_flg:
			mu = x.mean(axis=0)
			xc = x - mu
			var = np.mean(xc**2, axis=0)
			std = np.sqrt(var + 10e-7)
			xn = xc / std

			self.batch_size = x.shape[0]
			self.xc = xc
			self.xn = xn
			self.std = std
			self.running_mean = self.momentum * self.running_mean + (1-self.momentum) * mu
			self.running_var = self.momentum * self.running_var + (1-self.momentum) * var            
		else:
			xc = x - self.running_mean
			xn = xc / ((np.sqrt(self.running_var + 10e-7)))

		out = self.gamma * xn + self.beta 
		return out

	def backward(self, dout):
		if dout.ndim != 2:
			N, C, H, W = dout.shape
			dout = dout.reshape(N, -1)

		dx = self.__backward(dout)

		dx = dx.reshape(*self.input_shape)
		return dx

	def __backward(self, dout):
		dbeta = dout.sum(axis=0)
		dgamma = np.sum(self.xn * dout, axis=0)
		dxn = self.gamma * dout
		dxc = dxn / self.std
		dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
		dvar = 0.5 * dstd / self.std
		dxc += (2.0 / self.batch_size) * self.xc * dvar
		dmu = np.sum(dxc, axis=0)
		dx = dxc - dmu / self.batch_size

		self.dgamma = dgamma
		self.dbeta = dbeta

		return dx


class Convolution:
	def __init__(self, W, b, stride=1, pad=0):
		self.W = W
		self.b = b
		self.stride = stride
		self.pad = pad

		# 중간 데이터（backward 시 사용）
		self.x = None   
		self.col = None
		self.col_W = None

		# 가중치와 편향 매개변수의 기울기
		self.dW = None
		self.db = None

	def forward(self, x):
		FN, C, FH, FW = self.W.shape
		N, C, H, W = x.shape
		out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
		out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

		col = im2col(x, FH, FW, self.stride, self.pad)
		col_W = self.W.reshape(FN, -1).T

		out = np.dot(col, col_W) + self.b
		out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

		self.x = x
		self.col = col
		self.col_W = col_W

		return out

	def backward(self, dout):
		FN, C, FH, FW = self.W.shape
		dout = dout.transpose(0,2,3,1).reshape(-1, FN)

		self.db = np.sum(dout, axis=0)
		self.dW = np.dot(self.col.T, dout)
		self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

		dcol = np.dot(dout, self.col_W.T)
		dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

		return dx


class Pooling:
	def __init__(self, pool_h, pool_w, stride=1, pad=0):
		self.pool_h = pool_h
		self.pool_w = pool_w
		self.stride = stride
		self.pad = pad

		self.x = None
		self.arg_max = None

	def forward(self, x):
		N, C, H, W = x.shape
		out_h = int(1 + (H - self.pool_h) / self.stride)
		out_w = int(1 + (W - self.pool_w) / self.stride)

		col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
		col = col.reshape(-1, self.pool_h*self.pool_w)

		arg_max = np.argmax(col, axis=1)
		out = np.max(col, axis=1)
		out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

		self.x = x
		self.arg_max = arg_max

		return out

	def backward(self, dout):
		dout = dout.transpose(0, 2, 3, 1)

		pool_size = self.pool_h * self.pool_w
		dmax = np.zeros((dout.size, pool_size))
		dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
		dmax = dmax.reshape(dout.shape + (pool_size,)) 

		dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
		dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

		return dx
