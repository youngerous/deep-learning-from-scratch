import numpy as np

def AND(x1, x2):
	w1, w2, theta = 0.5, 0.5, 0.7
	tmp = x1 * w1 + x2 * w2
	if tmp <= theta:
		return 0
	elif tmp > theta:
		return 1

def AND_numpy(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.7
	tmp = np.sum(x*w) + b
	if tmp <= 0:
		return 0
	else:
		return 1
	
def NAND(x1, x2):
	x = np.array([x1, x2])
	w = np.array([-0.5, -0.5]) # 
	b = 0.7                    #
	tmp = np.sum(x*w) + b
	if tmp <= 0:
		return 0
	else:
		return 1	
	
def OR(x1, x2):
	x = np.array([x1, x2])
	w = np.array([0.5, 0.5])
	b = -0.2 #
	tmp = np.sum(x*w) + b
	if tmp <= 0:
		return 0
	else:
		return 1	
	
def XOR(x1, x2):
	s1 = NAND(x1, x2)
	s2 = OR(x1, x2)
	y  = AND(s1, s2)
	return y

#######################################

def step_function(x):
	y = x > 0
	return y.astype(np.int)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x) # return 1*(x>0)*x

def identity_function(x):
	return x

def softmax(a):
	"""
		Train 시에만 softmax를 사용한다.
		Test 시에는 softmax를 사용하지 않는다.
	"""
	c = np.max(a)
	exp_a = np.exp(a - c) # overflow 방지
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	
	return y