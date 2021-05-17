import numpy as np
import tensorflow as tf
# import tensorflow.tensorflow_probability as tfp

class Params():

	def __init__(self, agents, K, always_update=False, topology='ring', maxLam=0.01):
		self.agents = agents
		self.K = K
		self.k = -1

		self.always_update = always_update
		self.maxLam=maxLam

		if topology=='ring' and agents > 3:
			self.graph = np.zeros((agents, agents))
			for i in range(agents):
				self.graph[i][(i + 1) % agents] = 0.99
				self.graph[i][(i + agents - 1) % agents] = 0.99
				self.graph[i][i] = 1.02
		elif topology=='weakRing' and agents > 3:
			self.graph = np.zeros((agents, agents))
			for i in range(agents):
				self.graph[i][(i + 1) % agents] = 0.11
				self.graph[i][(i + agents - 1) % agents] = 0.11
				self.graph[i][i] = 2.78
		elif topology=='full' or topology=='ring':
			self.graph = np.ones((agents, agents))

		self.epochStart = False

	def iterate(self):
		self.k += 1
		if self.k <= self.K or self.always_update:
			self.genLam()
			self.genR()
			self.genA()
			self.genB()
			self.genC()
		
	def refresh(self):
		self.k = 0

	def genLam(self):

		if self.k < self.K:
			self.lam = np.diag(np.random.random(self.agents) * 2 * self.maxLam - self.maxLam)
		else:
			self.lam = np.identity(self.agents) * self.maxLam

	def genR(self):
		if self.k < self.K:
			maxR = 1
			self.r = np.random.random((self.agents, self.agents)) * maxR * 2 - maxR
			self.r = np.multiply(self.r, self.graph)
			self.r = self.r/self.r.sum(axis=1, keepdims=True)
		else:
			self.r = np.random.random((self.agents, self.agents))
			self.r = np.multiply(self.r, self.graph)
			self.r = self.r/self.r.sum(axis=1, keepdims=True)

	def genA(self):
		if self.k < self.K:
			maxA = 1
			self.a = np.random.random((self.agents, self.agents)) * maxA * 2 - maxA
			self.a = np.multiply(self.a, self.graph)
		else:
			self.a = np.random.random((self.agents, self.agents))
			self.a = np.multiply(self.a, self.graph)
			
	def genB(self):
		if self.k < self.K:
			maxB = 1
			self.b = np.random.random((self.agents, self.agents)) * maxB * 2 - maxB
			self.b = np.multiply(self.b, self.graph)
			self.b = self.b/self.b.sum(axis=0, keepdims=True)
		else:
			self.b = np.random.random((self.agents, self.agents))
			self.b = np.multiply(self.b, self.graph)
			self.b = self.b/self.b.sum(axis=0, keepdims=True)

	def genC(self):
		if self.k < self.K:
			maxC = 1
			self.c = np.random.random((self.agents, self.agents)) * maxC * 2 - maxC
			self.c = np.multiply(self.c, self.graph)
			self.c = self.c/self.c.sum(axis=0, keepdims=True)
		else:
			self.c = np.random.random((self.agents, self.agents))
			self.c = np.multiply(self.c, self.graph)
			self.c = self.c/self.c.sum(axis=0, keepdims=True)
	def genPi(self):
		return self.graph/self.graph.sum(axis=1, keepdims=True)
	def genBi(self):
		'''
		Generate column stochastic matrix
		'''
		b = np.random.rand(self.agents, self.agents)
		return b/b.sum(axis=0)[None,:]
		# return np.identity(self.agents)
	def genRand(self):
		return tf.random.uniform(shape=[1])
		# return 1
		# return tf.linalg.tensor_diag(tf.random.uniform(shape=[size])) # If gradient is matrix
	def genDiag(self):
		return np.diag(np.ones(self.agents))