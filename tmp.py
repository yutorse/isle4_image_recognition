import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
gamma = 1
beta = 0

def batch_normalization(x):
  delta = 1e-7
  mu = x.mean(axis = 0)
  print(mu)
  var = x.var(axis = 0)
  print(x - mu)
  print(var)
  print(np.sqrt(var + delta))
  x_hat = ((x - mu)/(np.sqrt(var + delta)))
  print(x_hat)
  y = gamma * x_hat + beta
  return y

def sigmoid_function(t):
  return 1/(1 + np.exp(-t))

def softmax_function(a): # ソフトマックス関数
  max_a = (a.max(axis=1)).reshape(len(a),1)
  sum = (np.sum(np.exp(a - max_a), axis=1)).reshape(len(a),1)
  return np.exp(a - max_a) / sum

images = np.loadtxt("le4MNIST_X.txt")
for i in range(1000):
  fig = plt.figure()
  plt.imshow((np.array(images[i])).reshape(28,28), cmap=cm.gray)
  fig.savefig(f"pic{i}")
