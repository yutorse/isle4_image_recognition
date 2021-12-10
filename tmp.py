import numpy as np

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

print(np.array([[1, 2, 3], [4, 5, 6]]) * np.array([[True, False, True], [True, True, True]]))
t = np.array([[-1, 2, -3], [4, 5, 6]])
print(np.mean(t, axis=0))