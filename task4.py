import mnist
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm

input_node_size = 784
inner_node_size = 100
output_node_size = 10

parameters = {}
def load_parameters(file_name):
  load_param = np.load(file_name)
  parameters["W_1"] = load_param["W_1"]
  parameters["W_2"] = load_param["W_2"]
  parameters["b_1"] = load_param["b_1"]
  parameters["b_2"] = load_param["b_2"]

def sigmoid_function(t):
  return 1/(1 + np.exp(-t))

def softmax_function(a):
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

def input_layer(image): #入力層
  image = np.array(image)
  trans_image = np.reshape(image, (input_node_size, 1))
  return trans_image

def inner_layer(x): #中間層
  return sigmoid_function(np.dot(parameters["W_1"], x) + parameters["b_1"])

def output_layer(y): #出力層
  return softmax_function(np.dot(parameters["W_2"], y) + parameters["b_2"])

def main():
  # 画像データの準備
  test_images = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
  test_labels = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")

  # パラメータの準備
  load_parameters("parameters_task3.npz")

  i = int(input('Input the number: '))
  image = test_images[i]
  processed_image = output_layer(inner_layer(input_layer(image)))
  prediction = np.argmax(processed_image)
  print(f"prediction: {prediction}")
  print(f"correct label: {test_labels[i]}")
  plt.imshow(test_images[i], cmap=cm.gray)
  plt.show()

if __name__ ==  '__main__':
  main()
