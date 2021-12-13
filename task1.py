import numpy as np
import mnist

input_node_size = 784
inner_node_size = 1000
output_node_size = 10

def sigmoid_function(t): #シグモイド関数
  return 1/(1 + np.exp(-t))

def softmax_function(a): #ソフトマックス関数
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

def input_layer(image): #入力層
  image = np.array(image)
  trans_image = np.reshape(image, (input_node_size, 1))
  return trans_image

def inner_layer(x): #中間層
  np.random.seed(10)
  W_1 = np.random.normal(loc=0, scale=np.sqrt(1/input_node_size), size=(inner_node_size, input_node_size))
  b_1 = np.random.normal(loc=0, scale=np.sqrt(1/input_node_size), size=(inner_node_size, 1))
  return sigmoid_function(np.dot(W_1, x) + b_1)

def output_layer(y): #出力層
  np.random.seed(20)
  W_2 = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(output_node_size, inner_node_size))
  b_2 = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(output_node_size, 1))
  return softmax_function(np.dot(W_2, y) + b_2)

def main():
  test_images = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
  test_labels = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")

  i = int(input())
  image = test_images[i]
  processed_image = output_layer(inner_layer(input_layer(image)))
  prediction = np.argmax(processed_image)
  print(prediction)

if __name__ ==  '__main__':
  main()
