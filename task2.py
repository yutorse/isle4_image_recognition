import numpy as np
import mnist
import random

train_images, train_labels, test_images, test_labels = None, None, None, None

input_node_size = 784
inner_node_size = 5
output_node_size = 10
batch_size = 100

inner_layer_seed = 10
output_layer_seed = 20

# 画像データの読み込み
def load_image():
  global train_images, train_labels, test_images, test_labels
  test_images = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
  test_labels = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") #確認済み
  train_images = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
  train_labels = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

# シグモイド関数
def sigmoid_function(t):
  return 1/(1 + np.exp(-t))

# ソフトマックス関数
def softmax_function(a):
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

# 損失関数
def loss_function(processed_images, labels): #cross_entropy_error
  cross_entropy_error_list = []
  for i in range(batch_size):
    processed_image, label = processed_images[i], labels[i]
    label_vector = np.zeros(output_node_size)
    label_vector[label] = 1
    cross_entropy_error = -(np.dot(label_vector, np.log(processed_image)))
    cross_entropy_error_list.append(list(cross_entropy_error))
    cross_entropy_error_mean = np.mean(cross_entropy_error_list)
  return cross_entropy_error_mean

# バッチの抽出
def get_batch():
  indexs = np.random.choice(len(train_images), size=batch_size, replace=False)
  batchs, labels = train_images[indexs], train_labels[indexs]
  return batchs, labels

#入力層
def input_layer(image):
  image = np.array(image)
  trans_image = np.reshape(image, (input_node_size, 1))
  return trans_image

#中間層
def inner_layer(x):
  np.random.seed(10)
  W_1 = np.random.normal(loc=0, scale=np.sqrt(1/input_node_size), size=(inner_node_size, input_node_size))
  b_1 = np.random.normal(loc=0, scale=np.sqrt(1/input_node_size), size=(inner_node_size, 1))
  return sigmoid_function(np.dot(W_1, x) + b_1)

#出力層
def output_layer(y):
  np.random.seed(20)
  W_2 = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(output_node_size, inner_node_size))
  b_2 = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(output_node_size, 1))
  return softmax_function(np.dot(W_2, y) + b_2)

def main():
  load_image()
  batchs, labels = get_batch()
  processed_batchs = np.array(list(map(output_layer, map(inner_layer, map(input_layer, batchs)))))
  cross_entropy_error = loss_function(processed_batchs, labels)
  print(f"The mean of losses is {cross_entropy_error}.")

if __name__ ==  '__main__':
  main()
