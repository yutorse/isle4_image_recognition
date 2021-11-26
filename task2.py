import numpy as np
import mnist
import random

train_images, train_labels, test_images, test_labels = None, None, None, None

d = 784
class_size = 10
inner_node_size = 5
batch_size = 100

def image_load():
  global train_images, train_labels, test_images, test_labels
  test_images = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
  test_labels = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") #確認済み
  train_images = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
  train_labels = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

def sigmoid_function(t): #動作確認OK.
  return 1/(1 + np.exp(-t))

def softmax_function(a): #numpy配列で受けとる #動作確認OK.
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

def calc_cross_entropy_error(y, label): #cross-entropy-error
  label_vector = np.zeros(class_size)
  label_vector[label] = 1
  return -(np.dot(label_vector, np.log(y)))

def choice_batch():
  batchs, labels = [], []
  indexs = np.random.choice(60000, size=batch_size, replace=False)
  for index in indexs:
    batchs.append(train_images[index])
    labels.append(train_labels[index])
  return batchs, labels

def input_layer(image):
  image = np.array(image)
  trans_image = np.reshape(image, (d, 1))
  return trans_image

def inner_layer(vecx):
  np.random.seed(10)
  W_1 = np.random.normal(loc=0, scale=np.sqrt(1/d), size=(inner_node_size, d))
  vecb_1 = np.random.normal(loc=0, scale=np.sqrt(1/d), size=(inner_node_size, 1))
  return sigmoid_function(np.dot(W_1, vecx) + vecb_1)

def output_layer(vecy):
  np.random.seed(20)
  W_2 = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(class_size, inner_node_size))
  vecb_2 = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(class_size, 1))
  return softmax_function(np.dot(W_2, vecy) + vecb_2)

def main():
  image_load()
  batchs, labels = choice_batch()
  batchs_result = np.array(list(map(output_layer, map(inner_layer, map(input_layer, batchs)))))
  cross_entropy_error_list = []
  for i in range(100):
    cross_entropy_error_list.append(list(calc_cross_entropy_error(batchs_result[i], labels[i])))
  cross_entropy_error_mean = np.mean(cross_entropy_error_list)

  print(f"クロスエントロピー誤差の平均は{cross_entropy_error_mean}です。")

if __name__ ==  '__main__':
  main()
