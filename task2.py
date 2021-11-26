import numpy as np
import mnist
import random

train_images, train_labels, test_images, test_labels = None, None, None, None

d = 784
class_num = 10
inner_node_num = 5
batch_num = 100

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

def calc_cross_entropy_error(y, label):
  label_vector = np.reshape(np.zeros(class_num), (class_num, 1))
  label_vector[label] = 1
  return -(np.dot(np.log(y), label_vector))
  
def input_layer(image):
  image = np.array(image)
  trans_image = np.reshape(image, (d, 1))
  return trans_image

def inner_layer(vecx):
  np.random.seed(10)
  W_1 = np.random.normal(loc=0, scale=np.sqrt(1/d), size=(inner_node_num, d))
  vecb_1 = np.random.normal(loc=0, scale=np.sqrt(1/d), size=(inner_node_num, 1))
  return sigmoid_function(np.dot(W_1, vecx) + vecb_1)

def output_layer(vecy):
  np.random.seed(20)
  W_2 = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_num), size=(class_num, inner_node_num))
  vecb_2 = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_num), size=(class_num, 1))
  return softmax_function(np.dot(W_2, vecy) + vecb_2)

def main():
  image_load()
  i = int(input())
  #selected_batchs = random.sample(train_images, batch_num)
  image = test_images[i]
  result = output_layer(inner_layer(input_layer(image)))
  ans = np.argmax(result)
  print(ans)
  calc_cross_entropy_error([0.1, 0.2, 0.5, 0.1, 0, 0, 0.1, 0, 0, 0], 3)

if __name__ ==  '__main__':
  main()
