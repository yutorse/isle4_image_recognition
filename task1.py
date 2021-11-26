import numpy as np
import mnist

d = 784
class_num = 10
inner_node_num = 5

def sigmoid_function(t): #動作確認OK.
  return 1/(1 + np.exp(-t))

def softmax_function(a): #numpy配列で受けとる #動作確認OK.
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

def input_layer(image):
  image = np.array(image)
  trans_image = np.reshape(image, (d, 1))
  return trans_image

def inner_layer(vecx):
  print(vecx)
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
  test_images = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
  test_labels = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") #確認済み

  i = int(input())
  image = test_images[i]
  result = output_layer(inner_layer(input_layer(image)))
  ans = np.argmax(result)
  print(ans)

if __name__ ==  '__main__':
  main()
