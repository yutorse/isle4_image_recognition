import numpy as np
import mnist
import matplotlib.pyplot as plt
import random

train_images, train_labels, test_images, test_labels = None, None, None, None

input_node_size = 784
inner_node_size = 100
output_node_size = 10
batch_size = 100
epoch = 50

inner_layer_seed = 10
output_layer_seed = 20

parameters = {}

learning_rate = 0.1

def image_load():
  global train_images, train_labels, test_images, test_labels
  test_images = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
  test_labels = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") #確認済み
  train_images = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
  train_labels = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

def plot_figure(list):
  plt.figure(figsize=(8, 6)) # 図の設定
  plt.plot(np.arange(1, len(list) + 1), list) # 折れ線グラフ
  plt.xlabel('iteration') # x軸ラベル
  plt.ylabel('loss') # y軸ラベル
  plt.title('Cross Entropy Error', fontsize=20) # タイトル
  plt.grid() # グリッド線
  plt.show()

def sigmoid_function(t):
  return 1/(1 + np.exp(-t))

def softmax_function(a):
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

def loss_function(processed_images, labels):
  cross_entropy_error_list = []
  for i in range(batch_size):
    processed_image, label = processed_images[i], labels[i]
    label_vector = np.zeros(output_node_size)
    label_vector[label] = 1
    cross_entropy_error = -(np.dot(label_vector, np.log(processed_image)))
    cross_entropy_error_list.append(list(cross_entropy_error))
    cross_entropy_error_mean = np.mean(cross_entropy_error_list)
  return cross_entropy_error_mean

def calc_derivative_a(processed_images, labels):
  derivative_a_list = []
  for i in range(batch_size):
    processed_image, label = processed_images[i].reshape(-1), labels[i]
    label_vector = np.zeros(output_node_size)
    label_vector[label] = 1
    derivative_a_list.append(list((processed_image - label_vector)/batch_size))
  return derivative_a_list

def calc_derivative_t(y, derivative_y):
  derivative_t_list = []
  for i in range(batch_size):
    derivative_y_i = (derivative_y.T[i]).reshape(inner_node_size, 1)
    derivative_t_list.append(list(derivative_y.T[i] * y[i] * (1-y[i])))
  return derivative_t_list

def get_batch():
  indexs = np.random.choice(len(train_images), size=batch_size, replace=False)
  batchs, labels = train_images[indexs], train_labels[indexs]
  return batchs, labels

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
  image_load()

  # W_1, W_2, b_1, b_2 を乱数で初期化
  parameters["W_1"] = np.random.normal(loc=0, scale=np.sqrt(1/input_node_size), size=(inner_node_size, input_node_size))
  parameters["b_1"] = np.random.normal(loc=0, scale=np.sqrt(1/input_node_size), size=(inner_node_size, 1))
  parameters["W_2"] = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(output_node_size, inner_node_size))
  parameters["b_2"] = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(output_node_size, 1))
  
  training_loss_list = []

  # 学習
  for i in range(epoch):
    print(f"epoch: {i+1}/{epoch}")
    for j in range(len(train_images)//batch_size):
      batchs, labels = get_batch()
      x = np.array(list(map(input_layer, batchs)))
      y = np.array(list(map(inner_layer, x)))
      processed_batchs = np.array(list(map(output_layer, y)))
      x = x.reshape(batch_size, input_node_size)
      y = y.reshape(batch_size, inner_node_size)
      
      derivative_a = np.array(calc_derivative_a(processed_batchs, labels))
      derivative_X_2 = np.dot(parameters["W_2"].T, derivative_a.T)
      derivative_W_2 = np.dot(derivative_a.T, y)
      derivative_b_2 = ((np.sum(derivative_a.T, axis=1)).reshape(output_node_size, 1))
      derivative_t = np.array(calc_derivative_t(y, derivative_X_2))
      derivative_X_1 = np.dot(parameters["W_1"].T, derivative_t.T)
      derivative_W_1 = np.dot(derivative_t.T, x)
      derivative_b_1 = ((np.sum(derivative_t.T, axis=1)).reshape(inner_node_size, 1))

      parameters["W_1"] -= (learning_rate*derivative_W_1)/(i+1)/(i+1)
      parameters["W_2"] -= (learning_rate*derivative_W_2)/(i+1)/(i+1)
      parameters["b_1"] -= (learning_rate*derivative_b_1)/(i+1)/(i+1)
      parameters["b_2"] -= (learning_rate*derivative_b_2)/(i+1)/(i+1)

      cross_entropy_error = loss_function(processed_batchs, labels)
      training_loss_list.append(cross_entropy_error)
      
      #print(f"The mean of losses is {cross_entropy_error}.")
  plot_figure(training_loss_list)   

  correct_num = 0
  for i in range(len(test_images)):
    test_image = test_images[i]
    result = output_layer(inner_layer(input_layer(test_image)))
    ans = np.argmax(result)
    if(ans == test_labels[i]):
      correct_num += 1
      
  np.savez('parameters.npz', W_1=parameters["W_1"], W_2=parameters["W_2"], b_1=parameters["b_1"], b_2=parameters["b_2"])
  print(correct_num/len(test_images))
  #batchs, labels = get_batch()
  #processed_batchs = np.array(list(map(output_layer, map(inner_layer, map(input_layer, batchs)))))
  #cross_entropy_error = loss_function(processed_batchs, labels)
  #print(f"The mean of losses is {cross_entropy_error}.")

if __name__ ==  '__main__':
  main()
