import numpy as np
import mnist
import matplotlib.pyplot as plt
import random
import time
import joblib
from tqdm import tqdm

train_images, train_labels, test_images, test_labels = None, None, None, None

input_node_size = 784
inner_node_size = 100
output_node_size = 10
batch_size = 100
epoch = 10

parameters = {}

learning_rate = 0.1

# ReLU
ReLU_mask = []

# dropout
training_flag = True
input_nonactive_ratio = 0.2
inner_nonactive_ratio = 0.5
dropout_mask = []

def load_image():
  global train_images, train_labels, test_images, test_labels
  test_images = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
  test_labels = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz") #確認済み
  train_images = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
  train_labels = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

def load_parameters(file_name):
  load_param = np.load(file_name)
  parameters["W_1"] = load_param["W_1"]
  parameters["W_2"] = load_param["W_2"]
  parameters["b_1"] = load_param["b_1"]
  parameters["b_2"] = load_param["b_2"]

def plot_figure(list):
  plt.figure(figsize=(8, 6))
  plt.plot(np.arange(1, len(list) + 1), list)
  plt.xlabel('iteration')
  plt.ylabel('loss')
  plt.title('Cross Entropy Error', fontsize=20) # タイトル
  plt.grid()
  plt.show()

def init_parameters():
  parameters["W_1"] = np.random.normal(loc=0, scale=np.sqrt(1/input_node_size), size=(inner_node_size, input_node_size))
  parameters["b_1"] = np.random.normal(loc=0, scale=np.sqrt(1/input_node_size), size=(inner_node_size, 1))
  parameters["W_2"] = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(output_node_size, inner_node_size))
  parameters["b_2"] = np.random.normal(loc=0, scale=np.sqrt(1/inner_node_size), size=(output_node_size, 1))

def ReLU_function(t): # ReLU関数
  global ReLU_mask
  ReLU_mask = np.where(t>0, 1, 0)
  return np.maximum(t, 0)

def softmax_function(a): # ソフトマックス関数
  max_a = (a.max(axis=1)).reshape(len(a), 1)
  sum = (np.sum(np.exp(a - max_a), axis=1)).reshape(len(a), 1)
  return np.exp(a - max_a) / sum

def loss_function(processed_images, labels): # 損失関数
  delta = 1e-7
  cross_entropy_error_list = []
  for i in range(batch_size):
    processed_image, label = processed_images[i], labels[i]
    label_vector = np.zeros(output_node_size)
    label_vector[label] = 1
    cross_entropy_error = -(np.dot(label_vector, np.log(np.maximum(processed_image, delta))))
    cross_entropy_error_list.append(cross_entropy_error)
  cross_entropy_error_mean = np.mean(cross_entropy_error_list)
  return cross_entropy_error_mean

def calc_derivative_softmax(processed_images, labels):
  derivative_softmax_list = []
  for i in range(batch_size):
    processed_image, label = processed_images[i].reshape(-1), labels[i]
    label_vector = np.zeros(output_node_size)
    label_vector[label] = 1
    derivative_softmax_list.append(list((processed_image - label_vector)/batch_size))
  return derivative_softmax_list

def calc_derivative_ReLU(derivative_y):
  return derivative_y * ReLU_mask

def calc_derivative_dropout(derivative_y):
  return derivative_y * dropout_mask

def get_batch(): # バッチの抽出
  indexs = np.random.choice(len(train_images), size=batch_size, replace=False)
  batchs, labels = train_images[indexs], train_labels[indexs]
  return batchs, labels

def dropout_layer(x, nonactive_ratio):
  if training_flag:
    mask = np.random.rand(*x.shape) > nonactive_ratio
    return x * mask, mask
  else:
    return (1-nonactive_ratio) * x

def input_layer(images):
  images = np.array(images, dtype=float)
  images /= 255 # pixel normalization
  if training_flag:
    trans_images = np.reshape(images, (len(images), input_node_size))
    dropout_images, _ = dropout_layer(trans_images, input_nonactive_ratio)
  else:
    trans_images = np.reshape(images, (1, input_node_size))
    dropout_images = dropout_layer(trans_images, input_nonactive_ratio)
  return dropout_images

def inner_layer(images):
  global dropout_mask
  affine_images = (np.dot(parameters["W_1"], images.T) + parameters["b_1"]).T
  activate_images = ReLU_function(affine_images)
  if training_flag:
    dropout_images, dropout_mask = dropout_layer(activate_images, inner_nonactive_ratio)
  else:
    dropout_images = dropout_layer(activate_images, inner_nonactive_ratio)
  return dropout_images

def output_layer(images):
  affine_images = (np.dot(parameters["W_2"], images.T) + parameters["b_2"]).T
  softmax_images = softmax_function(affine_images)
  return softmax_images

def main():
  global dropout_mask, training_flag

  # 画像データの準備
  load_image()

  choice = input("use parameter file ? [yes/no]: ")
  if choice == "yes":
    file_name =  input("input the parameter file name: ")
    load_parameters(file_name)
  elif choice == "no":
    init_parameters()

  # 各バッチに対する損失を保持しておく配列
  training_loss_list = []

  graph = []

  # 学習
  for i in tqdm(range(1, epoch+1)):
    with tqdm(total=(len(train_images)//batch_size), leave=False) as pbar:
      for j in range(len(train_images)//batch_size):
        pbar.set_description('Epoch {}'.format(i))
        dropout_mask = []
        batchs, labels = get_batch()

        x = input_layer(batchs)
        y = inner_layer(x)
        processed_batchs = output_layer(y)

        derivative_softmax = np.array(calc_derivative_softmax(processed_batchs, labels))
        derivative_X_2 = (np.dot(parameters["W_2"].T, derivative_softmax.T)).T
        derivative_W_2 = np.dot(derivative_softmax.T, y)
        derivative_b_2 = ((np.sum(derivative_softmax.T, axis=1)).reshape(output_node_size, 1))
        derivative_dropout = calc_derivative_dropout(derivative_X_2)
        derivative_ReLU = calc_derivative_ReLU(derivative_dropout)
        derivative_X_1 = np.dot(parameters["W_1"].T, derivative_ReLU.T)
        derivative_W_1 = np.dot(derivative_ReLU.T, x)
        derivative_b_1 = ((np.sum(derivative_ReLU.T, axis=1)).reshape(inner_node_size, 1))

        parameters["W_1"] -= (learning_rate*derivative_W_1)
        parameters["W_2"] -= (learning_rate*derivative_W_2)
        parameters["b_1"] -= (learning_rate*derivative_b_1)
        parameters["b_2"] -= (learning_rate*derivative_b_2)

        cross_entropy_error = loss_function(processed_batchs, labels)
        training_loss_list.append(cross_entropy_error)
        pbar.update(1)

    cross_entropy_error_mean = np.mean(training_loss_list[(i-1)*(len(train_images)//batch_size):len(training_loss_list)-1])
    graph.append(cross_entropy_error_mean)
    tqdm.write(f"The loss in epoch{i} is {cross_entropy_error_mean}.")

  # accuracy を計算
  correct_num = 0
  training_flag = False
  for i in range(10000):
    image = test_images[i]
    processed_image = output_layer(inner_layer(input_layer(image)))
    prediction = np.argmax(processed_image)
    if(prediction == test_labels[i]):
      correct_num += 1
  print(correct_num/10000)

  joblib.dump(graph, "taskA2.txt",compress=3)
  plot_figure(graph)

  if(input("save the parameter ? [yes/no]: ") == "yes"):
    file_name = input('Input the file name: ')
    np.savez(file_name, W_1=parameters["W_1"], W_2=parameters["W_2"], b_1=parameters["b_1"], b_2=parameters["b_2"])


if __name__ ==  '__main__':
  main()
