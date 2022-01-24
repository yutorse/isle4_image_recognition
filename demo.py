import numpy as np
import mnist
import matplotlib.pyplot as plt
import random
import time
from tqdm import tqdm

train_images, train_labels, test_images, test_labels = None, None, None, None

input_node_size = 784
inner_node_size = 100
output_node_size = 10
batch_size = 100
epoch = 10

parameters = {}

learning_rate = 0.1

'''
# 慣性項付きSGD
Momentum_SGD_lr = 0.1
delta_W1 = delta_W2 = delta_b1 = delta_b2 = 0
alpha = 0.9

# AdaGrad
AdaGrad_lr = 0.01
h_W1 = h_W2 = h_b1 = h_b2 = 1e-8

# RMSProp
RMSProp_lr = 0.01
epsilon = 1e-8
rho = 0.9
h_W1 = h_W2 = h_b1 = h_b2 = 0

# AdaDelta
rho = 0.95
epsilon = 1e-6
h_W1 = h_W2 = h_b1 = h_b2 = 0
s_W1 = s_W2 = s_b1 = s_b2 = 0
'''
# Adam
t = 0
m_W1 = m_W2 = m_b1 = m_b2 = m_beta = m_gamma = 0
v_W1 = v_W2 = v_b1 = v_b2 = v_beta = v_gamma = 0
alpha = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

# ReLU
ReLU_mask = []

# dropout
training_flag = True
input_nonactive_ratio = 0.2
inner_nonactive_ratio = 0.5
dropout_mask = []

# BN parameter
beta = np.zeros(inner_node_size)
gamma = np.ones(inner_node_size)
ave_list = np.array([])
var_list = np.array([])
BN_x = np.array([])
BN_x_hat = np.array([])

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

def calc_derivative_BN(derivative_y):
  delta = 1e-7
  ave = ave_list[-1]
  var = var_list[-1]
  derivative_x_hat = derivative_y * gamma
  derivative_var = np.sum((derivative_x_hat * (BN_x - ave) * (-1/2) * ((var + delta)**(-3/2))), axis=0)
  derivative_ave = (np.sum((derivative_x_hat * (-1) * ((var + delta)**(-1/2))), axis=0)) + (derivative_var * (1/batch_size) * (np.sum(((-2) * (BN_x - ave)), axis=0)))
  derivative_x = (derivative_x_hat * ((var + delta)**(-1/2))) + (derivative_var * (2/batch_size) * (BN_x - ave)) + ((1/batch_size) * derivative_ave)
  derivative_gamma = np.sum((derivative_y * BN_x_hat), axis=0)
  derivative_beta = np.sum(derivative_y, axis=0)
  return derivative_x, derivative_beta, derivative_gamma

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

def batch_normalization(x):
  global BN_x, BN_x_hat, ave_list, var_list
  delta = 1e-7
  if training_flag:
    BN_x = x
    ave = BN_x.mean(axis = 0)
    ave_list = np.append(ave_list, ave)
    var = BN_x.var(axis = 0)
    var_list = np.append(var_list, var)
    BN_x_hat = (BN_x - ave)/(np.sqrt(var + delta))
    y = gamma * BN_x_hat + beta
    return y
  else:
    ave = np.mean(ave_list)
    var = np.mean(var_list)
    return gamma * x / (np.sqrt(var+delta)) + beta - (gamma * ave / (np.sqrt(var+delta)))

def input_layer(images):
  images = np.array(images, dtype=float)
  images /= 255 # pixel normalization
  if training_flag:
    trans_images = np.reshape(images, (len(images), input_node_size))
  else:
    trans_images = np.reshape(images, (1, input_node_size))
  return trans_images
  '''
  if training_flag:
    trans_images = np.reshape(images, (len(images), input_node_size))
    dropout_images, _ = dropout_layer(trans_images, input_nonactive_ratio)
  else:
    trans_images = np.reshape(images, (1, input_node_size))
    dropout_images = dropout_layer(trans_images, input_nonactive_ratio)
  return dropout_images
  '''

def inner_layer(images):
  global dropout_mask
  affine_images = (np.dot(parameters["W_1"], images.T) + parameters["b_1"]).T
  activate_images = ReLU_function(affine_images)
  '''
  BN_images = batch_normalization(affine_images)
  activate_images = ReLU_function(BN_images)
  '''
  '''
  if training_flag:
    dropout_images, dropout_mask = dropout_layer(activate_images, inner_nonactive_ratio)
  else:
    dropout_images = dropout_layer(activate_images, inner_nonactive_ratio)
  return dropout_images
  '''
  return activate_images

def output_layer(images):
  affine_images = (np.dot(parameters["W_2"], images.T) + parameters["b_2"]).T
  softmax_images = softmax_function(affine_images)
  return softmax_images

def main():
  global dropout_mask, training_flag, beta, gamma
  global delta_W1, delta_W2, delta_b1, delta_b2 #慣性項付きSGD
  global h_W1, h_W2, h_b1, h_b2 #AdaGrad, RMSProp, AdaDelta
  global s_W1, s_W2, s_b1, s_b2 #AdaDelta
  global t, m_W1, m_W2, m_b1, m_b2, m_beta, m_gamma, v_W1, v_W2, v_b1, v_b2, v_beta, v_gamma #Adam

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

        # A1まで
        derivative_softmax = np.array(calc_derivative_softmax(processed_batchs, labels))
        derivative_X_2 = (np.dot(parameters["W_2"].T, derivative_softmax.T)).T
        derivative_W_2 = np.dot(derivative_softmax.T, y)
        derivative_b_2 = ((np.sum(derivative_softmax.T, axis=1)).reshape(output_node_size, 1))
        derivative_ReLU = calc_derivative_ReLU(derivative_X_2)
        derivative_X_1 = np.dot(parameters["W_1"].T, derivative_ReLU.T)
        derivative_W_1 = np.dot(derivative_ReLU.T, x)
        derivative_b_1 = ((np.sum(derivative_ReLU.T, axis=1)).reshape(inner_node_size, 1))
        # A2まで
        '''
        derivative_softmax = np.array(calc_derivative_softmax(processed_batchs, labels))
        derivative_X_2 = (np.dot(parameters["W_2"].T, derivative_softmax.T)).T
        derivative_W_2 = np.dot(derivative_softmax.T, y)
        derivative_b_2 = ((np.sum(derivative_softmax.T, axis=1)).reshape(output_node_size, 1))
        derivative_dropout = calc_derivative_dropout(derivative_X_2)
        derivative_ReLU = calc_derivative_ReLU(derivative_dropout)
        derivative_X_1 = np.dot(parameters["W_1"].T, derivative_ReLU.T)
        derivative_W_1 = np.dot(derivative_ReLU.T, x)
        derivative_b_1 = ((np.sum(derivative_ReLU.T, axis=1)).reshape(inner_node_size, 1))
        '''

        # A3まで
        '''
        derivative_softmax = np.array(calc_derivative_softmax(processed_batchs, labels))
        derivative_X_2 = (np.dot(parameters["W_2"].T, derivative_softmax.T)).T
        derivative_W_2 = np.dot(derivative_softmax.T, y)
        derivative_b_2 = ((np.sum(derivative_softmax.T, axis=1)).reshape(output_node_size, 1))
        derivative_dropout = calc_derivative_dropout(derivative_X_2)
        derivative_ReLU = calc_derivative_ReLU(derivative_dropout)
        derivative_BN, derivative_beta, derivative_gamma = calc_derivative_BN(derivative_ReLU)
        derivative_X_1 = np.dot(parameters["W_1"].T, derivative_BN.T)
        derivative_W_1 = np.dot(derivative_BN.T, x)
        derivative_b_1 = ((np.sum(derivative_BN.T, axis=1)).reshape(inner_node_size, 1))
        '''
        
        # SGD
        '''
        parameters["W_1"] -= (learning_rate*derivative_W_1)
        parameters["W_2"] -= (learning_rate*derivative_W_2)
        parameters["b_1"] -= (learning_rate*derivative_b_1)
        parameters["b_2"] -= (learning_rate*derivative_b_2)
        #beta -= (learning_rate*derivative_beta)
        #gamma -= (learning_rate*derivative_gamma)
        '''
        
        # 慣性項付きSGD
        '''
        delta_W1 = alpha * delta_W1 - Momentum_SGD_lr * derivative_W_1
        delta_W2 = alpha * delta_W2 - Momentum_SGD_lr * derivative_W_2
        delta_b1 = alpha * delta_b1 - Momentum_SGD_lr * derivative_b_1
        delta_b2 = alpha * delta_b2 - Momentum_SGD_lr * derivative_b_2
        parameters["W_1"] += delta_W1
        parameters["W_2"] += delta_W2
        parameters["b_1"] += delta_b1
        parameters["b_2"] += delta_b2
        '''
        
        # AdaGrad
        '''
        h_W1 += derivative_W_1 * derivative_W_1
        h_W2 += derivative_W_2 * derivative_W_2
        h_b1 += derivative_b_1 * derivative_b_1
        h_b2 += derivative_b_2 * derivative_b_2
        parameters["W_1"] -= AdaGrad_lr * derivative_W_1 / np.sqrt(h_W1)
        parameters["W_2"] -= AdaGrad_lr * derivative_W_2 / np.sqrt(h_W2)
        parameters["b_1"] -= AdaGrad_lr * derivative_b_1 / np.sqrt(h_b1)
        parameters["b_2"] -= AdaGrad_lr * derivative_b_2 / np.sqrt(h_b2)
        '''
        
        # RMSProp
        '''
        h_W1 = rho * h_W1 + (1-rho) * derivative_W_1 * derivative_W_1
        h_W2 = rho * h_W2 + (1-rho) * derivative_W_2 * derivative_W_2
        h_b1 = rho * h_b1 + (1-rho) * derivative_b_1 * derivative_b_1
        h_b2 = rho * h_b2 + (1-rho) * derivative_b_2 * derivative_b_2
        parameters["W_1"] -= RMSProp_lr * derivative_W_1 / (np.sqrt(h_W1) + epsilon)
        parameters["W_2"] -= RMSProp_lr * derivative_W_2 / (np.sqrt(h_W2) + epsilon)
        parameters["b_1"] -= RMSProp_lr * derivative_b_1 / (np.sqrt(h_b1) + epsilon)
        parameters["b_2"] -= RMSProp_lr * derivative_b_2 / (np.sqrt(h_b2) + epsilon)
        '''
        
        # AdaDelta
        '''
        h_W1 = rho * h_W1 + (1-rho) * derivative_W_1 * derivative_W_1
        h_W2 = rho * h_W2 + (1-rho) * derivative_W_2 * derivative_W_2
        h_b1 = rho * h_b1 + (1-rho) * derivative_b_1 * derivative_b_1
        h_b2 = rho * h_b2 + (1-rho) * derivative_b_2 * derivative_b_2
        delta_W1 = -np.sqrt((s_W1 + epsilon)/(h_W1 + epsilon)) * derivative_W_1
        delta_W2 = -np.sqrt((s_W2 + epsilon)/(h_W2 + epsilon)) * derivative_W_2
        delta_b1 = -np.sqrt((s_b1 + epsilon)/(h_b1 + epsilon)) * derivative_b_1
        delta_b2 = -np.sqrt((s_b2 + epsilon)/(h_b2 + epsilon)) * derivative_b_2
        s_h1 = rho * s_W1 + (1-rho) * delta_W1 * delta_W1
        s_h2 = rho * s_W2 + (1-rho) * delta_W2 * delta_W2
        s_b1 = rho * s_b1 + (1-rho) * delta_b1 * delta_b1
        s_b2 = rho * s_b2 + (1-rho) * delta_b2 * delta_b2
        parameters["W_1"] += delta_W1
        parameters["W_2"] += delta_W2
        parameters["b_1"] += delta_b1
        parameters["b_2"] += delta_b2
        '''
        
        # Adam
        t += 1
        m_W1 = beta_1 * m_W1 + (1-beta_1) * derivative_W_1
        m_W2 = beta_1 * m_W2 + (1-beta_1) * derivative_W_2
        m_b1 = beta_1 * m_b1 + (1-beta_1) * derivative_b_1
        m_b2 = beta_1 * m_b2 + (1-beta_1) * derivative_b_2
        #m_beta = beta_1 * m_beta + (1-beta_1) * derivative_beta
        #m_gamma = beta_1 * m_gamma + (1-beta_1) * derivative_gamma
        v_W1 = beta_2 * v_W1 + (1-beta_2) * derivative_W_1 * derivative_W_1
        v_W2 = beta_2 * v_W2 + (1-beta_2) * derivative_W_2 * derivative_W_2
        v_b1 = beta_2 * v_b1 + (1-beta_2) * derivative_b_1 * derivative_b_1
        v_b2 = beta_2 * v_b2 + (1-beta_2) * derivative_b_2 * derivative_b_2
        #v_beta = beta_2 * v_beta + (1-beta_2) * derivative_beta * derivative_beta
        #v_gamma = beta_2 * v_gamma + (1-beta_2) * derivative_gamma * derivative_gamma
        m_W1_hat = m_W1/(1-beta_1**t)
        m_W2_hat = m_W2/(1-beta_1**t)
        m_b1_hat = m_b1/(1-beta_1**t)
        m_b2_hat = m_b2/(1-beta_1**t)
        #m_beta_hat = m_beta/(1-beta_1**t)
        #m_gamma_hat = m_gamma/(1-beta_1**t)
        v_W1_hat = v_W1/(1-beta_2**t)
        v_W2_hat = v_W2/(1-beta_2**t)
        v_b1_hat = v_b1/(1-beta_2**t)
        v_b2_hat = v_b2/(1-beta_2**t)
        #v_beta_hat = v_beta/(1-beta_2**t)
        #v_gamma_hat = v_gamma/(1-beta_2**t)
        parameters["W_1"] -= alpha * m_W1_hat / (np.sqrt(v_W1_hat) + epsilon)
        parameters["W_2"] -= alpha * m_W2_hat / (np.sqrt(v_W2_hat) + epsilon)
        parameters["b_1"] -= alpha * m_b1_hat / (np.sqrt(v_b1_hat) + epsilon)
        parameters["b_2"] -= alpha * m_b2_hat / (np.sqrt(v_b2_hat) + epsilon)
        #beta -= alpha * m_beta_hat / (np.sqrt(v_beta_hat) + epsilon)
        #gamma -= alpha * m_gamma_hat / (np.sqrt(v_gamma_hat) + epsilon)

        cross_entropy_error = loss_function(processed_batchs, labels)
        training_loss_list.append(cross_entropy_error)
        pbar.update(1)

    cross_entropy_error_mean = np.mean(training_loss_list[(i-1)*(len(train_images)//batch_size):len(training_loss_list)-1])
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

  plot_figure(training_loss_list)

  if(input("save the parameter ? [yes/no]: ") == "yes"):
    file_name = input('Input the file name: ')
    np.savez(file_name, W_1=parameters["W_1"], W_2=parameters["W_2"], b_1=parameters["b_1"], b_2=parameters["b_2"])
    #np.savez(file_name, W_1=parameters["W_1"], W_2=parameters["W_2"], b_1=parameters["b_1"], b_2=parameters["b_2"], beta=beta, gamma=gamma, ave=np.mean(ave_list), var=np.mean(var_list))


if __name__ ==  '__main__':
  main()
