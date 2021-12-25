import numpy as np
import mnist

input_node_size = 784
inner_node_size = 1000
output_node_size = 10

parameters = {}
beta, gamma = [], []
ave = 0
var = 0

input_nonactive_ratio = 0.2
inner_nonactive_ratio = 0.5

def ReLU_function(t): # ReLU関数
  return np.maximum(t, 0)

def softmax_function(a):
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

def dropout_layer(x, nonactive_ratio):
  return (1-nonactive_ratio) * x

def batch_normalization(x):
  delta = 1e-7
  return gamma * x * ((var+delta)**(-1/2)) + beta - (gamma * ave * ((var+delta)**(-1/2)))

def input_layer(image): #入力層
  image = np.array(image)
  trans_image = np.reshape(image, (1, input_node_size))
  #dropout_image = dropout_layer(trans_image, input_nonactive_ratio)
  #return dropout_image
  return trans_image

def inner_layer(image): #中間層
  affine_image = (np.dot(parameters["W_1"], image.T) + parameters["b_1"]).T
  #BN_image = batch_normalization(affine_image)
  activate_image = ReLU_function(affine_image)
  #activate_image = ReLU_function(BN_image)
  #dropout_image = dropout_layer(activate_image, inner_nonactive_ratio)
  #return dropout_image
  return activate_image

def output_layer(image): #出力層
  affine_image = (np.dot(parameters["W_2"], image.T) + parameters["b_2"]).T
  softmax_image = np.array(list(map(softmax_function, affine_image)))
  return softmax_image

def main():
  global beta, gamma, ave, var
  
  images = np.loadtxt("le4MNIST_X.txt")
  
  file = open('predict.txt', 'w')

  #load_parameters = np.load("parameters_A3.npz")
  load_parameters = np.load("parameters_A1.npz")
  parameters["W_1"] = load_parameters["W_1"]
  parameters["W_2"] = load_parameters["W_2"]
  parameters["b_1"] = load_parameters["b_1"]
  parameters["b_2"] = load_parameters["b_2"]
  #beta = load_parameters["beta"]
  #gamma = load_parameters["gamma"]
  #ave = load_parameters["ave"]
  #var = load_parameters["var"]

  for i in range(len(images)):
    image = images[i]
    processed_image = output_layer(inner_layer(input_layer(image)))
    prediction = np.argmax(processed_image)
    file.write(str(prediction))
    file.write("\n")
    print(prediction)
    
  file.close()

if __name__ ==  '__main__':
  main()
