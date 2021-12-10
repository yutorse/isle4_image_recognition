import numpy as np
import mnist

input_node_size = 784
inner_node_size = 100
output_node_size = 10

parameters = {}

input_nonactive_ratio = 0.2
inner_nonactive_ratio = 0.5

def ReLU_function(t): # ReLU関数
  return np.maximum(t, 0)

def sigmoid_function(t):
  return 1/(1 + np.exp(-t))

def softmax_function(a):
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

def dropout_layer(x, nonactive_ratio):
  return (1-nonactive_ratio) * x

def input_layer(image): #入力層
  image = np.array(image)
  trans_image = np.reshape(image, (1, input_node_size))
  dropout_image = dropout_layer(trans_image, input_nonactive_ratio)
  return dropout_image

def inner_layer(image): #中間層
  affine_image = (np.dot(parameters["W_1"], image.T) + parameters["b_1"]).T
  activate_image = ReLU_function(affine_image)
  dropout_image = dropout_layer(activate_image, inner_nonactive_ratio)
  return dropout_image

def output_layer(image): #出力層
  affine_image = (np.dot(parameters["W_2"], image.T) + parameters["b_2"]).T
  softmax_image = np.array(list(map(softmax_function, affine_image)))
  return softmax_image

def main():
  # 画像データの準備
  images = np.loadtxt("le4MNIST_X.txt")
  
  file = open('predict.txt', 'a')

  load_parameters = np.load("parameters.npz")
  parameters["W_1"] = load_parameters["W_1"]
  parameters["W_2"] = load_parameters["W_2"]
  parameters["b_1"] = load_parameters["b_1"]
  parameters["b_2"] = load_parameters["b_2"]

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
