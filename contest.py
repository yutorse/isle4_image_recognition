import numpy as np
import mnist

input_node_size = 784
inner_node_size = 100
output_node_size = 10

parameters = {}

def sigmoid_function(t):
  return 1/(1 + np.exp(-t))

def softmax_function(a):
  max_a = a.max()
  sum = np.sum(np.exp(a - max_a))
  return np.exp(a - max_a) / sum

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
