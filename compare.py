import matplotlib.pyplot as plt
import numpy as np
import joblib

'''def plot_figure(list1, list2, label1, label2):
  plt.figure(figsize=(8, 6))
  plt.plot(np.arange(1, len(list1) + 1), list1, label=label1)
  plt.plot(np.arange(1, len(list2) + 1), list2, label=label2)
  plt.xlabel('epoch')
  plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  plt.ylabel('loss')
  plt.legend(prop = {"family" : "Hiragino sans"})
  #plt.title('sigmoidとReLUの違い', fontsize=20) # タイトル
  plt.grid()
  plt.show()
'''
def plot_figure(list1, list2, list3, list4, list5, list6, label1, label2, label3, label4, label5, label6):
  plt.figure(figsize=(8, 6))
  plt.plot(np.arange(1, len(list1) + 1), list1, label=label1)
  plt.plot(np.arange(1, len(list2) + 1), list2, label=label2)
  plt.plot(np.arange(1, len(list3) + 1), list3, label=label3)
  plt.plot(np.arange(1, len(list4) + 1), list4, label=label4)
  plt.plot(np.arange(1, len(list5) + 1), list5, label=label5)
  plt.plot(np.arange(1, len(list6) + 1), list6, label=label6)
  plt.xlabel('epoch')
  plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  plt.ylabel('loss')
  plt.legend(prop = {"family" : "Hiragino sans"})
  #plt.title('sigmoidとReLUの違い', fontsize=20) # タイトル
  plt.grid()
  plt.show()

def main():
  list1 = joblib.load("taskA1.txt")
  list2 = joblib.load("taskA4_momentum.txt")
  list3 = joblib.load("taskA4_RMSProp.txt")
  list4 = joblib.load("taskA4_AdaGrad.txt")
  list5 = joblib.load("taskA4_AdaDelta.txt")
  list6 = joblib.load("taskA4_Adam.txt")
  plot_figure(list1, list2, list3, list4, list5, list6, "SGD", "慣性項付きSGD", "RMSProp", "AdaGrad", "AdaDelta", "Adam")

if __name__ ==  '__main__':
  main()

