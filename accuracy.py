f1 = open('predict.txt', 'r')
f2 = open('answer.txt', 'r')

predict = f1.readlines()
answer = f2.readlines()
print(len(answer))
correct_count = 0
for i in range(1000):
  if(predict[i] == answer[i]):
    correct_count += 1

print(correct_count)