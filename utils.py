import json
import csv
import numpy as np

def calcu_acc(pred):
    label = json.load(open('data/valid_label.json'))
    valid_sign = json.load(open('data/valid_sign.json'))
    key = ['A', 'B', 'C', 'D']
    result = []
    temp = []
    sign = 0
    for i,j in enumerate(valid_sign):
        if j==sign:
            temp.append(pred[i][1])
        else:
            index = np.argmax(temp)
            result.append(key[index])
            temp = [pred[i][1]]
            sign += 1
    if temp!=[]:
        index = np.argmax(temp)
        result.append(key[index])
    correct = 0
    for i,j in zip(result, label):
        if i==j:
            correct += 1
    return correct/len(label)

def generate_submit(pred):
    test_sign = json.load(open('data/test_sign.json'))
    key = ['A', 'B', 'C', 'D']
    result = []
    temp = []
    sign = 0
    for i,j in enumerate(test_sign):
        if j==sign:
            temp.append(pred[i][1])
        else:
            index = np.argmax(temp)
            result.append(key[index])
            temp = [pred[i][1]]
            sign += 1
    if temp!=[]:
        index = np.argmax(temp)
        result.append(key[index])
    test_qid = json.load(open('data/test_qid.json'))
    print(len(test_qid))
    print(len(result))

    with open('data/result.csv', 'w', encoding='utf-8', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i,j in zip(test_qid, result):
            writer.writerow([i, j])