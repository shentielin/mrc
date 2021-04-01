import json
import csv

def calcu_acc(pred, sign):

    if sign == 'a':
        label = json.load(open('data/data_a/valid_label.json'))
    else:
        label = json.load(open('data/data_b/valid_label.json'))
    result = []
    for i in pred:
        if i[0] > i[1]:
            result.append(0)
        else:
            result.append(1)

    print(result)
    print(label)
    t1, t2, t3 = 0, 0, 0
    for i, j in zip(label, result):
        t1 += i
        t2 += j
        t3 += (i*j)
    print(t1)
    print(t2)
    print(t3)
    try:
        pre = t3 / t1
    except:
        pre = 0
    try:
        rec = t3 / t2
    except:
        rec = 0
    print(pre)
    print(rec)
    try:
        f1 = 2*pre*rec/(pre+rec)
    except:
        f1 = 0

    return f1

def generate_submit(pred, sign):

    if sign == 'a':
        test_id = json.load(open('data/data_a/test_id.json'))
    else:
        test_id = json.load(open('data/data_b/test_id.json'))
    result = []
    for i in pred:
        if i[0] > i[1]:
            result.append(0)
        else:
            result.append(1)

    a = []
    for i,j in zip(test_id, result):
        a.append([i, j])
    if sign == 'a':
        with open('data/data_a/result_a.json', 'w', encoding='utf-8') as f:
            json.dump(a, f)
    else:
        with open('data/data_b/result_b.json', 'w', encoding='utf-8') as f:
            json.dump(a, f)

def merge_submit():

    result_a = json.load(open('data/data_a/result_a.json'))
    result_b = json.load(open('data/data_b/result_b.json'))
    result = result_a + result_b

    with open('data/result.csv', 'w', encoding='utf-8', newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'label'])
        for i in result:
            writer.writerow(i)