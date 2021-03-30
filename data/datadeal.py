import json

train = json.load(open('data/train.json', encoding='utf-8'))
test = json.load(open('data/validation.json', encoding='utf-8'))

train_sent = []
train_label = []
valid_sent = []
valid_label = []
test_sent = []

for i in train[:int(0.7*len(train))]:
    questions = i['Questions']
    for question in questions:
        for choice in question['Choices']:
            temp = [question['Question'], choice[2:]]
            train_sent.append(temp)
            train_label.append(0) if choice[0]!=question['Answer'] else train_label.append(1)

valid_sign = []
sign = 0
for i in train[int(0.7*len(train)):]:
    questions = i['Questions']
    for question in questions:
        for choice in question['Choices']:
            temp = [question['Question'], choice[2:]]
            valid_sign.append(sign)
            valid_sent.append(temp)
        sign += 1
        valid_label.append(question['Answer'])

test_sign = []
sign = 0
test_qid = []
for i in test:
    questions = i['Questions']
    for question in questions:
        test_qid.append(question['Q_id'])
        for choice in question['Choices']:
            temp = [question['Question'], choice[2:]]
            test_sign.append(sign)
            test_sent.append(temp)
        sign += 1

print(len(train_sent))
print(len(train_label))
print(train_sent[:5])
print(train_label)
print(len(valid_sent))
print(len(valid_label))
print(valid_sent[:5])
print(valid_label)
print(len(test_sent))
print(test_sent[:5])
print(len(valid_sign))
print(valid_sign)
print(len(test_sign))
print(test_sign)
print(len(test_qid))
print(test_qid)

def load_json(file, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(file, f)

# load_json(train_sent, 'train_sent.json')
# load_json(train_label, 'train_label.json')
# load_json(valid_sent, 'valid_sent.json')
# load_json(valid_label, 'data/valid_label.json')
# load_json(test_sent, 'test_sent.json')
# load_json(valid_sign, 'data/valid_sign.json')
# load_json(test_sign, 'data/test_sign.json')
load_json(test_qid, 'data/test_qid.json')