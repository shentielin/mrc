import json

def generate_data(directory_name):

    path = 'data/original_data/' + directory_name + '/'

    # train
    trainpath = path + 'train.txt'
    train_sentpair, train_label = [], []
    f = open(trainpath, 'r', encoding='utf-8')
    for i in f:
        d = eval(i)
        source_sentence = d['source'][:min(len(d['source']), 200)]
        target_sentence = d['target'][:min(len(d['target']), 200)]
        train_sentpair.append([source_sentence, target_sentence])
        train_label.append(int(d['label'+directory_name[-1].upper()]))

    # valid
    validpath = path + 'valid.txt'
    valid_sentpair, valid_label = [], []
    f = open(validpath, 'r', encoding='utf-8')
    for i in f:
        d = eval(i)
        source_sentence = d['source'][:min(len(d['source']), 200)]
        target_sentence = d['target'][:min(len(d['target']), 200)]
        valid_sentpair.append([source_sentence, target_sentence])
        valid_label.append(int(d['label'+directory_name[-1].upper()]))

    # test
    testpath = path + 'test_with_id.txt'
    test_sentpair, test_id = [], []
    f = open(testpath, 'r', encoding='utf-8')
    for i in f:
        d = eval(i)
        source_sentence = d['source'][:min(len(d['source']), 200)]
        target_sentence = d['target'][:min(len(d['target']), 200)]
        test_sentpair.append([source_sentence, target_sentence])
        test_id.append(d['id'])

    return train_sentpair, train_label, valid_sentpair, valid_label, test_sentpair, test_id

def load_json(file, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(file, f)


train_asent, train_alabel, valid_asent, valid_alabel, test_asent, test_aid = [], [], [], [], [], []
train_bsent, train_blabel, valid_bsent, valid_blabel, test_bsent, test_bid = [], [], [], [], [], []
for i in ['ll-a', 'sl-a', 'ss-a']:
    a, b, c, d, e, f = generate_data(i)
    train_asent += a
    train_alabel += b
    valid_asent += c
    valid_alabel += d
    test_asent += e
    test_aid += f
for i in ['ll-b', 'sl-b', 'ss-b']:
    a, b, c, d, e, f = generate_data(i)
    train_bsent += a
    train_blabel += b
    valid_bsent += c
    valid_blabel += d
    test_bsent += e
    test_bid += f


load_json(train_asent, 'data/data_a/train_sent.json')
load_json(train_alabel, 'data/data_a/train_label.json')
load_json(valid_asent, 'data/data_a/valid_sent.json')
load_json(valid_alabel, 'data/data_a/valid_label.json')
load_json(test_asent, 'data/data_a/test_sent.json')
load_json(test_aid, 'data/data_a/test_id.json')

load_json(train_bsent, 'data/data_b/train_sent.json')
load_json(train_blabel, 'data/data_b/train_label.json')
load_json(valid_bsent, 'data/data_b/valid_sent.json')
load_json(valid_blabel, 'data/data_b/valid_label.json')
load_json(test_bsent, 'data/data_b/test_sent.json')
load_json(test_bid, 'data/data_b/test_id.json')