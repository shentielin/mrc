import json
from torch.utils import data as Data

class MRC(Data.Dataset):
    def __init__(self, sign='train_a'):
        self.sign = sign
        if self.sign == 'train_a':
            data = json.load(open('data/data_a/train_sent.json'))
            label = json.load(open('data/data_a/train_label.json'))
        elif self.sign == 'dev_a':
            data = json.load(open('data/data_a/valid_sent.json'))
            label = [1] * len(data)
        elif self.sign == 'test_a':
            data = json.load(open('data/data_a/test_sent.json'))
            label = [1] * len(data)
        elif self.sign == 'train_b':
            data = json.load(open('data/data_b/train_sent.json'))
            label = json.load(open('data/data_b/train_label.json'))
        elif self.sign == 'dev_b':
            data = json.load(open('data/data_b/valid_sent.json'))
            label = [1] * len(data)
        elif self.sign == 'test_b':
            data = json.load(open('data/data_b/test_sent.json'))
            label = [1] * len(data)

        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)