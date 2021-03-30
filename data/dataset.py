import json
from torch.utils import data as Data

class MRC(Data.Dataset):
    def __init__(self, sign='train'):
        self.sign = sign
        if self.sign == 'train':
            data = json.load(open('data/train_sent.json'))
            label = json.load(open('data/train_label.json'))
        elif self.sign == 'dev':
            data = json.load(open('data/valid_sent.json'))
            label = [1] * len(data)
        else:
            data = json.load(open('data/test_sent.json'))
            label = [1] * len(data)

        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)