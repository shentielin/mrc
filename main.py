import os
from tqdm import tqdm
import sys
import numpy as np
import torch as t
from utils import calcu_acc, generate_submit
from models.bert import Bert4MRC
from data.dataset import MRC
from torch.utils.data import DataLoader

def dev(model, dataloader):

    model.eval()

    pred = []
    with t.no_grad():
        for ii, data in enumerate(tqdm(dataloader)):
            input, label = data
            score = model(input)
            pred.append(score)
    if len(list(pred[-1].size())) == 1:
        pred[-1] = pred[-1].unsqueeze(0)
    pred = t.cat(pred)

    pred = pred.cpu().numpy()
    acc = calcu_acc(pred)

    model.train()
    return acc

def test(model):

    model.eval()

    test_data = MRC(sign='test')
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

    pred = []
    with t.no_grad():
        for ii, data in enumerate(tqdm(test_loader)):
            input, label = data
            score = model(input)
            pred.append(score)
    if len(list(pred[-1].size())) == 1:
        pred[-1] = pred[-1].unsqueeze(0)
    pred = t.cat(pred)

    pred = pred.cpu().numpy()
    generate_submit(pred)

    model.train()
    return 0

def train():

    model = Bert4MRC().cuda()

    train_data = MRC()
    dev_data = MRC(sign='dev')
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

    criterion = t.nn.CrossEntropyLoss()
    max_acc = 0

    for epoch in range(1, 100):
        print('epoch-' + str(epoch))
        optimizer = t.optim.Adam(model.parameters(), lr=2e-5)
        sum_loss = 0
        for data, label in tqdm(train_loader):
            label = label.cuda()
            optimizer.zero_grad()
            try:
                output = model(data)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print('oom error!')
                    if hasattr(t.cuda, 'empty_cache'):
                        continue
                else:
                    raise e
            loss = criterion(output, label)
            sum_loss += loss

            loss.backward()
            optimizer.step()

        print(sum_loss)
        acc = dev(model, dev_loader)
        print(acc)
        if acc>max_acc:
            max_acc = acc
            test(model)

    print('dev_result = ' + str(max_acc))



os.environ["CUDA_VISIBLE_DEVICES"] = "3"
train()