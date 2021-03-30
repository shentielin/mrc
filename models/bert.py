from pytorch_transformers import BertTokenizer, BertModel, BertConfig
import torch as t

class BasicModule(t.nn.Module):
   '''
   封装了nn.Module，主要提供save和load两个方法
   '''

   def __init__(self,opt=None):
       super(BasicModule,self).__init__()
       self.model_name = str(type(self)) # 模型的默认名字

   def load(self, path):
       '''
       可加载指定路径的模型
       '''
       self.load_state_dict(t.load(path))

   def save(self, name=None):
       '''
       保存模型，默认使用“模型名字+时间”作为文件名，
       如AlexNet_0710_23:57:29.pth
       '''
       if name is None:
           name = 'checkpoints/' + 'best_model' + '.pth'
       t.save(self.state_dict(), name)
       print("save in to " + name)
       return name

class Bert4MRC(BasicModule):
    def __init__(self):

        super(Bert4MRC, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('/home/stl/PTM/base/vocab.txt')
        config = BertConfig.from_pretrained('/home/stl/PTM/base/config.json')
        self.bert = BertModel.from_pretrained(config=config, pretrained_model_name_or_path='/home/stl/PTM/base/pytorch_model.bin')
        self.dense = t.nn.Linear(768, 2)

    def forward(self, input):

        temp = []
        for i,j in zip(input[0], input[1]):

            k = self.tokenizer.encode(text=i, text_pair=j, add_special_tokens=True)
            k_len = min(len(k), 200)
            temp.append(k[:k_len])

        length = max([len(j) for j in temp])
        for j in range(len(temp)):
            temp[j] = temp[j] + ([0] * (length - len(temp[j])))

        input_ids = t.tensor(temp, requires_grad=False).cuda()

        outputs = self.bert(input_ids)
        outputs = outputs[0]           # batch_size * sequence_length * hidden_length
        outputs = outputs[:, :1, :]

        outputs = self.dense(outputs)
        outputs = t.squeeze(outputs)
        if outputs.dim()==1:
            outputs = outputs.unsqueeze(0)
        return outputs