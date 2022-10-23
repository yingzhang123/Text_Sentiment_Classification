import torch
import torch.nn as nn
from torch import optim
from models import Model
from datasets import data_loader, text_CLS
from configs import Config

cfg = Config()

#读取数据
data_path = "sources/weibo_senti_100k.csv"
data_stop_path = "sources/hit_stopword"
dict_path = "sources/dict"

dataset = text_CLS(dict_path, data_path, data_stop_path)
train_dataloader = data_loader(dataset, cfg)

cfg.pad_size = dataset.max_len_seq

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)
#加载模型,保存好的模型
model_text_cls.load_state_dict(torch.load("models/10.pth"))


for i, batch in enumerate(train_dataloader):
    label, data = batch
    data = torch.tensor(data).to(cfg.devices)
    label = torch.tensor(label,dtype=torch.int64).to(cfg.devices)
    pred_softmax = model_text_cls.forward(data)

    #print(pred_softmax)
    print(label)
    pred = torch.argmax(pred_softmax, dim=1)
    print(pred)

    #统计准确率
    out = torch.eq(pred,label)
    print(out.sum() * 1.0 / pred.size()[0])

