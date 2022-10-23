import torch
import torch.nn as nn
from torch import optim
from models import Model
from datasets import data_loader, text_CLS
from configs import Config

cfg = Config()

data_path = "weibo_senti_100k.csv"
data_stop_path = "hit_stopwords.txt"
dict_path = "dict"

dataset = text_CLS(dict_path, data_path, data_stop_path)
train_dataloader = data_loader(dataset, cfg)

cfg.pad_size = dataset.max_len_seq  #

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)

loss_func = nn.CrossEntropyLoss()  # 损失函数。交叉熵损失函数

optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learn_rate)  # 定义优化器

for epoch in range(cfg.num_epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.tensor(data).to(cfg.devices)
        label = torch.tensor(label).to(cfg.devices)

        optimizer.zero_grad()
        pred = model_text_cls.forward(data)
        loss_val = loss_func(pred, label)

        print("epoch is {},ite is {},val is {}".format(epoch, i, loss_val))
        loss_val.backward()  # 后向传播
        optimizer.step()  # 更新参数

    if epoch % 10 == 0:
        torch.save(model_text_cls.state_dict(), "../models/{}.pth".format(epoch))
