import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# padding_idx：padding_idx (python:int, optional) – 填充id，比如，输入长度为100，但是每次的句子长度并不一样，后面就需要用统一的数字填充，
#                                     而这里就是指定这个数字，这样，网络在遇到填充id时，就不会计算其与其它符号的相关性。（初始化为0）
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 词嵌入层
        self.embeding = nn.Embedding(config.n_vocab,  # 字典大小：congif.n_vocab，表示词典中词的数量
                                     embedding_dim=config.embed_size,  # 词嵌入的输出大小，就是每个词经过embedding后用多少位向量表示。表示每个词对应的向量维度
                                     padding_idx=config.n_vocab - 1)  # padding_idx ,pad
        # lstm层
        self.lstm = nn.LSTM(input_size=config.embed_size,  # 输入大小，即每个词的维度
                            hidden_size=config.hidden_size,  # 隐藏层输出大小
                            num_layers=config.num_layers,  # lstm的层数
                            bidirectional=True,  # 双向lstm层
                            batch_first=True,  # 数据结构：[batch_size,seq_len,input_size]
                            dropout=config.dropout)  # 防止过拟合
        # 卷积层
        self.maxpooling = nn.MaxPool1d(config.pad_size)  # 一维卷积。积核长度：
        # 全连接层
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_size,  # 因为是双向LSTM，所以要*2，
                            config.num_classes)  # 第二个为预测的类别数

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.embeding(x)  # 输出为[batchsize, seqlen, embed_size] 标准RNN网络的输入
        # print("embed.size:",embed.size())
        out, _ = self.lstm(embed)  # out的shape:[batch_size, seq_len, hidden_size*2]
        # print("lstm层的输出size:",out.size())
        # torch.cat((x,y),dim)   在dim上拼接,x,y
        out = torch.cat((embed, out), 2)  # 这里解析全连接层输入大小为 config.hidden_size * 2 + cinfig.embed_size。
        # [batch_size,seg_len,config.hidden_size * 2 + cinfig.embed_size]
        # print("cat后的size:",out.size())
        out = F.relu(out)  # 经过relu层，增加非线性表达能力
        # print("relu层的out.size:",out.size())
        out = out.permute(0, 2, 1)  # 交换维度
        # print("交换维度后的out.size:",out.size())
        out = self.maxpooling(out).reshape(out.size()[0], -1)  # 转化为2维tensor
        # print("MaxPooling后的out.size:",out.size())
        out = self.fc(out)
        # print("全连接层的out.size:",out.size())
        out = self.softmax(out)
        # print("softmax后的out.size:",out.size())
        return out


# 测试网络是否正确
if __name__ == '__main__':
    cfg = Config()
    cfg.pad_size = 640
    model_textcls = Model(config=cfg)
    input_tensor = torch.tensor([i for i in range(640)]).reshape([1, 640])
    out_tensor = model_textcls.forward(input_tensor)
    print(out_tensor.size())
    print(out_tensor)