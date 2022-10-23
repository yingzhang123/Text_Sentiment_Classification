import torch

# 定义网路的配置类
class Config():
    def __init__(self):
        '''
        self.embeding = nn.Embedding(config.n_vocab,
                                config.embed_size,
                                padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed_size,
                            config.hidden_size,
                              config.num_layers,
                               bidirectional=True, batch_first=True,
                               dropout=config.dropout)
        self.maxpool = nn.MaxPool1d(config.pad_size)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embed_size,
                                  config.num_classes)
        self.softmax = nn.Softmax(dim=1)
        '''
        self.n_vocab = 1002  # 字典长度
        self.embed_size = 128  # 词嵌入表达的大小
        self.hidden_size = 128  # 隐藏层输出大小
        self.num_layers = 3  # lstm网络的层数
        self.dropout = 0.8  #
        self.num_classes = 2  # 二分类问题
        self.pad_size = 32
        self.batch_size = 128
        self.is_shuffle = True
        self.learn_rate = 0.001
        self.num_epochs = 100
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')