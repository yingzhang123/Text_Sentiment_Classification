import numpy as np
import jieba
from torch.utils.data import Dataset, DataLoader


# 传入字典路径，将文件读入内存
def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path).readlines()
    print(dict_list[0])  # '泪,0'
    for item in dict_list:
        item = item.split(",")  # ['泪', '0\n']
        voc_dict[item[0]] = int(item[1].strip())  # item[0]值'泪'  item[1].strip()值为'0'
    # print(voc_dict)
    return voc_dict


# 将数据集进行处理(分词，过滤...)
def load_data(data_path, data_stop_path):
    data_list = open(data_path, encoding='utf-8').readlines()[1:]
    stops_word = open(data_stop_path, encoding='utf-8').readlines()
    stops_word = [line.strip() for line in stops_word]
    stops_word.append(" ")
    stops_word.append("\n")
    voc_dict = {}
    data = []
    max_len_seq = 0  # 统计最长的句子长度
    np.random.shuffle(data_list)
    for item in data_list[:]:
        label = item[0]
        content = item[2:].strip()
        seg_list = jieba.cut(content, cut_all=False)
        seg_res = []
        for seg_item in seg_list:
            if seg_item in stops_word:
                continue
            seg_res.append(seg_item)
            if seg_item in voc_dict.keys():
                voc_dict[seg_item] = voc_dict[seg_item] + 1
            else:
                voc_dict[seg_item] = 1
        if len(seg_res) > max_len_seq:  # 以句子分词词语最长为标准
            max_len_seq = len(seg_res)
        data.append([label, seg_res])  # [标签，分词结果的列表]

        # print(max_len_seq)
    return data, max_len_seq  # 句子分词后，词语最大长度


# 定义Dataset
class text_CLS(Dataset):
    def __init__(self, voc_dict_path, data_path, data_stop_path):
        self.data_path = data_path
        self.data_stop_path = data_stop_path

        self.voc_dict = read_dict(voc_dict_path)  # 返回数据[[label,分词词语列表]，......]
        self.data, self.max_len_seq = load_data(self.data_path, self.data_stop_path)

        np.random.shuffle(self.data)  # 将数据的顺序打乱

    def __len__(self):  # 返回数据集长度
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]  # 句子分词后的词语列表

        input_idx = []
        for word in word_list:
            if word in self.voc_dict.keys():  # 如果词语在自己创建的字典中
                input_idx.append(self.voc_dict[word])  # 将这个单词的词频数放进列表
            else:
                input_idx.append(self.voc_dict["<UNK>"])  # 不在则统一归为其他类(词频太低的归为一类)
        if len(input_idx) < self.max_len_seq:  # 词语长度小于最长长度，则需要用PAD填充
            input_idx += [self.voc_dict["<PAD>"] for _ in range(self.max_len_seq - len(input_idx))]
            # input_idx += [1001 for _ in range(self.max_len_seq - len(input_idx))]
        data = np.array(input_idx)  # 将得到的词频数列表，转化为numpy数据

        return label, data


# 定义DataLoader
def data_loader(dataset, config):
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)


# if __name__ == "__main__":
#     data_path = "../sources/weibo_senti_100k.csv"
#     data_stop_path = "../sources/hit_stopwords.txt"
#     dict_path = "../sources/dict"
#
#     train_dataLoader = data_loader(data_path, data_stop_path, dict_path)
#     for i, batch in enumerate(train_dataLoader):
#         print(batch[0], batch[1].size())
#         print(batch[0], batch[1])
