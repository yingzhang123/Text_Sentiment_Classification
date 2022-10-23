# 数据来源 https://github.com/SophonPlus/ChineseNlpCorpus/blob/master/datasets/weibo_senti_100k/intro.ipynb
# 数据概览： 10 万多条，带情感标注 新浪微博，正负向评论约各 5 万条

# 停用词字典  https://github.com/goto456/stopwords

import jieba  # 导入中文分词的第三方库，jieba分词

data_path = "../sources/weibo_senti_100k.csv"  # 数据路径
data_stop_path = "../sources/hit_stopwords.txt"  # 停用词数据路径

data_list = open(data_path, encoding='UTF-8').readlines()[1:]  # 读出数据并去掉第一行的介绍标签,      每一行为一个大字符串
stops_word = open(data_stop_path, encoding='UTF-8').readlines()  # 读取停用词内容
stops_word = [line.strip() for line in stops_word]  # 将每行换行符去掉(去掉换行符)，并生成停用词列表
stops_word.append(" ")  # 可以自己根据需要添加停用词
stops_word.append("\n")

voc_dict = {}
min_seq = 1  # 用于过滤词频数
top_n = 1000
UNK = "<UNK>"
PAD = "<PAD>"

print(data_list[0])

# 对data_list进行分词的处理
# for item in data_list[:100]:      使用前100条数据测试，100000条数据太多
for item in data_list:
    label = item[0]  # 字符串的第一个为标签
    content = item[2:].strip()  # 从第三项开始为文本内容, strip()去掉最后的换行符
    seg_list = jieba.cut(content, cut_all=False)  # 调用结巴分词对每一行文本内容进行分词

    seg_res = []
    # 打印分词结果
    for seg_item in seg_list:
        if seg_item in stops_word:  # 如果分词字段在停用词列表里，则取出
            continue
        seg_res.append(seg_item)  # 如果不在则加入分词结果中
        if seg_item in voc_dict.keys():  # 使用字典统计词频seg_item in voc_dict.keys():
            voc_dict[seg_item] += 1
        else:
            voc_dict[seg_item] = 1

#     print(content)  # 打印未分词前的句子
#     print(seg_res)

# 对字典进行排序，取TOPK词，如果将所有词都要，将会导致字典过大。我们只关注一些高频的词
voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq],
                  key=lambda x: x[1],  # key：指定一个参数的函数，该函数用于从每个列表元素中提取比较键
                  reverse=True)[:top_n]  # 取排完序后的前top_n个词,

voc_dict = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}  # 根据排序后的字典重新字典
voc_dict.update({UNK: len(voc_dict), PAD: len(voc_dict) + 1})  # 将前top_n后面的归类为UNK

print(voc_dict)  # '泪': 0, '嘻嘻': 1, '都': 2,

# 保存字典
ff = open("../sources/dict.txt", "w")
for item in voc_dict.keys():
    ff.writelines("{},{}\n".format(item, voc_dict[item]))  # '泪': 0, '嘻嘻': 1, '都': 2,