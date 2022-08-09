import copy

import jieba

data_path = "sources/weibo_senti_100k.csv"
data_stop_path = "sources/hit_stopword"

with open(data_path, encoding="utf-8") as f:
    data_list = f.readlines()[1:]

with open(data_stop_path, encoding="utf-8") as f2:
    stops_word = [line.strip() for line in f2.readlines()]
stops_word.append(" ")
stops_word.append("\n")

min_seq = 1
top_N = 1000
voc_dict = dict()
UNK = "<UNK>"
PAD = "<PAD>"
for item in data_list:
    label = item[0]
    content = item[2:].strip()
    seg_list = jieba.cut(content, cut_all=False)

    seg_res = list()
    for seg_item in seg_list:
        if seg_item in stops_word:
            continue
        seg_res.append(seg_item)

        # 统计词频
        if seg_item in voc_dict.keys():
            voc_dict[seg_item] = voc_dict[seg_item] + 1
        else:
            voc_dict[seg_item] = 1

voc_dict_bk = copy.deepcopy(voc_dict)

voc_list = sorted([_ for _ in voc_dict.items() if _[1] > min_seq], key=lambda x: x[1], reverse=True)[:top_N]
voc_dict = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}
voc_dict.update({UNK: len(voc_dict), PAD: len(voc_dict) + 1})

# 保存数据
with open("sources/dict_rec", "w", encoding="utf-8") as f3:
    for key, value in voc_dict.items():
        f3.writelines("{},{}\n".format(key, value))
pass
