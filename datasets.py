import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader

voc_dict_path = "sources/dict_rec"
data_path = "sources/weibo_senti_100k.csv"
data_stop_path = "sources/hit_stopword"


def read_dict(voc_dict_path):
    voc_dict = {}
    with open(voc_dict_path, encoding="utf-8") as f:
        dict_list = f.readlines()
    for item in dict_list:
        key, id = item.split(",")
        voc_dict[key] = int(id.strip())
    return voc_dict


def load_data(data_path, data_stop_path):
    with open(data_path, encoding="utf-8") as f:
        data_list = f.readlines()[1:]

    with open(data_stop_path, encoding="utf-8") as f2:
        stops_word = [line.strip() for line in f2.readlines()]
    stops_word.append(" ")
    stops_word.append("\n")

    data = list()
    max_len = 0
    for item in data_list:
        label = item[0]
        content = item[2:].strip()
        seg_list = jieba.cut(content, cut_all=False)

        seg_res = list()
        for seg_item in seg_list:
            if seg_item in stops_word:
                continue
            seg_res.append(seg_item)

        data.append([label, seg_res])
        if len(seg_res) > max_len:
            max_len = len(seg_res)
    return data, max_len


class text_cls(Dataset):
    def __init__(self, data_path, data_stop_path, voc_dict_path):
        self.data_path = data_path
        self.data_stop_path = data_stop_path

        self.voc_dict = read_dict(voc_dict_path)
        self.data, self.max_len_seq = load_data(self.data_path, self.data_stop_path)

        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]
        input_idx = []
        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict["<UNK>"])
        if len(input_idx) < self.max_len_seq:
            input_idx += [self.voc_dict["<PAD>"] for _ in range(self.max_len_seq - len(input_idx))]

        data = np.array(input_idx)
        return label, data


def data_loader(dataset, config):
    # dataset = text_cls(data_path, data_stop_path, voc_dict_path)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_suffle)


if __name__ == '__main__':
    train_dataloader = data_loader()

    for i, batch in enumerate(train_dataloader):
        print(batch)
