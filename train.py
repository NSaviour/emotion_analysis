import torch
import torch.nn as nn
from torch import optim

from configs import Config
from datasets import data_loader, text_cls
from models import Model

cfg = Config()

voc_dict_path = "sources/dict_rec"
data_path = "sources/weibo_senti_100k.csv"
data_stop_path = "sources/hit_stopword"
dataset = text_cls(data_path, data_stop_path, voc_dict_path)

train_dataloader = data_loader(dataset, cfg)

cfg.pad_size = dataset.max_len_seq

model_text_cls = Model(cfg)
model_text_cls.to(device=cfg.devices)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learn_rate)

for epoch in range(cfg.num_of_epoche):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        data = torch.tensor(data)
        label = torch.tensor(label)
        optimizer.zero_grad()
        pred = model_text_cls.forward(data)
        loss_val = loss_func(pred, label)

        # print(pred)
        # print(label)
        print("epoch is {},ite is {}, val is {}".format(epoch, i, loss_val))
        loss_val.backward()
        optimizer.step()

    if epoch % 10 == 0:
        torch.save(model_text_cls.state_dict(), "models/{}.pth".format(epoch))
