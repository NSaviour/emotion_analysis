class Config():
    def __init__(self):
        self.n_vocab = 1002
        self.embed_size = 128
        self.hidden_size = 128
        self.num_layers = 3
        self.dropout = 0.8
        self.num_classes = 2
        self.pad_size = 32
        self.devices = "cpu"
        self.batch_size = 128
        self.is_suffle = True
        self.learn_rate = 0.0001
        self.num_of_epoche=100
