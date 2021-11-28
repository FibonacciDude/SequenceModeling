import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import data_index

info = data_index.info

def loss_calc_x(X, Y_hat, mask, config):
    batch_size, seq_len, ss, fs = X.size()
    X = X.reshape(batch_size, seq_len, ss*fs)
    Y_hat = Y_hat[:, :-1, :]  # 0, 1, ..., last-1
    Y = X[:, 1:, :]  # 1, 2, ..., last
    #X = X.reshape(batch_size, seq_len, ss, fs)
    Y = Y.reshape(batch_size, seq_len-1, ss, fs)
    Y_hat = Y_hat.reshape(batch_size, seq_len-1, ss, fs)
    mask = mask[:, :-1, :, :]
    loss = (Y - Y_hat) ** 2  # mse loss
    loss = loss * mask  # mask it
    loss = loss.sum()  / mask.sum() * (ss * fs) * config["loss_factor"]
    return loss

def loss_calc_y(Y, Y_hat, mask, config):
    #Y_hat = Y_hat[:, :-1, :]  # 0, 1, ..., last-1
    #Y = Y[:, 1:, :]  # 1, 2, ..., last
    print(Y.shape, Y_hat.shape)
    crit = nn.CrossEntropyLoss(weight=None)
    loss = crit(Y, Y_hat)
    return loss

class FeedForward(nn.Module):
    def __init__(self, h, d, in_, out_, device="cuda"):
        super(FeedForward, self).__init__()
        layers = [nn.Linear(in_, h), nn.ReLU()] + [nn.Linear(h, h), nn.ReLU()] * d + [nn.Linear(h, out_)]
        self.net = nn.Sequential(*layers)
        self.to(torch.device(device))

    def forward(self, X):
        out = self.net(X)
        return out

class Predictor(nn.Module):
    def __init__(self, config, device="cuda"):
        super(Predictor, self).__init__()
        self.net = FeedForward(config["predictor_hidden_size"], config["predictor_depth"], config["hidden_size"], config["label_size"])
        self.to(torch.device(device))
    def forward(self, X):
        return self.net(X)

class Baseline(nn.Module):
    def __init__(self, config, mode="no_pred", device="cuda"):
        super(Baseline, self).__init__()
        bhs = config["baseline_hidden_size"]
        self.x_size=info.feature_size*config["second_split"]
        self.l1 = FeedForward(bhs, config["baseline_depth"], self.x_size, bhs)
        btl = [nn.ReLU(), nn.Linear(bhs, config["hidden_size"])]
        self.bottleneck = nn.Sequential(*btl)
        self.l2 = nn.Sequential(nn.Linear(config["hidden_size"], self.x_size))
        self.to(torch.device(device))
        self.mode=mode

    def forward(self, X, lens):
        batch_size, seq_len, ss, fs = X.size()
        X = X.reshape(batch_size*seq_len, ss*fs)
        out = self.l1(X)
        out = self.bottleneck(out)
        if self.mode == "pred":
            return out.detach()
        out = self.l2(out)
        out = out.reshape(batch_size, seq_len, ss, fs)
        return out

class Error(nn.Module):
    def __init__(self, config, device="cuda"):
        super(Baseline, self).__init__()
        self.net = FeedForward(config["error_hidden_size"], config["error_depth"], x_size, info.feature_size)
        self.to(torch.device(device))
    def forward(self, X, lens):
        batch_size, seq_len, ss, fs = X.size()
        X = X.reshape(batch_size*seq_len, ss*fs)
        out = self.net(X)
        out = out.reshape(batch_size, seq_len, ss, fs)
        return out

class Rnn(nn.Module):
    # add self-correcting module later as another nn.Module
    def __init__(self, config, mode="no_pred", device="cuda"):
        super(Rnn, self).__init__()
        self.x_size = info.feature_size * config["second_split"]
        # proj size = hidden size
        self.rnn = nn.LSTM(self.x_size, hidden_size=config["hidden_size"],
                           num_layers=config["num_layers"], dropout=config["dropout"], batch_first=True)
        self.proj_net = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config["hidden_size"], config["proj_hidden_size"]),
            nn.ReLU(),
            nn.Linear(config["proj_hidden_size"],
                      info.feature_size * config["second_split"])
        )
        self.to(torch.device(device))
        self.mode=mode

    def forward(self, X, lens):
        # this forward goes through the entire length of the input and spits out all the predictions as output
        # input is NOT a padded sequence
        batch_size, seq_len, ss, fs = X.size()
        X = X.view(batch_size, seq_len, ss*fs)
        packed_X = rnn_utils.pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        out, self.hidden = self.rnn(packed_X)
        #unpack
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        if self.mode == "pred":
            return out[:, :-1, :].detach() # reshape after this
        #project
        out = self.proj_net(out)
        out = out.contiguous()
        out = out.view((batch_size, seq_len, -1))
        # last prediction (can't find it's loss) is still there
        return out

    def loss(self, X, Y_hat, mask):
        batch_size, seq_len, ss, fs = X.size()
        X = X.view(batch_size, seq_len, ss*fs)
        Y_hat = Y_hat[:, :-1, :]  # 0, 1, ..., last-1
        Y = X[:, 1:, :]  # 1, 2, ..., last
        #X = X.view(batch_size, seq_len, ss, fs)
        Y = Y.view(batch_size, seq_len-1, ss, fs)
        Y_hat = Y_hat.view(batch_size, seq_len-1, ss, fs)
        mask = mask[:, :-1, :, :]
        loss = (Y - Y_hat) ** 2  # mse loss
        loss = loss * mask  # mask it
        loss = loss.sum()  / mask.sum() * (ss * fs) * config["loss_factor"]
        return loss
