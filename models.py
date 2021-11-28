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
    Y = Y.reshape(batch_size, seq_len-1, ss, fs)
    Y_hat = Y_hat.reshape(batch_size, seq_len-1, ss, fs)
    mask = mask[:, :-1, :, :]
    loss = (Y - Y_hat) ** 2  # mse loss
    loss = loss * mask  # mask it
    loss = loss.sum()  / mask.sum() * (ss * fs) * config["loss_factor"]
    return loss, 0  # acc

def loss_calc_y(Y, Y_hat, mask, config):
    crit = nn.CrossEntropyLoss(weight=None)
    softmax=nn.Softmax(dim=1)(Y_hat)
    acc=(softmax.argmax(1)==Y.squeeze(0)).float().mean().item()
    loss = crit(Y_hat, Y)
    return loss, acc

class FeedForward(nn.Module):
    def __init__(self, h, d, in_, out_, device="cuda"):
        super(FeedForward, self).__init__()
        layers = [nn.Linear(in_, h), nn.ReLU()] + [nn.Linear(h, h), nn.ReLU()] * d + [nn.Linear(h, out_)]
        self.l1 = nn.Sequential(*layers)
        self.to(torch.device(device))

    def forward(self, X):
        out = self.l1(X)
        #print(self.l1(torch.rand(X.shape).cuda()))
        return out

class Predictor(nn.Module):
    def __init__(self, config, device="cuda"):
        super(Predictor, self).__init__()
        self.net = FeedForward(config["predictor_hidden_size"], config["predictor_depth"], config["hidden_size"], config["label_size"])
        self.to(torch.device(device))
    def forward(self, X):
        #X/=10
        #print(self.net(X), "output")
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
            return out.reshape(batch_size, seq_len, -1)[:, :-1, :].mean(1).detach()
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
        self.config = config
        self.x_size = info.feature_size * config["second_split"]
        # proj size = hidden size
        self.rnn = nn.LSTM(self.x_size, hidden_size=config["hidden_size"],
                           num_layers=config["num_layers"], dropout=config["dropout"], batch_first=True)
        hidden=config["proj_hidden_size"]
        d=config["inside_layers"]
        layers=[nn.ReLU(), nn.Linear(config["hidden_size"], hidden), nn.ReLU()] + [nn.Linear(hidden, hidden)]*d + [nn.Linear(hidden, info.feature_size * config["second_split"])]
        self.proj_net = nn.Sequential(*layers)
        self.to(torch.device(device))
        self.mode=mode

    def forward(self, X, lens):
        # this forward goes through the entire length of the input and spits out all the predictions as output
        # input is NOT a padded sequence
        batch_size, seq_len, ss, fs = X.size()
        X = X.reshape(batch_size, seq_len, ss*fs)
        packed_X = rnn_utils.pack_padded_sequence(X, lens, batch_first=True, enforce_sorted=False)
        out, self.hidden = self.rnn(packed_X)
        #unpack
        out, _ = rnn_utils.pad_packed_sequence(out, batch_first=True)
        if self.mode == "pred":
            #out=out[:, -1, :] # give only the last z vector
            out=out.mean(1)  # give the mean vec
            out=out.reshape(batch_size, ss*self.config["hidden_size"])
            return out.detach() # reshape after this
        #project
        out = self.proj_net(out)
        #out = out.contiguous()
        out = out.reshape((batch_size, seq_len, -1))
        # last prediction (can't find its loss) is still there
        return out
