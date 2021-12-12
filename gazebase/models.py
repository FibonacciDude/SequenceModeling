import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import data_index

info = data_index.info

def loss_calc_x(X, Y_hat, mask, config):
    batch_size, seq_len, ss, fs = X.size()
    X = X.reshape(batch_size, seq_len, ss*fs)
    Y_hat = Y_hat[:, 1:-1, :]  #  1, ..., last-1
    Y_prev = X[:, :-2, :] # 0, 1, ..., last-2
    Y = X[:, 2:, :]  # 2, ..., last
    Y = Y.reshape(batch_size, seq_len-2, ss, fs)
    Y_hat = Y_hat.reshape(batch_size, seq_len-2, ss, fs)
    Y_prev = Y_prev.reshape(batch_size, seq_len-2, ss, fs)
    # prediction of where it's going to ... move! (add vector to previous)
    mask = mask[:, 1:-1, :, :] # none of first or last
    loss = ( (Y + Y_prev) - Y_hat) ** 2  # mse loss
    loss = loss * mask  # mask it
    loss = loss.sum() / mask.sum() * (ss * fs) * config["loss_factor"]
    return loss, 0  # acc

def loss_calc_y(Y, Y_hat, mask, config):
    N = 9 # don't use first 3 seconds
    Y_hat = Y_hat.reshape(batch_size, seq_len, ss*fs)[:, N:, :]
    # average the raw scores
    Y_hat = Y_hat.mean(1)
    logits=nn.LogSoftmax(dim=1)(Y_hat)
    Y = Y.squeeze(0)
    loss = nn.NLLLoss()(logits, Y)
    acc=(logits.detach().argmax(1)==Y).detach().float().mean().item()
    return loss, acc

class FeedForward(nn.Module):
    def __init__(self, h, d, in_, out_, device="cuda"):
        super(FeedForward, self).__init__()
        layers = [nn.Linear(in_, h), nn.LeakyReLU()] + [nn.Linear(h, h), nn.LeakyReLU()] * d + [nn.Linear(h, out_)]
        self.l1 = nn.Sequential(*layers)
        self.to(torch.device(device))

    def forward(self, X):
        out = self.l1(X)
        return out

class Predictor(nn.Module):
    def __init__(self, config, device="cuda"):
        super(Predictor, self).__init__()
        self.net = FeedForward(h=int(config["predictor_hidden_size"]), d=int(config["predictor_depth"]), in_=int(config["hidden_size"]), out_=int(config["label_size"]))
        self.to(torch.device(device))

    def forward(self, X):
        out=self.net(X)
        return out

class Baseline(nn.Module):
    def __init__(self, config, mode="no_pred", device="cuda"):
        super(Baseline, self).__init__()
        bhs = int(config["baseline_hidden_size"])
        self.x_size=(info.feature_size-1)*config["second_split"] # no flag
        self.l1 = FeedForward(bhs, int(config["baseline_depth"]), self.x_size, bhs)
        btl = [nn.ReLU(), nn.Linear(bhs, int(config["hidden_size"]))]
        self.bottleneck = nn.Sequential(*btl)
        self.l2 = nn.Sequential(nn.Linear(int(config["hidden_size"]), self.x_size))
        self.to(torch.device(device))
        self.mode=mode

    def forward(self, X, lens):
        batch_size, seq_len, ss, fs = X.size()
        X = X.reshape(batch_size*seq_len, ss*fs)
        out = self.l1(X)
        out = self.bottleneck(out)
        if self.mode == "pred":
            return out.reshape(batch_size, seq_len, -1).mean(1).detach() # no flag
        out = self.l2(out)
        out = out.reshape(batch_size, seq_len, ss, fs)
        return out

class Error(nn.Module):
    def __init__(self, config, device="cuda"):
        super(Baseline, self).__init__()
        self.net = FeedForward(int(config["error_hidden_size"]), int(config["error_depth"]), x_size, info.feature_size-1) # no flag
        self.to(torch.device(device))
    def forward(self, X, lens):
        batch_size, seq_len, ss, fs = X.size()
        X = X.reshape(batch_size*seq_len, ss*fs)
        out = self.net(X)
        out = out.reshape(batch_size, seq_len, ss, fs)
        return out

class RNN(nn.Module):
    # add self-correcting module later as another nn.Module
    def __init__(self, config, mode="no_pred", model="gru", device="cuda"):
        super(RNN, self).__init__()
        self.config = config
        #self.x_size = info.feature_size * int(config["second_split"])
        # no flag
        self.x_size = (info.feature_size-1) * int(config["second_split"])
        # proj size = hidden size
        model_ = nn.GRU if model=="gru" else nn.LSTM
        self.rnn = model_(self.x_size, hidden_size=int(config["hidden_size"]),
                           num_layers=int(config["num_layers"]), dropout=config["dropout"], batch_first=True)
        hidden=int(config["proj_hidden_size"])
        d=int(config["inside_layers"])
        layers=[nn.ReLU(), nn.Linear(int(config["hidden_size"]), hidden), nn.ReLU()] + [nn.Linear(hidden, hidden)]*d + [nn.Linear(hidden, self.x_size)]
        self.proj_net = nn.Sequential(*layers)
        self.to(torch.device(device))
        self.mode=mode
        self.model=model

    def forward(self, X, lens, batch=True):
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
            #out=out.mean(1)  # give the mean vec
            #h=self.hidden if self.model == "gru" else self.hidden[1] # give hidden c, not h (last)

            #nl, bs, dim = h.shape
            #out=h[-1]
            #out=h.contiguous().permute((1, 0, 2)).reshape(bs, nl*dim)
            #out=out.reshape((batch_size, nl*int(self.config["hidden_size"])))
            #out=out.reshape((batch_size, int(self.config["hidden_size"])))
            if batch:
                return out.mean(1)
            else:
                out=out.reshape((batch_size, seq_len*ss*fs)) # give it all the individual vectors
            return out.detach() # reshape after this
        out = self.proj_net(out)
        out = out.reshape((batch_size, seq_len, -1))
        return out
