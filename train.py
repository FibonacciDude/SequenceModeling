import numpy as np
import csv
from collections import namedtuple
import json
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from data_index import get_dataset, get_batch
import data_index
import math
import os
import itertools
import tqdm 
from models import *

info = data_index.info
x_size = info.feature_size*config["second_split"]

def add_noise(X):
    dat = X[..., :-1]
    X[..., :-1] += torch.randn(dat.shape).cuda() / 1e2 * .9
    #X = X
    return X

def loop(model, optimizer, scheduler, dataset, data_mask, data_lens, TOT, type_=0, update=True):
    losses = []
    for batch_idx in range(TOT):
        batch, mask, lens = get_batch(
            batch_idx, dataset, data_mask, data_lens, type=type_, last=False)
        if update:
            batch = add_noise(batch)
        Y_hat = model(batch, lens)
        loss = loss_calc(batch, Y_hat, mask, config)

        losses.append(loss.item() * Y_hat.size()[0] / config["batch_size"])
        if batch_idx % config["print_every"] == 0 and type_==0 and (config["print_every"] <= TOT):
            # output the loss and stuff, all pretty
            print("\t\ttrain loss (%d): %.4f" % (batch_idx, loss.item()))

        if update:
            # grad step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_norm"])
   
    if update:
        scheduler.step()
    return np.array(losses)

def train_sequence_model(model, optimizer, scheduler, dataset, data_mask, data_lens, epochs):

    TOT = math.ceil(data_index.TRAIN / config["batch_size"])
    #summary(model, (config["batch_size"], 100, info.feature_size * config["second_split"]))
    summary(model, "rnn" if config["rnn"] else "baseline")
    #plot_train = []
    #plot_val = []
    for e in tqdm.trange(epochs):
        # loop through batches
        #print("Epoch %d" % e)
        losses = loop(model, optimizer, scheduler, dataset, data_mask, data_lens, TOT=TOT, type_=0, update=True)

        #print("Epoch %d over, average loss" % e, losses.mean())
        #val_losses = validate(model, dataset, data_mask, data_lens, final=True)
        #print("\t\tVal loss: %.4f" % (val_losses.mean()))

        #plot_val.append(val_losses.mean()) 
        #plot_train.append(losses.mean()) 
    #return plot_train, plot_val

def validate(model, dataset, data_mask, data_lens, final=False):
    TOT = (data_index.VAL // config["batch_size"])
    losses = loop(model, None, None, dataset, data_mask, data_lens, TOT=TOT, type_=1, update=False)

    if final:
        print("Final validation")
        print("\tMean validation loss:", losses.mean())
    return losses


# load dataset
print("loading dataset...")
dataset = torch.load("data/dataset.pt")
data_mask = torch.load("data/data_mask.pt")
# take out the mask for the None loss
#data_mask[data_mask==info.feature_size] = 0
data_mask[..., :-1] = 0

lens = torch.load("data/lens.pt")
print("dataset loaded...")

def main():
    epochs = config["epochs"]

    model = Rnn(config) if config["rnn"] else Baseline(config)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["l2_reg"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    #if load and os.path.exists("checkpoints/" + config["checkpoint_dir"]):
    #    rnn.load("checkpoints/" + config["checkpoint_dir"])
    if config["predictor"]:
    train_sequence_model(model, optimizer, scheduler, dataset, data_mask, lens, epochs)
    losses = validate(model, dataset, data_mask, lens, final=True)
    torch.save(model.state_dict(), "checkpoints/" + config["checkpoint_dir"])
    with open("crossval/"+config["checkpoint_dir"][:-2]+"txt", "w") as f:
        f.write(str(losses.mean()))


if __name__ == "__main__":

    info = data_index.info

    # cross_validation
    # universal
    lr = np.logspace(start=-5, stop=-1, num=7)
    l2_reg = np.logspace(start=-4, stop=-2, num=2)

    # lstm
    hidden_size = [32, 64, 128] # lstm
    proj_hidden_size = [32, 64] # feedforward
    inside_layers = [0, 1] # for feedforward in lstm
    num_layers = [1, 2] # for lstm

    # baseline / error
    #baseline_or_error = True
    rnn = False
    predictor = False
    baseline_hidden_size = [32, 64] # baseline (or error)
    baseline_depth = [2, 4, 6] # baseline depth (or errork)

    iter_dict = {
    #"num_layers" : num_layers, 
    #"inside_layers" : inside_layers,
    "baseline_hidden_size" : baseline_hidden_size,
    #"proj_hidden_size" : proj_hidden_size,
    "baseline_depth" : baseline_depth,
    "l2_reg" : l2_reg,
    #"hidden_size" : hidden_size,
    "lr" : lr
    }

    #iter_arr = [num_layers, inside_layers, error_hidden_size, proj_hidden_size, l2_reg, hidden_size, lr]
    iter_arr = list(iter_dict.values())
    config_dict = json.load(open("config.json"))
    config_dict["rnn"] = rnn
    config_dict["predictor"] = predictor
    #config = json.load(open("config_val_" + "rnn" if rnn else "baseline"+ ".json"))
    #main()
    x_size = info.feature_size * config_dict["second_split"]
    
    #"""
    #config = json.load(open("config_net.json"))
    for idx, params in enumerate(itertools.product(*iter_arr)):
        print("Currently:")
        #print(list(iter_dict.keys()))
        for i, (k, v) in enumerate(iter_dict.items()):
            #print(params)
            config_dict[k] = params[i]
            print("{}: {}".format(k, params[i]))
        config_dict["checkpoint_dir"] = "gazebase_" + "rnn" if rnn else "baseline" + "_%d.pt" % idx
        config = config_dict 
        torch.manual_seed(config['seed'])
        main() 
   #"""
