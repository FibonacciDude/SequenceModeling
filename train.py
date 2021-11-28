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

def add_noise(X):
    dat = X[..., :-1]
    X[..., :-1] += torch.randn(dat.shape).cuda() / 1e2 * .9
    #X = X
    return X

def loop(f_model, pred_model, optimizer, scheduler, data_tup, TOT, type_=0, update=True):
    predictor=config["predictor"] # predicting?
    loss_calc = loss_calc_y if predictor else loss_calc_x
    losses = []
    dataset, data_mask, data_lens, data_labels = data_tup
    model = pred_model if predictor else f_model

    for batch_idx in range(TOT):
        batch, mask, lens, labels  = get_batch(
            batch_idx, dataset, data_mask, data_lens, data_labels, type=type_)
        if update:
            batch = add_noise(batch)

        if predictor:
            batch=f_model(batch, lens).detach()
            Y_hat=pred_model(batch.view(batch.size()[0], -1))
            Y=labels
        else:
            Y_hat=f_model(batch, lens)
            Y=batch

        loss = loss_calc(Y, Y_hat, mask, config)
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

def train_model(f_model, pred_model, optimizer, scheduler, data_tup):
    TOT = math.ceil(data_index.TRAIN / config["batch_size"])
    #summary(pred_model if config["predictor"] else f_model, "pred" if config["predictor"]  else ("rnn" if config["rnn"] else "baseline"))
    for e in tqdm.trange(config["epochs"]):
        losses = loop(f_model, pred_model, optimizer, scheduler, data_tup, TOT=TOT, type_=0, update=True)
        #print("Epoch %d over, average loss" % e, losses.mean())
        #val_losses = validate(model, dataset, data_mask, data_lens, final=True)
        #print("\t\tVal loss: %.4f" % (val_losses.mean()))

def validate(f_model, pred_model, data_tup, final=False):
    TOT = (data_index.VAL // config["batch_size"])
    losses = loop(f_model, pred_model, None, None, data_tup, TOT=TOT, type_=1, update=False) # no optim or schedule
    if final:
        print("Final validation")
        print("\tMean validation loss:", losses.mean())
    return losses


# load dataset
print("loading dataset...")
dataset = torch.load("data/dataset.pt")
data_mask = torch.load("data/data_mask.pt")
# take out the mask for the None loss
data_mask[..., :-1] = 0
data_lens = torch.load("data/lens.pt")
data_labels = torch.load("data/labels.npy")
# tuple with everything
data_tup = (dataset, data_mask, data_lens, data_labels)

print("dataset loaded...")

def run():

    predictor=config["predictor"]
    mode = "pred" if predictor else "no_pred"
    f_model= Rnn(config, mode=mode) if config["rnn"] else Baseline(config, mode=mode)

    if predictor:
        pred_model=Predictor(config)
        f_model.load_state_dict(torch.load("best_models/" + config["checkpoint_dir_f"]))
        optimizer=optim.Adam(pred_model.parameters(), lr=config["lr"], weight_decay=config["l2_reg"])
    else:
        pred_model=None
        optimizer=optim.Adam(f_model.parameters(), lr=config["lr"], weight_decay=config["l2_reg"])
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    train_model(f_model, pred_model, optimizer, scheduler, data_tup)

    #if load and os.path.exists("checkpoints/" + config["checkpoint_dir"]):
    #    rnn.load("checkpoints/" + config["checkpoint_dir"])
    
    model = pred_model if predictor else f_model
    losses = validate(model, pred_model, data_tup, final=True)
    dir_type = "p" if predictor else "f"
    torch.save(model.state_dict(), "checkpoints/" + config["checkpoint_dir_"+dir_type])
    with open("crossval/"+config["checkpoint_dir_"+dir_type][:-2]+"txt", "w") as f:
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

    baseline_hidden_size = [32, 64] # baseline (or error) (this should be greater than hidden_size lstm)
    baseline_depth = [2, 4, 6] # baseline depth (or error), doesn't include bottleneck
    predictor_hidden_size = [32, 64] # predictor
    predictor_depth = [4, 6] # predictor
    error_hidden_size = [32, 64]
    error_depth = [4, 6]

    iter_dict = {
    #"num_layers" : num_layers, 
    #"inside_layers" : inside_layers,
    #"baseline_hidden_size" : baseline_hidden_size,
    #"baseline_depth" : baseline_depth,
    #"predictor_hidden_size" : predictor_hidden_size,
    #"predictor_depth" : predictor_depth,
    #"error_hidden_size" : error_hidden_size,
    #"error_depth" : error_depth,
    #"proj_hidden_size" : proj_hidden_size, # for lstm feedforward
    "l2_reg" : l2_reg,
    #"hidden_size" : hidden_size,
    "lr" : lr
    }

    iter_arr = list(iter_dict.values())
    config_file = "config.json"
    config_dict = json.load(open(config_file))
    rnn = True
    predictor = True
    epochs = 4
    config_dict["rnn"] = rnn
    config_dict["predictor"] = predictor
    config_dict["epochs"] = epochs 
    config_dict["print_every"] = 10_000

    checkpoint_append = info.name + "_" + (("predictor") if predictor else ("rnn" if rnn else "baseline"))
    dir_type="p" if predictor else "f"
    x_size = info.feature_size * config_dict["second_split"]
    
    for idx, params in enumerate(itertools.product(*iter_arr)):
        if idx > 0:
            break
        print("Currently:")
        for i, (k, v) in enumerate(iter_dict.items()):
            config_dict[k] = params[i]
            print("{}: {}".format(k, params[i]))

        config_dict["checkpoint_dir_"+dir_type] = checkpoint_append + "_%d.pt" % idx
        #with open("config_val_test.json", 'w', encoding='utf-8') as f: # dump when f
        #    json.dump(config_dict, f, ensure_ascii=False, indent=4)
        config_dict = json.load(open(config_file)) # read when pred

        config_dict["rnn"] = rnn
        config_dict["predictor"] = predictor
        config_dict["epochs"] = epochs 
        config_dict["print_every"] = 10_000

        config = config_dict
        print("dir:", config["checkpoint_dir_f"])
        torch.manual_seed(config['seed'])
        run()  # run
        # save and get best
        #os.system("cp checkpoints/gazebase_rnn_0.pt best_models/gazebase_test.pt")
