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
    X[..., :-1] += torch.randn(dat.shape).cuda() / 1e2 * .5
    return X

def loop(f_model, pred_model, optimizer, scheduler, data_tup, TOT, type_=0, update=True):
    predictor=config["predictor"]
    loss_calc = loss_calc_y if predictor else loss_calc_x
    losses, accs = [], []
    dataset, data_mask, data_lens, data_labels = data_tup
    model = pred_model if predictor else f_model
    if predictor:
        norm=torch.load("best_models/"+("rnn" if config["rnn"] else "baseline") + "_norm.pt")
        mean,std=norm
        mean=mean[None, :]
        std=std[None, :]

    for batch_idx in range(TOT):
        batch, mask, lens, labels  = get_batch(
            batch_idx, dataset, data_mask, data_lens, data_labels, type=type_)
        #if update and not predictor:
            #batch = add_noise(batch)

        if predictor:
            batch=f_model(batch, lens).detach()
            #normalize
            #print(batch[0, :-5])
            batch=(batch-mean)/std
            #print(batch[0, :-5])
            Y_hat=pred_model(batch) #batch.reshape(batch.size()[0], -1))
            Y=labels
        else:
            Y_hat=f_model(batch, lens)
            Y=batch

        loss, acc = loss_calc(Y, Y_hat, mask, config)
        losses.append(loss.item() * Y_hat.size()[0])
        accs.append(acc * Y_hat.size()[0])

        if batch_idx % config["print_every"] == 0 and type_==0 and (config["print_every"] <= TOT):
            # output the loss and stuff, all pretty
            print("\t\ttrain loss (%d): %.4f" % (batch_idx, loss.item()))

        if update:
            # grad step
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_norm"])
            optimizer.step()
   
    if update:
        scheduler.step()

    return np.array(losses), np.array(accs)

def train_model(f_model, pred_model, optimizer, scheduler, data_tup):
    TOT = math.ceil(data_index.TRAIN / config["batch_size"])
    for e in tqdm.trange(config["epochs"]):
        losses, accs = loop(f_model, pred_model, optimizer, scheduler, data_tup, TOT=TOT, type_=0, update=True)

def validate(f_model, pred_model, data_tup, final=False):
    TOT = (data_index.VAL // config["batch_size"])
    losses, accs = loop(f_model, pred_model, None, None, data_tup, TOT=TOT, type_=1, update=False) # no optim or schedule
    if final:
        print("Final validation")
        print("\tMean validation loss:", losses.sum() / data_index.VAL)
        if config["predictor"]:
            print("\tMean acc:", accs.sum() / data_index.VAL)
    return losses.sum() / data_index.VAL


# load dataset
print("loading dataset...")
dataset = torch.load("data/dataset.pt")
# scale input, too small
data_factor = 2 # not in config!
dataset *= data_factor # this is arbitrary
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
    loss = validate(f_model, pred_model, data_tup, final=True)
    dir_type = "p" if predictor else "f"
    torch.save(model.state_dict(), "checkpoints/" + config["checkpoint_dir_"+dir_type])
    with open("crossval/"+config["checkpoint_dir_"+dir_type][:-2]+"txt", "w") as f:
        f.write(str(loss))
    return loss


if __name__ == "__main__":

    # cross_validation
    # universal ---
    lr = np.logspace(start=-5, stop=-1, num=7)
    l2_reg = np.logspace(start=-4, stop=-2, num=7)

    # changes with conf
    hidden_size = [32, 64, 128] # lstm
    proj_hidden_size = [32, 64] # feedforward
    inside_layers = [0, 1] # for feedforward in lstm
    num_layers = [1, 2] # for lstm
    baseline_hidden_size = [32, 64] # baseline (or error) (this should be greater than hidden_size lstm)
    baseline_depth = [2, 4, 6] # baseline depth (or error), doesn't include bottleneck
    predictor_hidden_size = [32, 64] # predictor
    predictor_depth = [4, 6] # predictor
    error_hidden_size = [32, 64] # error
    error_depth = [4, 6] # error

    iter_dict = { # or load from somewhere else
    #"num_layers" : num_layers, 
    #"inside_layers" : inside_layers,
    #"baseline_hidden_size" : baseline_hidden_size,
    #"baseline_depth" : baseline_depth,
    #"predictor_hidden_size" : predictor_hidden_size,
    #"predictor_depth" : predictor_depth,
    #"error_hidden_size" : error_hidden_size,
    #"error_depth" : error_depth,
    #"proj_hidden_size" : proj_hidden_size, # for lstm feedforward
    }
    iter_dict = iter_dict.update({
    "l2_reg" : l2_reg,
    "lr" : lr
    }
    iter_dict = iter_dict.update("proj_hidden_size" : iter_dict["hidden_size"]) # if lstm

    iter_arr = list(iter_dict.values())

    config_dict = json.load(open("config.json"))
    torch.manual_seed(config_dict['seed'])
    rnn=config_dict["rnn"]
    predictor=config_dict["predictor"]
    epochs=config_dict["epochs"]
    every=config_dict["print_every"]

    checkpoint_append = info.name + "_" + (("predictor") if predictor else ("rnn" if rnn else "baseline"))
    dir_type="p" if predictor else "f"
    x_size = info.feature_size * config_dict["second_split"]

    def update(keys, params, config_dict, output=False):
        for i, k in enumerate(keys):
            config_dict[k] = params[i]
            if output:
                print("{}: {}".format(k, params[i]))
        return config_dict
    
    val_accs = {}
    f_model_type=("rnn" if rnn else "baseline")

    if predictor:
        best_idx=open("best_models/best_"+f_model_type+"_idx.txt").read()
        best_params=np.load("checkpoints/"+info.name+"_"+f_model_type+"_%s_params.npy"% best_idx)
    
    for idx, params in enumerate(itertools.product(*iter_arr)):
        print("Currently:")
        config_dict=update(list(iter_dict.keys()), params, config_dict, output=True)
        config_dict["checkpoint_dir_"+dir_type] = checkpoint_append + "_%d.pt" % idx
        # add params to checkpoint
        if not predictor:
            np.save("checkpoints/"+checkpoint_append+"_%d_params.npy" % idx, np.array(params))
        else:
            config_dict=update(list(iter_dict.keys()), best_params, config_dict)
        config = config_dict
        print("dir:", config["checkpoint_dir_f"])

        loss=run()  # run
        val_accs[idx]=loss

    if not predictor:
        # save and get best
        keys=np.array(list(val_accs.keys()))
        vals=np.array(list(val_accs.values()))
        best_idx=keys[vals.argmin()]
        best_dir= "checkpoints/%s_%d" % (checkpoint_append, best_idx)
        # load std, mean of z vector to normalize (for predictor)
        print("Computing and saving norm...")
        f_model= Rnn(config, mode="pred") if rnn else Baseline(config, mode="pred")
        zs=f_model(dataset[data_index.TEST+data_index.VAL:], data_lens[data_index.TEST+data_index.VAL:].cpu().numpy())
        std, mean = torch.std_mean(zs, dim=0)
        norm=torch.stack((mean, std))
        torch.save(norm, "best_models/"+f_model_type+"_norm.pt")

        with open("best_models/best_%s_idx.txt"%f_model_type, "w") as f:
            f.write(str(idx))
        print("Norm saved and computed")
        os.system("cp %s.pt best_models/gazebase_test.pt" % (best_dir))
