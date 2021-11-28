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
            # is norm wrong?
            batch=(batch-mean)/std
            Y_hat=pred_model(batch)
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

def run(save):

    predictor=config["predictor"]
    mode = "pred" if predictor else "no_pred"
    f_model= Rnn(config, mode=mode) if config["rnn"] else Baseline(config, mode=mode)

    if predictor:
        pred_model=Predictor(config)
        f_model.load_state_dict(torch.load("best_models/" + "%s_best_%s.pt" % (info.name, "rnn" if config["rnn"] else "baseline")))
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
    torch.save(model.state_dict(), "checkpoints/"+save+".pt")
    # config["checkpoint_dir_"+dir_type])
    with open("crossval/"+save+".txt", "w") as f:
        f.write(str(loss))
    return loss

def update(keys, params, config_dict, output=False):
    for i, k in enumerate(keys):
        config_dict[k] = params[i]
        if output:
            print("{}: {}".format(k, params[i]))
    return config_dict


def crossval(predictor, rnn, iter_dict):
    global config_dict, config
    f_model_type=["baseline", "rnn"][rnn]
    model_type=("pred" if predictor else ("rnn" if rnn else "baseline"))

    checkpoint_append = info.name + "_" + model_type
    dir_type="p" if predictor else "f"
    
    val_accs = {}
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
        save="%s_%s_%d" % (info.name, model_type, idx)
        loss=run(save)  # run
        val_accs[idx]=loss

    # save and get best
    keys=np.array(list(val_accs.keys()))
    vals=np.array(list(val_accs.values()))
    best_idx=keys[vals.argmin()]
    best_dir="%s_%d" % (checkpoint_append, best_idx)
    json.dump(val_accs, open("crossval/%s_v%d" % (f_model_type, predictor), "w"))

    # load std, mean of z vector to normalize (for predictor)
    if not predictor:
        print("Computing and saving norm...")
        f_model= Rnn(config, mode="pred") if rnn else Baseline(config, mode="pred")
        f_model.load_state_dict(torch.load("best_models/%s.pt" % best_dir))
        zs=f_model(dataset[data_index.TEST+data_index.VAL:], data_lens[data_index.TEST+data_index.VAL:].cpu().numpy())
        std, mean = torch.std_mean(zs, dim=0)
        norm=torch.stack((mean, std))
        torch.save(norm, "best_models/"+model_type+"_norm.pt")
        print("Norm saved and computed")
        os.system("cp checkpoints/%s.pt best_models/%s_best_%s.pt" % (best_dir, info.name, model_type))
        with open("best_models/best_%s_idx.txt"%model_type, "w") as f:
            f.write(str(idx))
    else:
        os.system("cp checkpoints/%s.pt best_models/%s_%s_pred_best.pt" % (best_dir, info.name, f_model_type))
        with open("best_models/best_%s_pred_idx.txt"%model_type, "w") as f:
            f.write(str(idx))

    return val_accs

if __name__ == "__main__":

    config_dict = json.load(open("config.json"))
    torch.manual_seed(config_dict['seed'])
    config=config_dict
    x_size = info.feature_size * config_dict["second_split"]

    # cross_validation
    # universal ---
    lr = np.logspace(start=-5, stop=-1, num=7)
    l2_reg = np.logspace(start=-4, stop=-2, num=2)
    cross_type=["baseline", "rnn", "pred"]
    val_accs_comb=[]
    config = {}

    for predictor in [False, True]:
        for rnn in [True, False]:
            config_dict["rnn"]=rnn
            config_dict["predictor"]=predictor
            epochs=config_dict["epochs"]
            every=config_dict["print_every"]
            type_ = cross_type[rnn] if not predictor else cross_type[-1]
            print("predictor: %s, rnn: %s, %s" % (predictor, rnn, type_))
            #iter_dict = json.load(open("crossval_config_"+type_+".json"))
            iter_dict={}
            iter_dict.update({
            "l2_reg" : l2_reg,
            "lr" : lr
            })
            iter_arr = list(iter_dict.values())
            val_accs=crossval(predictor, rnn, iter_dict)
            val_accs_comb.append(val_accs)
         #print("Finalized %s... Got loss:.4f%, acc:.4f%" % (loss, acc))

    #for model_type in cross_type:
        #best=int(open("best_models/best_%s_idx.txt"%model_type, "w").read())
        #print(best)
        

    # changes with conf
    #{# or load from somewhere else
    #"num_layers" : num_layers, 
    #"inside_layers" : inside_layers,
    #"baseline_hidden_size" : baseline_hidden_size,
    #"baseline_depth" : baseline_depth,
    #"predictor_hidden_size" : predictor_hidden_size,
    #"predictor_depth" : predictor_depth,
    #"error_hidden_size" : error_hidden_size,
    #"error_depth" : error_depth,
    #"proj_hidden_size" : proj_hidden_size, # for lstm feedforward
    #}


