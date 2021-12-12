import numpy as np
import csv
from collections import namedtuple
import json
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.optim as optim
from data_index import get_dataset, get_batch
import data_index
import math
import os
import itertools
import tqdm 
from models import *

info = data_index.info
print("loading dataset...")
dataset = torch.load("data/dataset.pt")
data_mask = torch.load("data/data_mask.pt")
data_lens = torch.load("data/lens.pt")
data_labels = torch.load("data/labels.npy")
data_tup = (dataset, data_mask, data_lens, data_labels) # tuple with everything
print("dataset loaded...")

def add_noise(X):
    X[..., :-1] += torch.randn(dat.shape).cuda() / 1e2 * .5
    return X

def loop(f_model, pred_model, optimizer, scheduler, data_tup, TOT, type_=0, update=True):
    predictor=config["predictor"]
    loss_calc = loss_calc_y if predictor else loss_calc_x
    losses, accs = [], []
    dataset, data_mask, data_lens, data_labels = data_tup
    model = pred_model if predictor else f_model
    model_type = ("rnn" if config["rnn"] else "baseline")
    if predictor:
        norm=torch.load("best_models/"+model_type+ "_norm.pt")
        mean,std=norm
        zs=torch.load("best_models/"+model_type+"_zs.pt")
    data_x = dataset if config["rnn"] else zs

    for batch_idx in range(TOT):
        batch, mask, lens, labels  = get_batch(
            batch_idx, data_x, data_mask, data_lens, data_labels, type=type_)

        if predictor:
            #batch=f_model(batch, lens).detach() # this is bad, just store all zs in one dataset
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
            #for name, param in model.named_parameters(): # get gradient
            #        print(name, param.grad.abs().sum())
            optimizer.step()
   
    if update:
        scheduler.step()

    return np.array(losses), np.array(accs)

def train_model(f_model, pred_model, optimizer, scheduler, data_tup):
    TOT = math.ceil(data_index.TRAIN / config["batch_size"] * config["p"])
    for e in tqdm.trange(config["epochs"]):
        losses, accs = loop(f_model, pred_model, optimizer, scheduler, data_tup, TOT=TOT, type_=0, update=True)

def validate(f_model, pred_model, data_tup, final=False):
    TOT = int(data_index.VAL // config["batch_size"])
    losses, accs = loop(f_model, pred_model, None, None, data_tup, TOT=TOT, type_=1, update=False) # no optim or schedule
    if final:
        print("Final validation")
        print("\tMean validation loss:", losses.sum() / data_index.VAL)
        if config["predictor"]:
            print("\tMean acc:", accs.sum() / data_index.VAL)
    return (losses.sum() / data_index.VAL) if not config["predictor"] else (accs.sum() / data_index.VAL)

def run(save):

    predictor=config["predictor"]
    mode = "pred" if predictor else "no_pred"
    f_model= RNN(config, mode=mode, model=config["rnn"]) if config["rnn"] else Baseline(config, mode=mode)

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


def crossval(predictor, rnn, iter_dict, skip=False):
    global config
    f_model_type=["baseline", "rnn"][rnn!=""]
    model_type=("pred" if predictor else ("rnn" if rnn else "baseline"))

    checkpoint_append = info.name + "_" + model_type
    dir_type="p" if predictor else "f"
    
    val_accs = {}
    if predictor:
        best_idx=open("best_models/best_"+f_model_type+"_idx.txt").read()
        best_params=np.load("checkpoints/"+info.name+"_"+f_model_type+"_%s_params.npy"% best_idx)
    
    if not skip:
        for idx, params in enumerate(itertools.product(*iter_arr)):
            print("Currently:", config["rnn"])
            params=np.array(params)
            config=update(list(iter_dict.keys()), params, config, output=True)
            config["checkpoint_dir_"+dir_type] = checkpoint_append + "_%d.pt" % idx
            # add params to checkpoint
            if not predictor:
                np.save("checkpoints/"+checkpoint_append+"_%d_params.npy" % idx, params)
            else:
                f_model_keys = list(json.load( open("crossval_config_%s.json" % (f_model_type))).keys())
                config=update(f_model_keys, best_params, config)
            save="%s_%s_%d" % (info.name, model_type, idx)
            loss=run(save)  # run
            val_accs[idx]=np.inf if np.isnan(loss) else loss

        # save and get best
        json.dump(val_accs, open("crossval/%s_v%d.json" % (f_model_type, predictor), "w"))

        keys=np.array(list(val_accs.keys()))
        vals=np.array(list(val_accs.values()))
        best_idx=keys[vals.argmin()]
        best_dir="%s_%d" % (checkpoint_append, best_idx)
        best_params=np.load("checkpoints/"+info.name+"_"+f_model_type+"_%s_params.npy"% best_idx)


    # do regardless, we want the best model in either case
    val_accs = json.load(open("crossval/%s_v%d.json" % (f_model_type, predictor)))
    keys=np.array(list(val_accs.keys()))
    vals=np.array(list(val_accs.values()))
    best_idx=keys[vals.argmin()]
    best_dir="%s_%d" % (checkpoint_append, int(best_idx))
    best_params=np.load("checkpoints/"+info.name+"_"+f_model_type+"_%s_params.npy"% best_idx)

    config=update(list(iter_dict.keys()), best_params, config)

    # load std, mean of z vector to normalize (for predictor)
    if not predictor:
        # save model
        os.system("cp checkpoints/%s.pt best_models/%s_best_%s.pt" % (best_dir, info.name, model_type))
        print("Computing and saving norm...")
        f_model= RNN(config, model=config["rnn"], mode="pred") if rnn else Baseline(config, mode="pred")
        f_model.load_state_dict(torch.load("best_models/%s_best_%s.pt" % (info.name, model_type)))
        zs=f_model(dataset, data_lens.cpu().numpy())
          # all data, all sequence (bs, seq, feature_size-1)
        # save zs as well
        torch.save(zs, "best_models/"+model_type+"_zs.pt")
        std, mean = torch.std_mean(zs[data_index.TEST+data_index.VAL:].mean(1), dim=0) # only train data + mean of sequence
        norm=torch.stack((mean, std))
        torch.save(norm, "best_models/"+model_type+"_norm.pt")
        print("Norm saved and computed")
        with open("best_models/best_%s_idx.txt"%model_type, "w") as f:
            f.write(str(best_idx))
    else:
        os.system("cp checkpoints/%s.pt best_models/%s_%s_pred_best.pt" % (best_dir, info.name, f_model_type))
        with open("best_models/best_%s_pred_idx.txt"%model_type, "w") as f:
            f.write(str(best_idx))

    return val_accs

if __name__ == "__main__":

    config = json.load(open("config.json"))
    print(config)
    torch.manual_seed(config['seed'])
    #x_size = info.feature_size * config["second_split"]
    rnn_model=config["rnn"]

    # cross_validation
    # universal ---
    lr = np.logspace(start=-5, stop=-1, num=7)
    l2_reg = np.logspace(start=-4, stop=-2, num=2)
    l2_reg = np.concatenate((l2_reg, [0]))
    cross_type=["baseline", "rnn", "pred"]
    val_accs_comb=[]

    for predictor in [False, True]:
        config["epochs"]=config["epochs"]+2 if predictor else config["epochs"]
        for rnn in [rnn_model, ""]:
            type_ = cross_type[rnn != ""] if not predictor else cross_type[-1]
            config["rnn"]=rnn
            config["predictor"]=predictor
            #config["seed"]= 123 if predictor else config["seed"]
            every=config["print_every"]
            print("predictor: %s, rnn: %s, %s" % (predictor, rnn, type_))
            iter_dict = json.load(open("crossval_config_"+type_+".json"))
            iter_dict.update({
            "l2_reg" : l2_reg,
            "lr" : lr
            })
            iter_arr = list(iter_dict.values())
            skip = not predictor #or rnn != ""
            #skip=False
            #skip = rnn != ""
            if predictor:
                config["p"] = .001 # of data
            val_accs=crossval(predictor, rnn, iter_dict, skip=skip)
            val_accs_comb.append(val_accs)
         #print("Finalized %s... Got loss:.4f%, acc:.4f%" % (loss, acc))
