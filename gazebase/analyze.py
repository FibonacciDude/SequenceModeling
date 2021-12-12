import numpy as np
import itertools
import json


# cross_validation
lr = np.logspace(start=-4.1, stop=-1.9, num=5)
hidden_size = [32, 64] # lstm
l2_reg = np.logspace(start=-4, stop=-2, num=2)
inside_layers = [0, 1]
proj_hidden_size = [32, 64] # feedforward
num_layers = [1, 2]

iter_dict = {"num_layers" : num_layers, 
"inside_layers" : inside_layers,
"proj_hidden_size" : proj_hidden_size,
"l2_reg" : l2_reg,
"hidden_size" : hidden_size,
"lr" : lr}

iter_arr = [num_layers, inside_layers, proj_hidden_size, l2_reg, hidden_size, lr]
iter_dict = json.load(open("config.json"))

acc = {}
index = {}
config = {}

#config = json.load(open("config_net.json"))
    
for idx, params in enumerate(itertools.product(*iter_arr)):
    points = "gazebase_baseline_%d.txt"  % idx
    for i, (k, v) in enumerate(iter_dict.items()):
        config_dict[k] = params[i]
        acc_params = float(open("crossval/" + points).read())
    config_dict["checkpoint_dir"] = points
    acc[idx] = (acc_params)
    index[idx] = (params)
    config[idx] = config_dict

arr = np.array(list(acc.values()))
argmin = arr.argmin()
indexes = np.array(list(index.values()))
#print(list(iter_dict.keys()))
#print(indexes[argmin])
print(acc[argmin])
print(config[argmin])

"""
with open('config_val.json', 'w', encoding='utf-8') as f:
    json.dump(config[argmin], f, ensure_ascii=False, indent=4)
"""
