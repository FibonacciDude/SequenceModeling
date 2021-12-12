import pandas
import json
import numpy as np

for i in [1]:
    for s in ["baseline_v"]: # "rnn_v"]:
        s = s+str(i)
        d = json.load(open("crossval/"+s+".json"))
        values = np.array(list(d.values()))
        keys = np.array(list(d.keys()))
        ind = np.argpartition(values, 10)[:10]
        d = {k: v for k, v in zip(list(range(10)), values[ind])}
        pd = pandas.DataFrame(d, index=[0])
        pd.to_excel("output/"+s+"_.1.xlsx")
