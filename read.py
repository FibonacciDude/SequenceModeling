import pandas
import json
import numpy as np

for s in ["crossval/baseline_v0", "crossval/rnn_v0"]:
	d=json.load(open(s+".json"))
	values=np.array(list(d.values()))
	keys=np.array(list(d.keys()))
	ind=np.argpartition(values, 10)[:10]
	pd=pandas.DataFrame(d, index=[0])
	pd.to_excel(s+"_v1.xlsx")
