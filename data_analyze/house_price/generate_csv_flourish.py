import os
import csv
import re
from save_data import dump_pickle,save_dict_to_pickle
import pandas as pd

data = [
    ["Alice", 25, "New York"],
    ["Bob", 30, "Los Angeles"],
    ["Charlie", 35, "Chicago"]
]

# df = pd.DataFrame(data, columns=["Name", "Age", "City"])
# df.to_csv("output.csv", index=False)  # 不写入索引



all_data = dump_pickle("data.pkl")

# print(len(all_data))

all_data = dict(sorted(all_data.items()))

# print(all_data.keys())

after_cal = []
col_names = []
# print((all_data))
for k,v in all_data.items():
    # print(k,v)
    col_names=list(v.keys())
    break

data = []

df = pd.DataFrame(data, columns=col_names)
print(df)

