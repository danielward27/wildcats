import pandas as pd
import numpy as np
import pyarrow
import os

df_list = []
df2_list = []
df3_list = []
missing_files = []
missing_thetas = []
missing_times = []
num_files = 10000

for i in range(1, num_files):
    filename = "./output/stats/sumstats{}.pickle".format(i)
    try:
        df = pd.read_pickle(filename)
        df = df.reset_index(drop=True)
        df_list.append(df)
        os.remove(filename)
    except pyarrow.lib.ArrowIOError:
        missing_files.append(i)

sum_stats = pd.concat(df_list, axis=0).reset_index(drop=True)

if len(missing_files) != 0:
    print("Warning: there are files missing in specified range")

sum_stats.to_csv("./output/summary_stats.csv", index=False)

for i in range(1, num_files):
    filename2 = "./output/thetas/theta{}.pickle".format(i)
    try:
        df2 = pd.read_pickle(filename2)
        df2 = df2.reset_index(drop=True)
        df2_list.append(df2)
        os.remove(filename2)
    except pyarrow.lib.ArrowIOError:
        missing_thetas.append(i)

thetas = pd.concat(df2_list, axis=0).reset_index(drop=True)

thetas.to_csv("./output/thetas.csv", index=False)

for i in range(1, num_files):
    filename3 = "./output/times/time{}.pickle".format(i)
    try:
        df3 = pd.read_pickle(filename3)
        df3_list.append(df3)
        os.remove(filename3)
    except pyarrow.lib.ArrowIOError:
        missing_times.append(i)

times = pd.DataFrame(df3_list)

times.to_csv("./output/times.csv", index=False)