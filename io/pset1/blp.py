import pandas as pd
import numpy as np
from scipy.optimize import minimize
import scipy
from numba import jit, njit, prange
import time
import multiprocessing as mp
import pickle
from sklearn.linear_model import LinearRegression
import pyblp
import os

my_working_dir = "/home/chris/files/school/ucla/second_year/fall/io/pset1/"
os.chdir(my_working_dir)
os.getcwd()

df = pd.read_csv("blp.csv")
df = df.drop(df.columns[0], axis=1)



df.columns.str.split(".")

new_column_names = [i[-1] for i in df.columns.str.split(".")]

new_column_names[0] = "model_name"
new_column_names[1] = "model_id"
new_column_names[2] = "firm_id"
df.columns = new_column_names

df[["ln_hpwt", "ln_space", "ln_mpg", "ln_mpd", "ln_price"]] = \
    df[["hpwt", "space", "mpg", "mpd", "price"]].apply(lambda x: np.log(x))

# instrument change
df["trend"] = df.market.map(lambda x: x + 70)  # use with non pyblp instruments
# df["trend"] = df.market

df["cons"] = 1

df["s_0"] = np.log(1 - df.share.groupby(df["model_year"]).transform("sum"))

df["s_i"] = np.log(df.share)
df["dif"] = df.s_i - df.s_0
df["dif_2"] = np.log(df.share) - np.log(df.share_out)
df["ln_price"] = np.log(df.price)

df.columns
