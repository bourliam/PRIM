import os
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 
os.environ["OMP_NUM_THREADS"] = "4" 

import numpy as np
import pandas as pd
import pprint
import matplotlib
import matplotlib.pyplot as plt        
import sys
import multiprocessing as mp
import time


from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import worker

speedDF = pd.read_pickle("../data/full_day_speed_df.pckl")


nSegments = len(speedDF)

print(nSegments)

W = worker.T_optim(speedDF)

W.run()