import os
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 
os.environ["OMP_NUM_THREADS"] = "4" 

import numpy as np
import pandas as pd      

import modelSwitching

speedDF = pd.read_pickle("../data/full_day_speed_df.pckl")

W = modelSwitching.ModelSwitching(speedDF)
W.run()