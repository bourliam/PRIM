import os
os.environ["MKL_NUM_THREADS"] = "4" 
os.environ["NUMEXPR_NUM_THREADS"] = "4" 
os.environ["OMP_NUM_THREADS"] = "4" 


import multiprocessing as mp

import numpy as np
import pandas as pd
import pprint
import matplotlib
import matplotlib.pyplot as plt        
import time

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

X1_train, Y1_train, X2_train, Y2_train = None, None, None, None



def init(X1, Y1, X2, Y2):
    global X1_train
    global Y1_train
    global X2_train
    global Y2_train
    
    X1_train, Y1_train, X2_train, Y2_train = X1, Y1, X2, Y2

    

def fit_lasso_double(i):
    
    A_lasso_parra_1 = linear_model.LassoCV(n_jobs=4, cv=5, max_iter=5000, tol=0.001, selection='random', n_alphas=20, fit_intercept=False, eps=0.0001)
    A_lasso_parra_2 = linear_model.LassoCV(n_jobs=4, cv=5, max_iter=5000, tol=0.001, selection='random', n_alphas=20, fit_intercept=False, eps=0.0001)
    
    
    A_lasso_parra_1.fit(X1_train, Y1_train[:, i])
    A_lasso_parra_2.fit(X2_train, Y2_train[:, i])

    return A_lasso_parra_1, A_lasso_parra_2

class T_optim:

    def __init__(self, speedDF):
        print('worker v0.3')
        self.Z_train, self.Z_test = self.split_Z(speedDF)
        self.Ts = []
        


    def split_Z(self, speedDF):
        Z = []
        
        times_per_day = 56
        
        for i in range(int((speedDF.shape[1])/times_per_day)):
            Z.append(speedDF.iloc[:,i*times_per_day:(i+1)*times_per_day].values)

        print("Z3 Created!")
        n = len(Z)
        print("n =", n)

        Z = np.array(Z)

        Z_train = Z[:45]
        Z_test = Z[45:]

        M = (1/45) * Z_train.sum(axis=0)
        for i in range(45):
            Z_train[i] = Z_train[i] - M
        for i in range(61-45):
            Z_test[i] = Z_test[i] - M

        print("Z Centré !")
        return Z_train, Z_test



    def X_Y(self, Z, T):
        X1 = Z[:, :, :T]
        Y1 = Z[:, :, 1:T+1]
        X2 = Z[:, :, T:-1]
        Y2 = Z[:, :, T+1:]
        X1 = np.concatenate(X1, axis=1)
        Y1 = np.concatenate(Y1, axis=1)
        X2 = np.concatenate(X2, axis=1)
        Y2 = np.concatenate(Y2, axis=1)
        return X1.T, Y1.T, X2.T, Y2.T



    def one_step(self, T):
        print("\n--------------------")
        print("------ STEP", T ,"------")
        print("--------------------\n")
        print('Splitting data')
        X1_train, Y1_train, X2_train, Y2_train = self.X_Y(self.Z_train, T)
        print("Train: X1 shape:", X1_train.shape, "X2 shape:", X2_train.shape)
        X1_test, Y1_test, X2_test, Y2_test = self.X_Y(self.Z_test, T)
        print("Test: X1 shape:", X1_test.shape, "X2 shape:", X2_test.shape)
        print()

        nSegments = X1_train.shape[1]
        
        nb_proc=12
        
        print("Training with", nb_proc,"...")
        start = time.time()
        
        pool = mp.Pool(nb_proc, init, [X1_train, Y1_train, X2_train, Y2_train])
        
        results_double = pool.map(fit_lasso_double, range(nSegments))
        end=time.time()
        print("Training done in ", end - start, "seconds")
        pool.close()
        
        times_per_day = 56
        
        mse=0
        for i in range(nSegments):
            mse += T * np.mean(results_double[i][0].mse_path_[np.where(results_double[i][0].alphas_ == results_double[i][0].alpha_)[0][0]])
            mse += ((times_per_day-1)-T) * np.mean(results_double[i][1].mse_path_[np.where(results_double[i][1].alphas_ == results_double[i][1].alpha_)[0][0]])
        mse = mse/((times_per_day-1)*nSegments)

        print('MSE:', mse)
        
        
        alphas = [] 
        
        for i in range(nSegments):
            alphas.append([results_double[i][0].alpha_,results_double[i][1].alpha_])   #alpha[i][0] = alpha pour la premiere periode pour la section i
        alphas = np.array(alphas)
        return mse, alphas


    def run(self):
        self.Ts = np.array([])
        self.all_alphas = np.array([])
        times = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 52, 53, 54]
        for t in times:
            mse, alphas = self.one_step(t)
            self.Ts = np.append(self.Ts, mse)
            self.all_alphas = np.append(self.all_alphas, alphas)
            np.savetxt('alphas_per_T_new2.txt', self.all_alphas)

        
        print("DONE")
        print("T's", times)
        print("Scores:", self.Ts)
