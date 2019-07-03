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
    """
    A function to initialize each worker by giving it the data.
    """

    global X1_train
    global Y1_train
    global X2_train
    global Y2_train
    
    X1_train, Y1_train, X2_train, Y2_train = X1, Y1, X2, Y2

    

def fit_double_lasso(i):
    """
    The function to fit the models for each section.
    """
    
    lasso_1 = linear_model.LassoCV(n_jobs=4, cv=5, max_iter=5000, tol=0.001, selection='random', n_alphas=20, fit_intercept=False, eps=0.0001)
    lasso_2 = linear_model.LassoCV(n_jobs=4, cv=5, max_iter=5000, tol=0.001, selection='random', n_alphas=20, fit_intercept=False, eps=0.0001)
    
    
    lasso_1.fit(X1_train, Y1_train[:, i])
    lasso_2.fit(X2_train, Y2_train[:, i])

    return lasso_1, lasso_2

class ModelSwitching:

    def __init__(self, speedDF):
        """
        Initialization of the class. Creates the matrix Z.
        """
        
        print('Worker v0.3')
        self.Z_train, self.Z_test = self.split_Z(speedDF)
        self.Ts = []
        


    def split_Z(self, speedDF):
        """
        Creates the matrix Z, center it and then splits it into Z_train and Z_test according to days.
        """

        Z = []
        times_per_day = 56
        
        for i in range(int((speedDF.shape[1])/times_per_day)):
            Z.append(speedDF.iloc[:,i*times_per_day:(i+1)*times_per_day].values)

        print("Z Created!")
        n = len(Z)
        print("Number of days: n =", n)

        Z = np.array(Z)

        Z_train = Z[:45]                        #The train set is of 45 days.
        Z_test = Z[45:]

        M = (1/45) * Z_train.sum(axis=0)        #Centering
        for i in range(45):
            Z_train[i] = Z_train[i] - M
        for i in range(61-45):
            Z_test[i] = Z_test[i] - M

        print("Z centrée")
        return Z_train, Z_test



    def cut_Z(self, Z, T):
        """
        Cutting Z in two at time T, and creating the X's and Y's for training.
        """
        X1 = Z[:, :, :T]
        Y1 = Z[:, :, 1:T+1]
        X2 = Z[:, :, T:-1]
        Y2 = Z[:, :, T+1:]

        X1 = np.concatenate(X1, axis=1)
        Y1 = np.concatenate(Y1, axis=1)
        X2 = np.concatenate(X2, axis=1)
        Y2 = np.concatenate(Y2, axis=1)

        return X1.T, Y1.T, X2.T, Y2.T


    def compute_mse(self, results, T, nSegments):
        """
        Computing the mse of a step by using the cross validation results in mse_path_
        """
        times_per_day = 56
        mse=0
        for i in range(nSegments):
            mse += T * np.mean(results[i][0].mse_path_[np.where(results[i][0].alphas_ == results[i][0].alpha_)[0][0]])
            mse += ((times_per_day-1)-T) * np.mean(results[i][1].mse_path_[np.where(results[i][1].alphas_ == results[i][1].alpha_)[0][0]])
        mse = mse/((times_per_day-1)*nSegments)
        return mse



    def one_step(self, T):
        """
        Perfoms one step of the algorithm, ie trains the two models and computes the mse for a given T.
        """
        print("\n--------------------")
        print("------ STEP", T ,"------")
        print("--------------------\n")

        print('Splitting data')
        X1_train, Y1_train, X2_train, Y2_train = self.cut_Z(self.Z_train, T)
        print("Train: X1 shape:", X1_train.shape, "X2 shape:", X2_train.shape, '\n')
        
        nSegments = X1_train.shape[1]
        
        nb_proc=12                                     #number of simultaneous processus to use during trainning
        
        print("Training with", nb_proc,"processus...")

        start = time.time()
        pool = mp.Pool(nb_proc, init, [X1_train, Y1_train, X2_train, Y2_train])
        
        results = pool.map(fit_double_lasso, range(nSegments))
        end=time.time()

        print("Training done in ", end - start, "seconds")
        pool.close()
        
        #Computing the MSE for this step
        mse = self.compute_mse(results, T, nSegments)
        print('MSE:', mse)
        
        
        alphas = [] 
        
        for i in range(nSegments):
            alphas.append([results[i][0].alpha_,results[i][1].alpha_])   #alpha[i][0] = alpha pour la premiere periode pour la section i
        alphas = np.array(alphas)
        return mse, alphas



    def run(self, times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 52, 53, 54]):
        """
        Main function of the class.
        """
        self.mse_per_step = np.array([])
        self.all_alphas = np.array([])
        start_time = time.time()
        for t in times:
            mse, alphas = self.one_step(t)
            self.mse_per_step = np.append(self.mse_per_step, mse)
            self.all_alphas = np.append(self.all_alphas, alphas)
            np.savetxt('alphas_per_T_new2.txt', self.all_alphas)


        print("Done in", time.time() - start_time, "seconds.")
        print("T's", times)
        print("Scores:", self.mse_per_step)
