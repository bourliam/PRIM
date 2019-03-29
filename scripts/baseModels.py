
import numpy as np
import pandas as pd

import statsmodels.tsa.api as smt

import datetime
from datetime import timedelta

class BaseModels:
    '''This class contains the base models that we aim to surpass.
    prédicteur historique,
    prédicteur historique sur le temps d’interet,
    dernière valeur observée,
    AR(5).
    '''

    def __init__(self, model, historic_data=None,nb_segments=748):
        '''Base models class initialisation
        
        Arguments:
            model {string} -- The model to create ['lastValue', 'historic', 'timehistoric', 'AR5']
            historic_data {ps.Series} -- The historical data, typically updatedSpeed.iloc[:,:200]
        '''

        self.history = historic_data
        self.type = model

        if model == 'AR5':
            self.models = self.AR5_train(historic_data)
        


    def predict(self, x, time=datetime.time(14,0)):
        y=[]
        if self.type == 'lastValue':
            y = x.iloc[:, -1]
        elif self.type == 'historic':
            y = self.history.mean(axis=1).values
        elif self.type == 'timeHistoric':
            columns = [d for d in self.history.columns if d.time()==time]
            y = self.history[columns].mean(axis=1).values
        elif self.type == 'AR5':
            y = self.AR5(x)
        return y



    def AR5_train(self, updatedSpeeds):
        print('Training the AR(5) model')
        train = updatedSpeeds.values
        train_dates = updatedSpeeds.columns
        print('Train data shape:', train.shape)
        
        print('\nFilling the voids...')
        delta = timedelta(minutes=15)

        train_filled = train[:,0].reshape(-1,1)
        train_dates_filled=[train_dates[0]]
        for i in range(1, len(train_dates)):
            while train_dates_filled[-1] + delta != train_dates[i]:
                train_dates_filled.append(train_dates_filled[-1] + delta)
                train_filled = np.concatenate((train_filled, np.zeros((train_filled.shape[0],1))), axis=1)
                
            train_filled = np.concatenate((train_filled, train[:,i].reshape(-1,1)), axis=1)
            train_dates_filled.append(train_dates[i])
            
        train_dates_filled = np.array(train_dates_filled)
        print('Filling done. New train data shape:', train_filled.shape)

        print('\nTraining the models...')
        
        max_lag = 5
        print('Params: max_lag:', max_lag)
        models = [smt.AR(train_filled[i], dates=train_dates_filled, freq='15min').fit(maxlag=max_lag, trend='c') for i in range(train_filled.shape[0]) ]
        print('\nTraining finished !')

        return models


    def AR5_single_pred(self, data, section):
        coefs = self.models[section].params
        pred = coefs[0]
        
        for i in range(1,6):
            pred += coefs[i] * data.iloc[section, 5-i]
        
        return pred


    def AR5(self, x):

        predictions = np.array([self.AR5_single_pred(x, i) for i in range(x.shape[0])])
        
        return predictions


    
