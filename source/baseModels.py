
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda

from statsmodels.tsa.ar_model import AR


class BaseModels:
    '''This class contains the base models that we aim to surpass.
    prédicteur historique,
    prédicteur historique sur le temps d’interet,
    dernière valeur observée,
    AR(5).
    '''

    def __init__(self, model, historic_data=None):
        '''Base models class initialisation
        
        Arguments:
            model {string} -- The model to create ['lastValue', 'historic', 'timehistoric', 'AR5']
            historic_data {ps.Series} -- The historical data, typically updatedSpeed.iloc[:,:x_train.shape[0]]
        '''

        self.history = historic_data

        if model == 'lastValue':
            self.model = self.lastValueModel()
        elif model == 'historic':
            self.model = self.historicModel()  
        elif model == 'timeHistoric':
            self.model = self.timeHistoricModel()
        #elif model == 'ar5':
            #self.model = self.AR5()
        else:
            self.model = None

    def lastValueModel(self, loss='MSE', optimizer='adam', nb_segments=748):
        '''This model only predicts the last known value.
        
        Keyword Arguments:
            loss {str} -- The loss used (default: {'MSE'})
            optimizer {str} -- The optimizer used (default: {'adam'})
            nb_segments {int} -- The number of segments studied (default: {748})
        
        Returns:
            Model -- The keras model
        '''

        inputs = Input(shape=(None, nb_segments), name='input')
        
        predictions = Lambda(lambda x: x[:, -1])(inputs)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss=loss, optimizer=optimizer)
        return model



    def historicModel(self, loss='MSE', optimizer='adam', nb_segments=748):
        '''This model predicts the average of the historical values for each section
        
        Keyword Arguments:
            loss {str} -- The loss used (default: {'MSE'})
            optimizer {str} -- The optimizer used (default: {'adam'})
            nb_segments {int} -- The number of segments studied (default: {748})
        
        Returns:
            Model -- The keras model
        '''

        inputs = Input(shape=(None, nb_segments), name='input')
        
        historic_values = self.history.mean(axis=1).values

        def const(x):
            vec = tf.constant(historic_values,  dtype='float32')
            matrix = tf.ones_like(x[:,1,:]) * vec
            return matrix
        
        predictions = Lambda(const)(inputs)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss=loss, optimizer=optimizer)
        return model



    def timeHistoricModel(self, loss='MSE', optimizer='adam', nb_segments=748):
        ''' NOT DONE YET
        
        This model predicts the average of the historical values at the current time for each section

        Keyword Arguments:
            loss {str} -- The loss used (default: {'MSE'})
            optimizer {str} -- The optimizer used (default: {'adam'})
            nb_segments {int} -- The number of segments studied (default: {748})

        Returns:
            Model -- The keras model
        '''

        inputs = Input(shape=(None, nb_segments), name='input')
        
        historic_values = self.history.mean(axis=1).values

        def const(x):
            vec = tf.constant(historic_values,  dtype='float32')
            matrix = tf.ones_like(x[:,1,:]) * vec
            return matrix
        
        predictions = Lambda(const)(inputs)

        model = Model(inputs=inputs, outputs=predictions)
        model.compile(loss=loss, optimizer=optimizer)
        return model


    def AR5(self, train):
        '''NOT DONE YET
        
        AR5 implementation

        '''

        model = AR(train)
        model = model.fit()
        return model
        
