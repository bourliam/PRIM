
import datetime

import numpy as np
import pandas as pd
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import CustomUtils
from functools import reduce
import matplotlib
import folium
import Plotting

from datetime import timedelta


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
            historic_data {ps.Series} -- The historical data, typically updatedSpeed.iloc[:,:200]
        '''

        self.history = historic_data
        self.type = model

        if model == 'AR5':
            self.models = self.AR5_train(historic_data)
        


    def predict(self, x, time=datetime.time(14,0)):
        '''Method to make predictions, depending on the model used.
        
        Arguments:
            x {pandas DataFRame} -- The input, the last data, just before the what we want to predict.
        
        Keyword Arguments:
            time {datetime.time} -- Only for timeHistoric model. The time for which we want the prediction. (default: {datetime.time(14,0)})
        
        Returns:
            numpy.array -- The array of values predicted.
        '''

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
        '''The method to train teh AR(5) model
        
        Arguments:
            updatedSpeeds {pandas.DataFrame} -- The historic data for training
        
        Returns:
            list -- A list of AR5 models. One per section.
        '''

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
        '''Method to predict the foloowing value for a single section
        
        Arguments:
            data {pandas.DataFrame} -- The input data
            section {int} -- The id of the section considered
        
        Returns:
            float -- The speed prediction for this one section
        '''

        coefs = self.models[section].params
        pred = coefs[0]
        
        for i in range(1,6):
            pred += coefs[i] * data.iloc[section, 5-i]
        
        return pred


    def AR5(self, x):
        '''
        The method for AR5 predictions
        
        Arguments:
            x {panda.DataFrame} -- The input data. (data of last lags)
        
        Returns:
            numpy.array -- The predictions
        '''

        predictions = np.array([self.AR5_single_pred(x, i) for i in range(x.shape[0])])
        
        return predictions


    
class DataModel:
    
    def __init__(self,data, input_lag, output_lag, sequence_length,scale_max=False,scale_log=False,shift_mean=False,y_only=False,add_time=False,max_value=130,valid_split=0.8,min_max_scale=False,differentiate_y=False):

        self.data = data
        self.input_lag = input_lag
        self.output_lag = output_lag
        self.sequence_length = sequence_length
        self.scale_max = scale_max
        self.scale_log = scale_log
        self.shift_mean = shift_mean
        self.y_only=y_only
        self.add_time = add_time
        self.max_value = max_value
        self.min_max_scale = min_max_scale
        self.differentiate_y=differentiate_y
        self.model=None
        self.valid_split=valid_split
        self.x,self.y,self.t =self.getXY()
        self.n_segments = len(data)

        self.__reversed_process=[]
    
    
    
    def getDaysTypes(self):
        """
        returns the types of day (monday to friday), and real value representing the time of day for each example (number of seconds/ 60*60) 
        
        """
        day_types = pd.DatetimeIndex(self.t.reshape(-1)).weekday.values.reshape(self.t.shape)
        time_fraction = (CustomUtils.timeToSeconds(pd.DatetimeIndex(self.t.reshape(-1)))/(60*60)).values.reshape(self.t.shape)
        time_input = np.concatenate([day_types,time_fraction],1)
        return time_input[:int(len(self.x)*(self.valid_split))],time_input[int(len(self.x)*(self.valid_split)):]
    
    def getExamples(self,sequence,hours):
        """
        create examples (inputlag,outputlag) for one day by shifting time by one step (default to 15 minutes)
        
        """
        
        sequence_length=len(sequence)
        sub_sequence_length = self.input_lag+self.output_lag
        if sub_sequence_length > sequence_length :
            raise ValueError("sequence length {} too small for lags : {},{}".format(sequence_length,self.input_lag,self.output_lag))
        return [sequence[i:i+self.input_lag] for i in range(0,sequence_length-sub_sequence_length+1,1)],\
               [sequence[i+self.input_lag:i+self.input_lag+self.output_lag] for i in range(0,sequence_length-sub_sequence_length+1,1)],\
               [hours[i+self.input_lag:i+self.input_lag+self.output_lag] for i in range(0,sequence_length-sub_sequence_length+1,1)]  
    
    def getXY(self):
        """
        create X and Y matricies out of the original speed dataFrame
        X shape : n_samples, inputLag, n_segments
        Y shape : n_samples, outputLag, n_segments (middle dimension is dropped if output lag =1)

        """
        
        
        nsegs,ntime=self.data.shape
        if(ntime%self.sequence_length)!= 0 :
            raise ValueError("sequence length {} not compatible with number of time features {}".format(self.sequence_length,ntime))

        shapedData = self.data.values.T.reshape(int(ntime/self.sequence_length),self.sequence_length,nsegs)
        timestamps = pd.Series(self.data.columns).values.reshape(int(ntime/self.sequence_length),self.sequence_length)
        
        examples=[self.getExamples(x,hours) for x,hours in zip(shapedData,timestamps)]

        x,y,t = list(zip(*examples))
        return np.concatenate(x), np.concatenate(y), np.concatenate(t)
    
    
    def getIndexes(self,idx):
        """
        restore the time index of all lags used given a sample position on the X matrix
        """
        cx,cy= (idx +(self.input_lag+self.output_lag-1)*(idx//(self.sequence_length - self.input_lag-self.output_lag+1)),\
                idx +(self.input_lag+self.output_lag-1)*(idx//(self.sequence_length - self.input_lag-self.output_lag+1))+self.input_lag )
        return (self.data.columns[cx:cy].values,self.data.columns[cy:cy+self.output_lag].values)
    
    def scaleMax(self):
        """
        divide all values by a max_value (default 130)
        """
        self.__reversed_process.append(self.reverseScaleMax)
        self.x/=self.max_value
        if not self.y_only:
            self.y/=self.max_value
        
    def scaleMinMax(self):
        """
        normalize data to 0-1 scale
        """
        
        self.__reversed_process.append(self.reverseMinMaxScale)
        self.min =self.x[:int(len(self.x)*(self.valid_split))].min()
        self.max =self.x[:int(len(self.x)*(self.valid_split))].max()
        diff = self.max - self.min
        self.x = (self.x-self.min)/diff
        self.y = (self.y-self.min)/diff

        
    def reverseMinMaxScale(self,x):
        """
        reverse normalisation
        """
        return x*(self.max-self.min)+self.min
    
    def reverseScaleMax(self,y):
        """
        reverse max scaling
        """
        return y*self.max_value

        
    def scaleLog(self):
        """
        apply log(x+1) on all values
        
        """
        self.__reversed_process.append(self.reverseScaleLog)
        
        self.x=np.log1p(self.x)
        self.y=np.log1p(self.y)
        
    def reverseScaleLog(self,y):
        """
        reverse the log scale
        """
        return np.expm1(y)       
        
    def addTime(self):
        """
        add time represntation of all input lags
        """
        self.__reversed_process.append(self.removeTime)
        self.x=np.concatenate((self.x,self.t.reshape(-1,self.t.shape[1],1)),2)
        
        
    def removeTime(self,y):
        """
        remove time for input
        """
        if y.shape == self.x.shape :
            return np.delete(data_model.x,data_model.x.shape[2]-1,axis=2)
        return y
        
    def shiftMean(self):
        """
        Compute local time mean on train data and substract it from all data
        
        """
        self.__reversed_process.append(self.resetMean)
        self.means  =  self.data[self.data.columns[:(int(len(self.data.columns)*self.valid_split))]].mean(axis=1).values
        if not self.y_only :
            self.x-=self.means
        self.y-=self.means
        
    def resetMean(self,y):
        """
        add local time mean to input
        """
        return y+self.means
        
        
    def preprocessData(self):
        """
        apply different preprocessings if requested
        
        """


        if self.differentiate_y :
            self.differentiateY()

        if self.shift_mean :
            self.shiftMean()
            
        if self.scale_max :
            self.scaleMax()
            
        if self.scale_log :
            self.scaleLog()
        if self.min_max_scale : 
            self.scaleMinMax()
        if self.add_time :
            self.addTime()
        if self.output_lag == 1 :
            self.y=self.y.reshape(-1,self.y.shape[2])

    def getRawYData(self,y):
        """
        reverse all preprocessings done on data
        """
        return reduce(lambda res, f:f(res), self.__reversed_process[::-1], y)

    
    def differentiateY(self):
        """
        compute the difference between output y and the last x lag (transforming the problem to prediction of change from last value)
        """
        self.__reversed_process.append(self.reverseDifferentiatedY)
        self.y = self.y-self.x[:,-1:,:]
    
    def reverseDifferentiatedY(self,y):
        """
        reverse y differntiation
        """
        if len(y.shape)>2  : return y
        if self.output_lag >1 :
            return y+self.x[:,-1:,:]
        return y+self.x[:,-1,:]
    def mse(self,p,y=None):
        """
        Compute mse between predictions and true values on the original scale
        """
        
        pred = self.getRawYData(p)
        if y is not None :
            raw_y = self.getRawYData(y)
        else :
            raw_y = self.getRawYData(self.y)
        return np.mean((pred-raw_y)**2)
    
    def mae(self,p,y=None):
        """
        Compute mae between predictions and true values on the original scale
        """
        pred = self.getRawYData(p)
        if y is not None :
            raw_y = self.getRawYData(y)
        else :
            raw_y = self.getRawYData(self.y)
        return np.mean(abs(pred-raw_y))
    
    def trainSplit(self):
        """
        split data into train, validation sets (using valid_split attribute)
        """
        
        x_train = self.x[:int(len(self.x)*(self.valid_split))]
        x_test = self.x[int(len(self.x)*(self.valid_split)):]
        y_train = self.y[:int(len(self.x)*(self.valid_split))]
        y_test = self.y[int(len(self.x)*(self.valid_split)):]
        return x_train,y_train,x_test,y_test
    
    def getSplitSequences(self,values,sequence_length,skip=0):
        """
        add nans where prediction is not possible (instead of linking non sequential data)
        
        """
        
        def addNans(values,sequence_length,skip):

            values=values.reshape(-1,sequence_length)
            nans=np.array([np.nan]*(values.shape[0]*(skip+1))).reshape(values.shape[0],-1)
            values = np.concatenate((values,nans),axis=1).reshape(-1)
            return values
        return addNans(np.arange(len(values)),sequence_length,skip), addNans(values,sequence_length,skip)
    
    def restorePredictionsAsDF(self,preds):
        """
        create a data frame from predictions with time index
        """
        index = [self.getIndexes(i)[1][0] for i in range(len(preds))]
        df = pd.DataFrame(self.getRawYData(preds),index=index,columns=self.data.index)
        return df.T
    
    def restoreXAsDF(self,x):
        """
        create data frame from X matrix
        """
        index = [self.getIndexes(i)[1][0] for i in range(len(x))]
        df = pd.DataFrame(self.getRawYData(x).swapaxes(1,2).tolist(),index=index,columns=self.data.index)
        return df.T
    
    def predict(self,split="full"):
        
        """
        make prediction on the data using the stored model (baseline or lstm for now)
        """
            
        time_index = [self.getIndexes(i)[1][0] for i in range(len(self.x))]
        
        secondary_input = self.getDaysTypes()
        if split.lower() == "full":
            main_input = self.x
            secondary_input = np.concatenate(secondary_input)
        if split.lower() == "train":
            main_input,*_ = self.trainSplit()
            time_index = time_index[:int(len(self.x)*(self.valid_split))]
            secondary_input=secondary_input[0]
        if split.lower() == "test":
            *_,main_input,_ = self.trainSplit()
            time_index = time_index[int(len(self.x)*(self.valid_split)):]
            secondary_input=secondary_input[1]
            
            
            
        if isinstance(self.model,BaseModels):
            return np.array([self.model.predict(pd.DataFrame(x_i).T,time_index[i].time()) for  i,x_i in enumerate(main_input)])
            
        if(len(self.model.input_shape)==1):
            if len(self.model.outputs)>1 :
                return self.model.predict(main_input)[0]
        
            return self.model.predict(main_input)
        if len(self.model.outputs)>1 :
            return self.model.predict([main_input, secondary_input])[0]
        return self.model.predict([main_input, secondary_input])
    
    
class DataCleaner:
    """
    this calse is used to clean data:
    reindexing new roads
    merging roads data
    dropping weekends
    imputing missing values
    droping unwanted erroneous data
    ...
    """
    def __init__(self,data,segmentsMeta,mergeResults,counts=None):
        
        self.data = data
        self.counts =counts
        self.segmentsMeta=segmentsMeta
        self.mergeResults=mergeResults
        self.mergedIndex=None
        self.dropWeekends()
        if self.countsAvailable : 
            self.dropErroneousData()
        self.computeMergeData()
        self.fillNaWithHistoricalValues()
        self.segments_tags = segmentsMeta[segmentsMeta.segmentID.isin(self.data.index)].set_index('segmentID').reindex(self.data.index).tag.apply(lambda x :x['highway'])

    def countsAvailable(self):
        return not self.counts is None
    
    def dropWeekends(self):
        """
        drop weekends from data
        """
        self.data.drop(
            self.data.columns[
                [ x.date().weekday()>=5 for x  in self.data.columns]
            ],
            axis=1,
            inplace=True)
        if self.countsAvailable():
            self.counts.drop(
                self.counts.columns[
                    [ x.date().weekday()>=5 for x  in self.counts.columns]
                ],
                axis=1,
                inplace=True
            )            
        
    def fillNaWithHistoricalValues(self):
        """
        replacing missing data with local time mean values
        """
        oldIdx = self.data.columns
        # splitting index form datetime to multi index (date,time)
        idx=[pd.to_datetime(self.data.columns.values).date,pd.to_datetime(self.data.columns.values).time]
        mIdx=pd.MultiIndex.from_arrays(idx,names=['day','time'])
        self.data.set_axis(mIdx,axis=1,inplace=True)
        # computing local time means
        self.data = self.data.add(
            self.data.isna()*self.data.groupby(by=self.data.columns.get_level_values(1),axis=1).mean(),
            fill_value=0)
        
        # resetting old index
        self.data.set_axis(oldIdx,axis=1,inplace=True)        
        
    def computeMergeData(self, thresh=0.8):
        """
        setting mean speed and sum counts as values for merged segments
        """
        self.mergedIndex=pd.Series(data=self.segmentsMeta.loc[self.mergeResults]['segmentID'].values,index = self.segmentsMeta['segmentID'].values)
        self.data = self.data.assign(newIndex =self.mergedIndex.reindex(self.data.index).values)
        self.data = self.data[~self.data.newIndex.isna()]
        self.data=self.data.groupby('newIndex').mean().dropna(thresh = int(thresh*len(self.data.columns)))
        if self.countsAvailable():
            self.counts = self.counts.assign(newIndex =self.mergedIndex.reindex(self.counts.index).values)
            self.counts = self.counts[~self.counts.newIndex.isna()]
            self.counts = self.counts.groupby('newIndex').sum().loc[self.data.index]
    
    def dropErroneousData(self):
        """
        drop some erroneous data will probably change (should be more dynamic)
        """
        days_count =self.counts.groupby(pd.DatetimeIndex(self.data.columns).date,axis=1).sum().sum()
        days_quarter_count = pd.Series(self.data.columns.date).value_counts()
        days_index=np.intersect1d(days_count[days_count>100000].index,days_quarter_count[days_quarter_count==20].index)
        self.data=self.data[self.data.columns[[ x.date() in days_index for x  in self.data.columns]]]
        self.counts = self.counts[self.data.columns[[ x.date() in days_index for x  in self.data.columns]]]
        
        
        
class ModelPlots:
    
    """
    functions used to plot results , losses of the model
    
    """
    def __init__(self,data_model, data_cleaner):
        self.data_model = data_model
        self.data_cleaner = data_cleaner
        self.preds = data_model.getRawYData(data_model.predict('full'))
        self.y = data_model.getRawYData(data_model.y)
        
    def createSubPlots(self,data, pltFunc=plt.plot, figsize=(12,12),titles=None):
        """
        create sub plots using data and pltFunc
        """
        nCols= int(np.sqrt(len(data)))+1
        plt.figure(figsize=figsize)
        for i,vals in enumerate(data):
            plt.subplot(nCols,nCols,i+1)
            plt.plot(vals)
            plt.xlabel("epochs")
            plt.ylabel("MSE")
            if type(titles)!=type(None):
                plt.title(titles[i])
        plt.tight_layout()
        
    def plotSegmentSeries(self,idx,subplot=False):
        """
        plot the series of a segment (both true and predicted values)
        """
        if not subplot :
            plt.figure(figsize=(30,4))
        plt.plot(*self.data_model.getSplitSequences(
                                                self.y[:,idx],
                                                self.data_model.sequence_length-self.data_model.input_lag,
                                                skip=self.data_model.input_lag)
                )
        
        plt.plot(*self.data_model.getSplitSequences(
                                                self.preds[:,idx],
                                                self.data_model.sequence_length-self.data_model.input_lag,
                                                skip=self.data_model.input_lag)
                )
        
        dates = np.array([self.data_model.getIndexes(i)[1][0] for i in range(len(self.y))])
        
        if not subplot :
            plt.xticks(ticks  = np.arange(len(self.y))[np.r_[:len(self.y)-1:30j].astype(int)],
                       labels = dates[np.r_[:len(self.y)-1:30j].astype(int)],rotation='vertical');
        plt.ylabel("Speed")
        plt.axvline((self.data_model.valid_split)*self.data_model.x.shape[0],c='r')
        plt.legend(['y','pred','validationSplit'])
        plt.title(" segment : {}, tag : {:}".format(idx,self.data_cleaner.segments_tags.iloc[idx]))

        
    def plotMultipleSegmentsSeries(self,ids=None):
        """
        plots multiple series using index in "ids" if provided else plots 20 series ordered by mean difference in  predictions
        """
        if ids is None : 
            ids = np.argsort(self.data_model.y.mean(axis=0)[0]-self.data_model.predict('full').mean(axis=0))[np.r_[:self.data_model.n_segments-1:20j].astype(int)]

        plt.figure(figsize=(24,36))
        for ix, xSample in enumerate(ids):
            plt.subplot(len(ids),1,ix+1)
            self.plotSegmentSeries(xSample,subplot=True)
        plt.tight_layout()
        
        
        
    def plotPredictionMatchHeatMap(self,split='full',size_rate=4,figsize=(10,10)):
        """
        density plot of rounded values of predictions and true data
        """
        
        train_split = int(len(self.y)*self.data_model.valid_split)
        if split.lower() == 'train':
            prdsVsYDF=pd.DataFrame([self.preds[:train_split].flatten(),self.y[:train_split].flatten()],index=['pred','y'])
        
        if split.lower()[:5]=='valid':
            prdsVsYDF=pd.DataFrame([self.preds[train_split:].flatten(),self.y[train_split:].flatten()],index=['pred','y'])
        
        if split.lower() =='full':
            prdsVsYDF=pd.DataFrame([self.preds.flatten(),self.y.flatten()],index=['pred','y'])
        prdsVsYDF=prdsVsYDF.T.astype(int)
        
        fig= plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(size_rate, size_rate)
        ax_main = plt.subplot(gs[1:, :-1])
        ax_x_hist = plt.subplot(gs[0, :-1],sharex=ax_main)
        ax_y_hist = plt.subplot(gs[1:, -1],sharey=ax_main)     
        ax_x_hist.hist(prdsVsYDF.pred.values,bins=len(set(prdsVsYDF.pred.values)),align='mid')
        ax_y_hist.hist(prdsVsYDF.y.values,bins=len(set(prdsVsYDF.y.values)),align='mid',orientation='horizontal')  


        prdsVsYDF=prdsVsYDF.groupby(['pred','y']).size().unstack().fillna(0).T

        prdmin  = -prdsVsYDF.columns.values.min()


        heat_map=ax_main.imshow(prdsVsYDF,aspect='auto',origin='bottom-left',cmap =plt.cm.gist_ncar,interpolation='spline16')
        ax_main.set_xticks(np.arange(len(prdsVsYDF.columns.values))[::12])
        ax_main.set_xticklabels(labels=prdsVsYDF.columns.values[::12])
        ax_main.plot([prdmin,130+prdmin],[0,130],c='red',linewidth=3)
        ax_main.set(xlabel="x Prediction", ylabel="y True")
        plt.colorbar(heat_map,ax=ax_y_hist)
        plt.tight_layout()
        
        
    def plotPredictions(self, yDF,predDF, timesteps,folium_map=None):
        """
        creates a map representing the error in prediction
        """
        
        if folium_map == None :
            folium_map = Plotting.getFoliumMap()
        layers=[]
        colors = ((np.abs(yDF.clip(lower=15) - predDF.clip(lower=15))+1)/(yDF.clip(lower=15)+1)).clip(upper=1)
        laggedX = self.data_model.restoreXAsDF(self.data_model.x)
        predSegs = self.data_cleaner.segmentsMeta[self.data_cleaner.segmentsMeta.segmentID.isin(self.data_cleaner.mergedIndex[self.data_cleaner.mergedIndex.isin(yDF.index)].index)]
        segment_tag=predSegs.tag.apply(lambda x:x['highway']).values
        segment_overall_mean = [self.data_model.data.mean(axis=1).loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
        segment_timestamp_mean = self.data_model.data.groupby(pd.DatetimeIndex(self.data_model.data.columns).time,axis=1).mean()
        
        for t in timesteps :
            colorList=[colors[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
            y= [yDF[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
            preds = [predDF[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
            segCounts=[self.data_cleaner.counts[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]
            timestampLaggedX= [laggedX[t].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]

            current_segment_timestamp_mean = [segment_timestamp_mean[t.time()].loc[self.data_cleaner.mergedIndex.loc[idx]] for idx in predSegs.segmentID]

            popups = ["segment : {:},<br>tag : {:},<br> y : {:.2f},<br> pred : {:.2f},<br> %error : {:.0f}%,<br> count : {:}<br>mean: {:}<br>timestamp_mean: {:}<br>x: {:} "\
                      .format(seg,seg_tag,yi,predi,props*100,count,mean,timestamp_mean,np.array(x).astype(int)) 
                      for seg,seg_tag,yi,predi,props,count,mean,timestamp_mean,x 
                      in zip(predSegs.segmentID,segment_tag,y,preds,colorList,segCounts,segment_overall_mean,current_segment_timestamp_mean,timestampLaggedX)]
            pos = yDF.columns.get_loc(t)
            layer = self.getPredictionLayer(predSegs,colorList,segCounts,folium_map,str(t),popups)
            layers.append(layer)

        return Plotting.stackHistotyLayers([*layers,folium.TileLayer()],folium_map)

    def getPredictionLayer(self,segments,colors,counts,folium_map,name='layer',popups=[]):
        """
        create folium layer for one timestamp prediction
        """
        layer = folium.plugins.FeatureGroupSubGroup(folium_map,name=name,show=False, overlay=False)
        [folium.PolyLine(locations=[lo[::-1] for lo in x['coordinates']],weight = count//5+1, color=matplotlib.colors.rgb2hex(plt.cm.brg_r(color/2)),popup=pop).add_to(layer) for x,color,pop,count in zip(segments['loc'],colors,popups,counts)]
        return layer
