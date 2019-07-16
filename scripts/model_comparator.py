from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from   functools import reduce
import sys
import seaborn as sns
class ModelCompare:
    
    """
    functions used to compare results , losses different models    
    """
    def __init__(self,models_predictions,true_values,user_time_index=None,segments_index=None):
        """
        models_predictions is a dictionary model_name: predictiond dataframe
        true_values is the truth dataframe
        """

        self.models_predictions = models_predictions
        
        self.markers=dict(zip(self.models_predictions.keys(),[  '->',
                        ':,',
                        ':o',
                        ':v',
                        ':^',
                        ':<',
                        
                        ':s',
                        ':p',
                        ':*',
                        ':h',
                        ':H',
                        ':+',
                        ':x',
                        ':D',
                        ':d',
                        ':1',
                        ':2',
                        ':3',
                        ':4',
                        ':|',
                        ':_'
                     ]))
        for k,v in self.markers.items() :
#             self.markers[k]='-'
            pass
        self.true_values = true_values
        self.segments_index=segments_index
        
        if not segments_index is None :
            self.true_values=self.true_values.loc[segments_index].copy()
            for model_name, model_data in self.models_predictions.items() :
                self.models_predictions[model_name]=self.models_predictions[model_name].loc[segments_index].copy()

        if not self.compatibility() :
            print("check data shape",file=sys.stderr)
            print("shape used is the intersection of all columns")
        
            self.time_index = reduce(np.intersect1d, [df.columns for df in self.models_predictions.values()])
            self.time_index = np.intersect1d(self.time_index,pd.to_datetime(self.true_values.columns))
            print(self.time_index.shape)
        if not user_time_index is None :
            self.time_index = user_time_index
            

    def compatibility(self):
        true_shape = self.true_values.shape
        compatible = True
        for model_name, model_data in self.models_predictions.items() :
            if true_shape != model_data.shape :
                print(model_name +" has different shape from true values, shapes : model "+str(model_data.shape)+" true shape "+str(true_shape),file=sys.stderr)
                compatible=False
                
        return compatible
    
    
    
    
    def plotDiscreteSpeedError(self,intercept =0,xlabel=""):
        """
        plots average absolute error per discrete speed 
        """
        if not type(intercept) is int :
            intercept = intercept[self.time_index]
            if not self.segments_index is None :
                intercept = intercept.loc[self.segments_index].copy()
        
        for model_name, model_data in self.models_predictions.items() :

            error = abs((model_data[self.time_index] -self.true_values[self.time_index]).values.flatten().round())
            true_y=(self.true_values[self.time_index] + intercept).values.flatten().round()
            arsort=true_y.argsort()
            error = error[arsort]
            true_y = true_y[arsort]
            
            y_idx=np.unique(true_y,return_index=True)[0]
            
            split_idx = np.unique(true_y,return_index=True)[1][1:]
            
            y_mean_error=np.fromiter([np.mean(x) for x in np.split(error ,split_idx)],dtype=float)
            
            plt.plot(y_idx,y_mean_error,self.markers[model_name],label=model_name)
            
        plt.xlabel(xlabel)
        plt.ylabel("mean absolute error")
        plt.legend(loc=2)
        plt.twinx(plt.gca())
        (self.true_values[self.time_index] + intercept).round().stack().value_counts().sort_index().plot(label="counts",style='k:',grid=False)
        plt.ylabel("counts");

        plt.legend(loc=1)
        
    def comparisonTable(self):
        results =[]
        for model_name, model_data in self.models_predictions.items() :
            preds = model_data[self.time_index]
            true = self.true_values[self.time_index]
            results.append({
                                'model_name':model_name,
                                'mse':self.mse(preds.values.flatten(),true.values.flatten()),
                                'mae':self.mae(preds.values.flatten(),true.values.flatten())
                           }
                          )
        return pd.DataFrame(results).set_index('model_name')
    
    
    def plotTimeError(self):
        for model_name, model_data in self.models_predictions.items() :
            preds = model_data[self.time_index]
            true = self.true_values[self.time_index]
            error = abs(preds - true).groupby(pd.to_datetime(self.time_index).time,axis=1).agg(np.mean).mean()
            plt.plot(error.values,self.markers[model_name],label=model_name)
            plt.xticks(range(len(error.index)),error.index,rotation=90)
        plt.ylabel("mean absolute error")
        plt.xlabel("time")

        plt.legend()

    def plotAveragePrediction(self,intercept =0,subplot=False):
        subplot_idx=1
        if not type(intercept) is int :
            intercept = intercept[self.time_index]
            if not self.segments_index is None :
                intercept = intercept.loc[self.segments_index].copy()
        
        true = self.true_values[self.time_index]+intercept
        if not subplot:
            plt.plot(true.mean().values, label='True values')
            xPos=  np.r_[:len(true.mean().index)-1:10j].astype(int)
            plt.xticks(xPos,labels=true.mean().index[xPos],rotation=90)
            
        for model_name, model_data in self.models_predictions.items() :
            if subplot :
                self.asSubPlots(subplot_idx)
                subplot_idx+=1
                plt.plot(true.mean().values, label='True values')
                xPos=  np.r_[:len(true.mean().index)-1:10j].astype(int)
                plt.xticks(xPos,labels=true.mean().index[xPos],rotation=90)
            preds = model_data[self.time_index]+intercept
            plt.plot(preds.mean().values,self.markers[model_name],label=model_name)
            plt.xlabel("time")
            plt.ylabel("speed")
            plt.legend()

        plt.tight_layout()
        plt.legend()

        
    def plotErrorHistogram(self,intercept =0,subplot=True,bins=40):
        subplot_idx=1
        if not type(intercept) is int :
            intercept = intercept[self.time_index]
            if not self.segments_index is None :
                intercept = intercept.loc[self.segments_index].copy()
        
        true = (self.true_values[self.time_index]+intercept).values.flatten()
        
        for model_name, model_data in self.models_predictions.items() :
            if subplot :
                self.asSubPlots(subplot_idx)
                subplot_idx+=1
            preds =(model_data[self.time_index]+intercept).values.flatten()
            plt.hist2d(preds,true,self.markers[model_name],label=model_name,bins=bins)
            plt.xlabel("error")
            plt.ylabel("true")
            plt.legend()
            plt.title(model_name)
        plt.tight_layout()
        plt.legend()
        
    def plotAverageError(self,smooth_sigma=1,subplot=False):
        subplot_idx=1
        
        true = self.true_values[self.time_index]
        
        for model_name, model_data in self.models_predictions.items() :
            if subplot :
                self.asSubPlots(subplot_idx)
                subplot_idx+=1
                plt.plot(true.mean().values-true.mean().values, label='Perfect fit')
                xPos=  np.r_[:len(true.mean().index)-1:10j].astype(int)
                plt.xticks(xPos,labels=true.mean().index[xPos],rotation=90)
            error = model_data[self.time_index]-true
            plt.plot(gaussian_filter1d(error.mean().values,sigma=smooth_sigma),self.markers[model_name],label=model_name)
            plt.xlabel("time")
            plt.ylabel("average error")
            plt.legend()

        plt.tight_layout()
        plt.legend()
        
    def plotSortedError(self,smooth_sigma=1,subplot=False):
        subplot_idx=1
        
        true = self.true_values[self.time_index].values.flatten()
        
        for model_name, model_data in self.models_predictions.items() :
            if subplot :
                self.asSubPlots(subplot_idx)
                subplot_idx+=1
                plt.axhline(0, label='True values')                
                
            preds = (model_data[self.time_index].values.flatten()-true)
            preds.sort()
            
            plt.plot(np.fromiter([np.mean(x) for x in np.split(preds,np.r_[:len(preds):100j].astype(int)[1:-1])],dtype=float),self.markers[model_name],label=model_name)
            plt.xlabel("time")
            plt.ylabel("error")
            plt.legend()

        plt.tight_layout()
        plt.legend()

        

    def cdfPlot(self,error_type ="mape",plot_lines=True):
        """
        cumulative distribution plot of the given *error_type* for the current model

        """
        true = self.true_values[self.time_index].values.flatten()

        for model_name, model_data in self.models_predictions.items() :
            preds = model_data[self.time_index].values.flatten()

            if error_type.lower() == "mape":
                error = (abs(true - preds)/abs(true)).flatten()
            if error_type.lower() =="mae":
                error = abs(true - preds).flatten()
            if error_type.lower() =="mse":
                error = ((true - preds)**2).flatten()

            error.sort()
            idx_error=np.cumsum(np.arange(len(error)))
            plt.plot(error,(idx_error/idx_error.max()).round(2),self.markers[model_name],label=model_name)
        if plot_lines :
            plt.axvline(0.5,c="red",label="0.5")
            plt.axvline(0.25,c="green",label="0.25")
            plt.axvline(0.15,c="pink",label="0.15")
        plt.ylabel("cumulative probability")
        plt.xlabel(error_type)
        plt.title("CDF")
        plt.legend()
        
    def qqPlot(self,intercept=0,smooth_sigma=1,subplot=False):
        subplot_idx=1
        
        if not type(intercept) is int :
            intercept = intercept[self.time_index]
            if not self.segments_index is None :
                intercept = intercept.loc[self.segments_index].copy()
        
        
        true = (self.true_values[self.time_index]+intercept).values.flatten()
        
        true_arg_sort = true.argsort()
        if not subplot:
            plt.plot(np.arange(true.min(),true.max()),np.arange(true.min(),true.max()),label="perfect fit")
        for model_name, model_data in self.models_predictions.items() :
            if subplot :
                self.asSubPlots(subplot_idx)
                subplot_idx+=1
                plt.plot(np.arange(true.min(),true.max()),np.arange(true.min(),true.max()),label="perfect fit")
            preds = (model_data[self.time_index]+intercept).values.flatten()
            
            plt.plot(true[true_arg_sort],gaussian_filter1d(preds[true_arg_sort],sigma=smooth_sigma),self.markers[model_name], label=model_name)
            
            plt.xlabel("True")
            plt.ylabel("preds")
            plt.legend()

        plt.tight_layout()
        plt.legend()
        

    def differencedErrorPlot(self,intercept= 0,subplot=False):
        subplot_idx=1
        if not type(intercept) is int :
            intercept = intercept[self.time_index]
            if not self.segments_index is None :
                intercept = intercept.loc[self.segments_index].copy()
        
        true = self.true_values[self.time_index]+intercept
        
        true_diff = true-true.shift(axis=1)   
        new_index=true_diff.columns[ pd.to_datetime(true_diff.columns).time != pd.to_datetime(self.time_index).time.min()]
        new_index = new_index[1:]
        true_diff=true_diff[new_index]
        
        for model_name, model_data in self.models_predictions.items() :
            if subplot :
                self.asSubPlots(subplot_idx)
                subplot_idx+=1
                plt.plot(true_diff.mean().values, label='True values')
                xPos=  np.r_[:len(true_diff.mean().index)-1:10j].astype(int)
                plt.xticks(xPos,labels=true_diff.mean().index[xPos],rotation=90)
                
            preds = model_data[self.time_index]+intercept-true.shift(axis=1) 
            preds=preds[new_index]
            plt.plot(preds.mean().values,self.markers[model_name],label=model_name)
            plt.xlabel("time")
            plt.ylabel("error")
            plt.legend()

        plt.tight_layout()
        plt.legend()

        
    def errorSegmentStdOrder(self,smooth_sigma=0,selection=None):
        
        if selection is None:
            selection = self.models_predictions.keys()

        true = self.true_values[self.time_index]
        std_sorted_idx = true.std(axis=1).sort_values().index

        for model_name in selection :
            preds = self.models_predictions[model_name][self.time_index]
            if smooth_sigma==0 :
                plt.plot(abs(preds-true).mean(axis=1).reindex(std_sorted_idx).values,label=model_name)
            else :
                plt.plot(gaussian_filter1d(abs(preds-true).mean(axis=1).reindex(std_sorted_idx).values,sigma=smooth_sigma),self.markers[model_name],label=model_name)
            plt.legend()
        plt.ylabel("mean absolute error")
        plt.xlabel("segments")
        smooth_label = "" if smooth_sigma==0 else "smoothed"
        plt.title("mean absoulute error for each segment (ordered by standard deviation)"+smooth_label)
        
    def futurError(self):
        for model_name, model_data in self.models_predictions.items() :
            preds = model_data[self.time_index]
            true = self.true_values[self.time_index]
            error = abs(preds - true).groupby(pd.to_datetime(self.time_index).date,axis=1).agg(np.mean).mean()
            plt.plot(error.values,self.markers[model_name],label=model_name)
            plt.xticks(range(len(error.index)),error.index,rotation=90)
        plt.ylabel("mean absolute error")
        plt.xlabel("dates")

        plt.legend()

    def asSubPlots(self,i,nb_plots=0):
        if nb_plots ==0 :
            nb_plots = len(self.models_predictions)
        nrows = np.sqrt(nb_plots)//1+1
        if i == 1:
            self.__ax=plt.subplot(nrows,nrows,i)
        else: 
            plt.subplot(nrows,nrows,i,sharey=self.__ax)
    def mse(self,x,y):
        return np.mean((x-y)**2)
    def mae(self,x,y):
        return np.mean(abs(x-y))

    def mseclip(x,y):
        return np.mean((x.clip(15)-y.clip(15))**2)
    def maeclip(self,x,y):
        return np.mean(abs(x.clip(15)-y.clip(15)))

    def mape(self,x,y):
        return np.mean(abs(x.clip(15)-y.clip(15))/y.clip(15))
    
    
    def boxenplotError(self):
        for model_name, model_data in self.models_predictions.items() :
            preds = model_data[self.time_index]
            true = self.true_values[self.time_index]
            error = abs(preds - true).groupby(pd.to_datetime(self.time_index).time,axis=1).mean()
            sns.boxenplot(data=error)
            plt.xticks(rotation=90)

#             plt.xticks(range(len(error.index)),error.index,rotation=90)
        plt.ylabel("mean absolute error")
        plt.xlabel("time")
        
        plt.legend()
        
        
    def boxenplotError(self,**kwdargs):
        errors=[]

        for model_name, model_data in self.models_predictions.items() :
            preds = model_data[self.time_index]
            true = self.true_values[self.time_index]
            error = (preds - true).stack()
            errors.append(error)
            

        sns.boxenplot(data=errors,**kwdargs)
        
        
        