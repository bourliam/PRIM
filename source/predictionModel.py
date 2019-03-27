
from functools import reduce
import CustomUtils
import Plotting
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import folium
import matplotlib

import time
import sys
sys.path.append('../source')


class PredictionModel:
    def __init__(self, data, input_lag, output_lag, sequence_length, scale_max=False, scale_log=False, shift_mean=False, y_only=False, add_time=False, max_value=130, valid_split=0.8, min_max_scale=False):
        self.data = data
        self.input_lag = input_lag
        self.output_lag = output_lag
        self.sequence_length = sequence_length
        self.scale_max = scale_max
        self.scale_log = scale_log
        self.shift_mean = shift_mean
        self.y_only = y_only
        self.add_time = add_time
        self.max_value = max_value
        self.min_max_scale = min_max_scale
        self.model = None
        self.valid_split = valid_split
        self.x, self.y, self.t = self.getXY()
        self.__reversed_process = []

    def getDaysTypes(self):
        day_types = pd.Series(
            self.t.reshape(-1)).dt.weekday.values.reshape(self.t.shape)
        time_fraction = (CustomUtils.timeToSeconds(pd.DatetimeIndex(
            self.t.reshape(-1)))/(60*60)).values.reshape(self.t.shape)
        time_input = np.concatenate([day_types, time_fraction], 1)
        return time_input[:int(len(self.x)*(self.valid_split))], time_input[int(len(self.x)*(self.valid_split)):]

    def getExamples(self, sequence, hours):
        sequence_length = len(sequence)
        sub_sequence_length = self.input_lag + self.output_lag
        if sub_sequence_length > sequence_length:
            raise ValueError("sequence length {} too small for lags : {}, {}".format(
                sequence_length, self.input_lag, self.output_lag))
        return [sequence[i:i + self.input_lag] for i in range(0, sequence_length-sub_sequence_length + 1, 1)], \
               [sequence[i + self.input_lag:i + self.input_lag + self.output_lag] for i in range(0, sequence_length - sub_sequence_length + 1, 1)], \
               [hours[i + self.input_lag:i + self.input_lag + self.output_lag]
                   for i in range(0, sequence_length - sub_sequence_length + 1, 1)]

    def getXY(self):
        nsegs, ntime = self.data.shape
        if(ntime % self.sequence_length) != 0:
            raise ValueError("sequence length {} not compatible with number of time features {}".format(
                self.sequence_length, ntime))

        shapedData = self.data.values.T.reshape(
            int(ntime/self.sequence_length), self.sequence_length, nsegs)
        timestamps = pd.Series(self.data.columns).values.reshape(
            int(ntime/self.sequence_length), self.sequence_length)

        examples = [self.getExamples(x, hours)
                    for x, hours in zip(shapedData, timestamps)]

        x, y, t = list(zip(*examples))
        return np.concatenate(x), np.concatenate(y), np.concatenate(t)

    def getIndexes(self, idx):
        cx, cy = (idx + (self.input_lag+self.output_lag-1)*(idx//(self.sequence_length - self.input_lag-self.output_lag+1)),
                  idx + (self.input_lag+self.output_lag-1)*(idx//(self.sequence_length - self.input_lag-self.output_lag+1))+self.input_lag)
        return (self.data.columns[cx:cy].values, self.data.columns[cy:cy+self.output_lag].values)

    def scaleMax(self):
        self.__reversed_process.append(self.reverseScaleMax)
        self.x /= self.max_value
        if not self.y_only:
            self.y /= self.max_value

    def scaleMinMax(self):
        self.__reversed_process.append(self.reverseMinMaxScale)
        self.min = self.x[:int(len(self.x)*(self.valid_split))].min()
        self.max = self.x[:int(len(self.x)*(self.valid_split))].max()
        diff = self.max - self.min
        self.x = (self.x-self.min)/diff
        self.y = (self.y-self.min)/diff

    def reverseMinMaxScale(self, x):
        return x*(self.max-self.min)+self.min

    def reverseScaleMax(self, y):
        return y*self.max_value

    def scaleLog(self):
        self.__reversed_process.append(self.reverseScaleLog)

        self.x = np.log1p(self.x)
        self.y = np.log1p(self.y)

    def reverseScaleLog(self, y):
        return np.expm1(y)

    def addTime(self):
        self.__reversed_process.append(self.removeTime)
        self.x = np.concatenate(
            (self.x, self.t.reshape(-1, self.t.shape[1], 1)), 2)

    def removeTime(self, y):
        if y.shape == self.x.shape:
            return np.delete(self.x, self.x.shape[2]-1, axis=2)
        return y

    def shiftMean(self):
        self.__reversed_process.append(self.resetMean)
        self.means = self.data[self.data.columns[:(
            int(len(self.data.columns)*self.valid_split))]].mean(axis=1).values
        if not self.y_only:
            self.x -= self.means
        self.y -= self.means

    def resetMean(self, y):
        return y+self.means

    def preprocessData(self):
        if self.shift_mean:
            self.shiftMean()
        if self.scale_max:
            self.scaleMax()
        if self.scale_log:
            self.scaleLog()
        if self.min_max_scale:
            self.scaleMinMax()
        if self.add_time:
            self.addTime()

        # if self.input_lag == 1:
        #     self.x = self.x.reshape(-1, self.x.shape[2])
        if self.output_lag == 1:
            self.y = self.y.reshape(-1, self.y.shape[2])

    def getRawYData(self, y):
        return reduce(lambda res, f: f(res), self.__reversed_process[::-1], y)

    def mse(self, p, y=None):
        pred = self.getRawYData(p)
        if y is not None:
            raw_y = self.getRawYData(y)
        else:
            raw_y = self.getRawYData(self.y)
        return np.mean((pred-raw_y)**2)

    def mae(self, p, y=None):
        pred = self.getRawYData(p)
        if y is not None:
            raw_y = self.getRawYData(y)
        else:
            raw_y = self.getRawYData(self.y)
        return np.mean(abs(pred-raw_y))

    def trainSplit(self):

        x_train = self.x[:int(len(self.x)*(self.valid_split))]
        x_test = self.x[int(len(self.x)*(self.valid_split)):]
        y_train = self.y[:int(len(self.x)*(self.valid_split))]
        y_test = self.y[int(len(self.x)*(self.valid_split)):]
        return x_train, y_train, x_test, y_test

    def getSplitSequences(self, values, sequence_length, skip=0):
        def addNans(values, sequence_length, skip):

            values = values.reshape(-1, sequence_length)
            nans = np.array([np.nan]*(values.shape[0]*(skip+1))
                            ).reshape(values.shape[0], -1)
            values = np.concatenate((values, nans), axis=1).reshape(-1)
            return values
        return addNans(np.arange(len(values)), sequence_length, skip), addNans(values, sequence_length, skip)

    def restorePredictionsAsDF(self, preds):

        index = [self.getIndexes(i)[1][0] for i in range(len(preds))]
        df = pd.DataFrame(self.getRawYData(preds),
                          index=index, columns=self.data.index)
        return df.T

    def restoreXAsDF(self, x):
        index = [self.getIndexes(i)[1][0] for i in range(len(x))]
        df = pd.DataFrame(self.getRawYData(x).swapaxes(
            1, 2).tolist(), index=index, columns=self.data.index)
        return df.T

    def predict(self, split="full"):
        secondary_input = self.getDaysTypes()
        if split.lower() == "full":
            main_input = self.x
            secondary_input = np.concatenate(secondary_input)
        if split.lower() == "train":
            main_input, *_ = self.trainSplit()
            secondary_input = secondary_input[0]
        if split.lower() == "test":
            *_, main_input, _ = self.trainSplit()
            secondary_input = secondary_input[1]

        if(len(self.model.input_shape) == 2):
            return self.model.predict(main_input)

        return self.model.predict([main_input, secondary_input])

    """ PLOTS """

    def plotPredHeatMap(self, cost=None, save=False):
        yDF = self.restorePredictionsAsDF(self.y)
        predDF = self.restorePredictionsAsDF(self.predict('full'))

        if cost:
            colors = cost(yDF, predDF)
        else:
            colors = ((np.abs(yDF - predDF)+1)/(yDF+1)).clip(upper=1)

        plt.figure(figsize=(18, 12))
        plt.imshow(colors, aspect='auto')
        plt.colorbar()
        if save:
            plt.savefig("Heatmap.png", bbox_inches='tight')
        plt.show()

    def plotPredictions(self, segments, mergedIndex, updatedcounts, folium_map=None):
        yDF = self.restorePredictionsAsDF(self.y)
        predDF = self.restorePredictionsAsDF(self.predict('full'))
        timesteps = yDF.columns[np.concatenate[:len(yDF.columns)-1:10j].astype(int)]

        if folium_map == None:
            folium_map = Plotting.getFoliumMap()
        layers = []
        colors = ((np.abs(yDF.clip(lower=15) - predDF.clip(lower=15))+1) /
                  (yDF.clip(lower=15)+1)).clip(upper=1)
        laggedX = self.restorePredictionsAsDF(self.x)
        predSegs = segments[segments.segmentID.isin(
            mergedIndex[mergedIndex.isin(yDF.index)].index)]
        segment_overall_mean = [self.data.mean(
            axis=1).loc[mergedIndex.loc[idx]] for idx in predSegs.segmentID]
        segment_timestamp_mean = self.data.groupby(
            pd.Series(self.data.columns).dt.time, axis=1).mean()
        for t in timesteps:
            colorList = [colors[t].loc[mergedIndex.loc[idx]]
                         for idx in predSegs.segmentID]
            y = [yDF[t].loc[mergedIndex.loc[idx]]
                 for idx in predSegs.segmentID]
            preds = [predDF[t].loc[mergedIndex.loc[idx]]
                     for idx in predSegs.segmentID]
            segCounts = [updatedcounts[t].loc[mergedIndex.loc[idx]]
                         for idx in predSegs.segmentID]
            timestampLaggedX = [laggedX[t].loc[mergedIndex.loc[idx]]
                                for idx in predSegs.segmentID]

            current_segment_timestamp_mean = [segment_timestamp_mean[t.time(
            )].loc[mergedIndex.loc[idx]] for idx in predSegs.segmentID]

            popups = ["segment : {:}, <br> y : {:.2f}, <br> pred : {:.2f}, <br> %error : {:.0f}%, <br> count : {:}<br>mean: {:}<br>timestamp_mean: {:}<br>x: {:} "
                      .format(seg, yi, predi, props*100, count, mean, timestamp_mean, np.array(x).astype(int))
                      for seg, yi, predi, props, count, mean, timestamp_mean, x
                      in zip(predSegs.segmentID, y, preds, colorList, segCounts, segment_overall_mean, current_segment_timestamp_mean, timestampLaggedX)]

          #  pos = yDF.columns.get_loc(t)
            layer = self.getPredictionLayer(
                predSegs, colorList, folium_map, str(t), popups)
            layers.append(layer)

        return Plotting.stackHistotyLayers([*layers, folium.TileLayer()], folium_map)

    def getPredictionLayer(self, segments, colors, folium_map, name='layer', popups=[]):
        layer = folium.plugins.FeatureGroupSubGroup(
            folium_map, name=name, show=False, overlay=False)
        [folium.PolyLine(locations=[lo[::-1] for lo in x['coordinates']], color=matplotlib.colors.rgb2hex(
            plt.cm.brg_r(color/2)), popup=pop).add_to(layer) for x, color, pop in zip(segments['loc'], colors, popups)]
        return layer
