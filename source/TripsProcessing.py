import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
def buildOdMatrix(trips,illeEtVilaineIRIS,maxSpeed=0,minDuration=5):
    """ 
    create OD matrix from trips
    trips : the reconstructed trips
    illeEtVilaineIRIS :  the iris of ille et vilaine
    maxSpeed : the maximum speed at starting and ending points (if a different value then the one used while filtering trips is desired)
    minDuration : the minimum duration of trip in minutes (if a different value then the one used while filtering trips is desired)
    """
    beginDF=pd.DataFrame([*trips[trips.dur>=np.timedelta64(minDuration,'m')].begin.values])
    endDF=pd.DataFrame([*trips[trips.dur>=np.timedelta64(minDuration,'m')].end.values])
    tripEdgesDF=beginDF.join(endDF,lsuffix='_begin',rsuffix='_end')
    tripEdgesDF=tripEdgesDF[(tripEdgesDF.speed_begin<=maxSpeed) & (tripEdgesDF.speed_end<=maxSpeed)]
    irisMatrix=pd.DataFrame(index=illeEtVilaineIRIS.INSEE_iris_code.values,columns=illeEtVilaineIRIS.INSEE_iris_code.values)
    irisMatrix.update(tripEdgesDF[(tripEdgesDF.INSEE_iris_code_begin.isin(illeEtVilaineIRIS.INSEE_iris_code))&(tripEdgesDF.INSEE_iris_code_end.isin(illeEtVilaineIRIS.INSEE_iris_code))].groupby(by=['INSEE_iris_code_begin','INSEE_iris_code_end']).size().unstack())
    return irisMatrix.fillna(0)


def buildTrees(trips):
    """ 
    (Ongoing work)
    build a tree of distances between trips starting and ending points
    trips : data frame of trips
    """
    startLocs=trips['loc'].apply(lambda x:x[0]['coordinates'])
    endLocs=trips['loc'].apply(lambda x:x[len(x)-1]['coordinates'])
    startData=np.array([*startLocs.values])
    endData=np.array([*endLocs.values])
    startTree=KDTree(startData)
    endTree=KDTree(endData)
    return startTree,endTree,startData,endData

def irisFlowRate(OdMatrix):
    """ 
    Flow between iris
    OdMatrix : OD matrix of iris
    """
    return OdMatrix.sum(axis=1)+OdMatrix.sum(axis=0)-OdMatrix.values[tuple([np.arange(len(OdMatrix))])*2]


def getGroupedTripEdges(tripsDF):
    edgesDF = tripsDF.groupby('id').apply(lambda x :pd.Series({"edges": [coorX['loc']['coordinates'] for coorX in x.begin]+[coorX['loc']['coordinates'] for coorX in x.end]}))
    edgesDF=edgesDF.assign(length=edgesDF.edges.apply(len))
    return edgesDF

def filterEdgesLength(edgesDF, minPoints=0):
    return edgesDF[edgesDF>=minPoints]

def findUserRegionsOfInterst(userEdges,metric,min_samples=5,leaf_size=2):
    res = DBSCAN(min_samples=min_samples,metric=metric,leaf_size=leaf_size).fit(userEdges.edges)
    nbClusts=len(set(res.labels_))
    return nbClusts, res.labels_