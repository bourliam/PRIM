import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
import CustomUtils
from Trips import tripDistance
def buildOdMatrix(trips,illeEtVilaineIRIS,maxSpeed=0,minDuration=5):
    """ 
    create OD matrix from trips
    
    trips : pandas dataFrame 
        the reconstructed trips
        
    illeEtVilaineIRIS : pandas dataFrame 
        the iris of ille et vilaine
        
    maxSpeed : float
        the maximum speed at starting and ending points (if a different value then the one used while filtering trips is desired)
        
    minDuration : int 
        the minimum duration of trip in minutes (if a different value then the one used while filtering trips is desired)
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
    
    trips : pandas dataFrame
        data frame of trips
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
    
    OdMatrix : pandas dataFrame 
        OD matrix of iris
    """
    return OdMatrix.sum(axis=1)+OdMatrix.sum(axis=0)-OdMatrix.values[tuple([np.arange(len(OdMatrix))])*2]


def getGroupedTripEdges(tripsDF):
    """
    group start and end points of trips for each user
    
    tripsDF : pandas dataFrame 
        dataFrame of trips
    """
    edgesDF = tripsDF.groupby('id').apply(lambda x :pd.Series({"edges_begin": np.array([coorX['loc']['coordinates'] for coorX in x.begin]),"edges_end":np.array([coorX['loc']['coordinates'] for coorX in x.end]),"edges_trip_id":x.index.get_values()}))
    edgesDF=edgesDF.assign(length=edgesDF.edges_begin.apply(len)*2)
    return edgesDF

def filterEdgesLength(edgesDF, minPoints=0):
    """
    filter users with low trip rate
    
    edgesDF : pandas dataFrame
        dataFrame of edges
        
    minPoints : int
        minimum number of start/end points (2*number of trips) 
    """
    return edgesDF[edgesDF.length>=minPoints]

def findUserRegionsOfInterst(userEdges, metric=CustomUtils.reverseVincenty, min_samples=5,eps=0.5):
    """
    find recurent regions for the user
    
    userEdges = pd.Series
        user start/End positions
        
    metric : string or callable, optional
        The metric to use when calculating distance between instances 
        
    min_samples : int, optional
        The number of samples in a neighborhood
        
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    """
    res = DBSCAN(metric=metric,min_samples=min_samples,eps=eps).fit(userEdges)
    nbClusts=len(set(res.labels_))-1
    return nbClusts, res.labels_

def findAllUsersROI(roiDF,metric=CustomUtils.reverseVincenty,min_samples=5,eps=0.5):
    """
    returns data frame of clustering results on each user using @findUserRegionsOfInterst
    
    roiDF : pandas DataFrame
        grouped start and end points of all users #refer to getGroupedTripEdges
        
    min_samples : int, optional
        The number of samples in a neighborhood    
        
    metric : string or callable, optional
        The metric to use when calculating distance between instances 
        
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    """
    
    return roiDF.apply(lambda userEdges:  pd.Series([
                                                                *findUserRegionsOfInterst(userEdges.edges_begin, min_samples=min_samples,eps=eps,metric=metric),
                                                                *findUserRegionsOfInterst(userEdges.edges_end, min_samples=min_samples,eps=eps,metric=metric)
                                                            ],index=['n_clusters_begin', 'clusters_begin','n_clusters_end', 'clusters_end']), 
                              axis=1)

def getEndClusterStats(trips,userEdges):
    """
    returns structured dataframe 
    
    trips : pandas dataFrame
        data frame of trips
        
    userEdges = pd.Series
        user start/End positions
        
    clustersEnd : array of int
        clustering results
    """
    endStats = trips.loc[userEdges.edges_trip_id].apply(lambda x :pd.Series({"time":x.time[len(x.time)-1],"INSEE_iris_code":x.INSEE_iris_code[len(x.INSEE_iris_code)-1]}),axis=1)
    return endStats.assign(cluster=userEdges.clusters_end,deltaTime=CustomUtils.timeToTimeDelta(endStats.time))

def getBeginClusterStats(trips, userEdges):
    """
    return structured dataframe 
    
    trips : pandas dataFrame
        data frame of trips
        
    userEdges : pandas dataFrame
        dataFrame of user edges
        
    clustersBegin : array of int
        clustering results
    """
    
    beginStats=trips.loc[userEdges.edges_trip_id].apply(lambda x :pd.Series({"time":x.time[0],"INSEE_iris_code":x.INSEE_iris_code[0]}),axis=1)
    return beginStats.assign(cluster=userEdges.clusters_begin,deltaTime=CustomUtils.timeToTimeDelta(beginStats.time))

def recurentTrips(userEdges,metric=tripDistance,min_samples=5,eps=0.5):
    """
    find recurent trips of the user
       
    userEdges = pd.Series
        user start/End positions
        
    min_samples : int
        The number of samples in a neighborhood
        
    metric : string, or callable
        The metric to use when calculating distance between instances
        
    eps : float, optional
        The maximum distance between two samples for them to be considered
        as in the same neighborhood.
    """
    tripsData = [[*x,*y]  for x , y in zip(userEdges.edges_begin,userEdges.edges_end)]
    res = DBSCAN(metric=metric,min_samples=min_samples,eps=eps).fit(tripsData)
    clusters = res.labels_
    return clusters