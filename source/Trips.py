import time

# Data and processing
import datetime
import numpy as np
import pandas as pd
# Printing
from CustomUtils import reverseVincenty

def timeSpentSince(startTime):
    """ 
    returns the number of seconds since startTime
    
    startTime  : int
        timestamp
    """
    endTime = time.time()
    print('took {:.3f} ms'.format((endTime-startTime)*1000.0))

def temporelTripsFilter(st,thresh= 15):
    """
    return the index of records where the difference between the two successive values is higher than thresh(default 15mins)
    
    st : pandas dataFrame 
        dataFrame of logs
        
    thresh : int
        the maximum numbre of minutes before truncating the trip (startin a new one)
    """
    
    time=np.array(st['time'])
    timeDiff=time[1:]-time[:-1]
    return  np.where(timeDiff >np.timedelta64(thresh,'m'))[0]+1 

def stopsFilter(st, tempFilter, thresh= 10):
    """ 
    filter stop periods 
    
    st : pandas dataFrame
        data frame of logs
        
    tempFilter : array 
        the indexes return by (temporelTripsFilter)
        
    thresh : int
        the duration of stop allowed before starting a new trip (in minutes)
    """
    speed=np.array(st['speed'])
    time=np.array(st['time'])
    sepIdx= np.empty(0, int)
    for segS,segE in zip(np.append(0,tempFilter),np.append(tempFilter,len(st['loc']))):
        seg=np.arange(segS,segE)
        values=speed[seg]>0
        values=values[1:]==values[:-1]
        idx=np.where(~values)[0]+1
        if(len(idx)==0) : continue
        bounds = [np.append(0,idx),np.append(idx,len(seg)-1)]
        rng= np.arange(speed[segS]!=0,len(bounds[0]),2)
        timeDiff=time[seg[bounds[1][rng]]]-time[seg[bounds[0][rng]]]
        idx = np.where(timeDiff >np.timedelta64(thresh,'m'))[0]
        sepIdx=np.append(sepIdx,seg[bounds[1][rng][idx]])
    shifts = (time[sepIdx] - time[sepIdx-1]) < np.timedelta64(int(thresh/2),'m')
    return np.sort(np.append(tempFilter,sepIdx-shifts))

def segmentsOn(locs, idx):
    """ 
    create trips out  of points (one user)
    
    locs : pandas dataFrame
        dataframe of logs
        
    idx : array 
        the starting positions of new trips
    """
    return [[locs[cl][i:j] for cl in locs.index] for i,j in zip(np.append(0,idx),np.append(idx,len(locs['loc'])))]

def buildTrips(df,adhocCoef= 1.24, minPointsDuration= 15,stopsDuration= 10):
    """
    Create trips out logs for each user and compute multiple features for each trip
    
    df : pandas dataFrame
        dataframe of logs
        
    adhocCoef : float 
        coeficient to normalize distance
        
    minPointsDuration : int
        the duration between two points after which we start a new trip
        
    stopsDuration : int
        the duration of stop after which we start a new trip
    """ 
    startTime = time.time()

    print('grouping data points by car : ....\n')
    carsDF=df.groupby(by='id').agg(lambda x:[*x.values])
    print('grouping data points by car : Done\n')
    timeSpentSince(startTime)
    startTime = time.time()
    print('extracting trips : ....\n')
    trips =carsDF.apply(lambda x : list(zip(*segmentsOn(x,stopsFilter(x,temporelTripsFilter(x,thresh=minPointsDuration),thresh=stopsDuration)))),axis=1)
    print('extracting trips : Done\n')
    timeSpentSince(startTime)
    startTime = time.time()
    print('splitting trips by car : ....\n')
    trips=pd.DataFrame([[trips.index[idx],*row] for idx in range(len(trips)) for row in zip(*trips.iloc[idx])],columns=['id',*carsDF.columns])
    print('splitting trips by car : Done \n')
    timeSpentSince(startTime)
    #return trips
    startTime = time.time()
    print('adding columns (day,begin point,end point, duration : ....\n')
    trips=trips.assign(day=trips.time.apply(lambda x:pd.to_datetime(x[0]).date()))
    trips=trips.assign(begin=trips.apply(lambda x : dict([(c,x[c][0]) for c in ['loc','time','heading','speed','INSEE_iris_code']]),axis=1))
    trips=trips.assign(end=trips.apply(lambda x : dict([(c,x[c][len(x[c])-1]) for c in ['loc','time','heading','speed','INSEE_iris_code']]),axis=1))
    trips=trips.assign(dur=trips.time.apply(lambda x :x[len(x)-1]-x[0] ))
    trips=trips.assign(time_difference_seconds = trips.time.apply(lambda loc : np.array([(y-x)/np.timedelta64(1, 's') for x,y in zip(loc[:-1],loc[1:])])))
    print('adding columns (day,begin point,end point, duration : Done \n')
    timeSpentSince(startTime)
    startTime = time.time()
    print('Calculating distances: ... \n')
    trips=trips.assign(pairs_distances_km=trips['loc'].apply(lambda loc : np.array([reverseVincenty(x['coordinates'],y['coordinates'])*adhocCoef for x,y in zip(loc[:-1],loc[1:])])))
    trips=trips.assign(trip_distance_km=trips.pairs_distances_km.apply(sum))
    trips=trips.assign(trip_distance_km_vo = trips['loc'].apply(lambda x: reverseVincenty(x[0]['coordinates'],x[len(x)-1]['coordinates'])))
    print('Calculating distances: Done \n')
    timeSpentSince(startTime)
    startTime = time.time()
    return trips

#Filtering trips

def filterTrips(trips,cnd='ALL',irisFilter=[],maxOverallSpeed=0.2,minDuration=60,minDistance=0.2,maxSpeed=0,maxJumpSpeed=0.05,minOverallCoyoteSpeed=0 )                              :
    """ 
    apply multiple filters on trips
    
    cnd : the filters to apply (not used #TODO)
    
    irisFilter : array 
        the iris labels to take into consideration
        
    maxOverallSpeed : float
        the maximum speed overall trip km/second
        
    minDuration : int
        the minimum duration of a trip (in seconds)
        
    minDistance : float
        the minimum distance covered in the trip (in km)
        
    maxSpeed : float
        maximum speed at starting and ending points
        
    maxJumpSpeed : float
        maximum speed between the first two observed points (km/second)
        
    """
    # Max overall speed (km/second)
    tripOverallSpeed = trips.trip_distance_km/trips.dur.apply(lambda x : x/np.timedelta64(1, 's'))
    globalSpeedFilter = ((tripOverallSpeed < maxOverallSpeed))
    trips=trips[globalSpeedFilter]
    # Min Overall Coyote (observed) speed (km/second)
    coyoteSpeedFilter = trips.speed.apply(np.mean)>minOverallCoyoteSpeed
    trips=trips[coyoteSpeedFilter]

    # minimumTripDistance (km)
    minimumTripDistance = (trips.trip_distance_km>=minDistance)
    trips=trips[minimumTripDistance]
    # minimum trip duration (seconds)
    minTripDuration=(trips.dur>=np.timedelta64(minDuration, 's'))
    trips=trips[minTripDuration]
    # maximum Start point speed (default 0)
    maxStartSpeed = (trips.begin.apply(lambda x : x['speed']<=maxSpeed))
    trips=trips[maxStartSpeed]
    # maximum end point speed
    maxEndSpeed = (trips.end.apply(lambda x : x['speed']<=maxSpeed))
    trips=trips[maxEndSpeed]
    # geographique jump on start
    geoJump=(trips.pairs_distances_km.apply(lambda x : x[0])/trips.time_difference_seconds.apply(lambda x : x[0]))<=maxJumpSpeed
    trips=trips[geoJump]
    # Trips passing by  ile et vilaine IRIS
    inIris = (trips.INSEE_iris_code.apply(lambda x : any(iris in x for iris in irisFilter)))
    trips=trips[inIris]
    return trips

def tripDistance(x,y):
    """
    returns the distance between two trips as mean distance between thier starting points and ending points
    x,y : array of float (length 4)
        starting and ending point long lat  
    """
    #print(x,y)
    return max(reverseVincenty(x[:2],y[:2]),reverseVincenty(x[-2:],y[-2:]))