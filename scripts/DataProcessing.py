import pandas as pd
import numpy as np
from pymongo import MongoClient
try:
    client
except NameError:
    client = MongoClient("mongodb://mbouchouia:cbf20Li34!@mongodb-tp.enst.fr")
    osmData = client.geolytics.ways

def correctIRIS(point,maxDistance):
    """
    Affect point to nearest IRIS
    
    point : Geojason point
    
    maxDistance : float
        max distance between point and IRIS (meters)
    
    returns closest IRIS or 'N/A'
    """
    res=db.iris_geo_coords.find(
    {   
        "loc":
        {
         "$near": {
            "$geometry": point,
            "$maxDistance": maxDistance,

         }
       }
    },{'INSEE_iris_code':1,'_id':0}
    ).limit(1)
    return next(res,{'INSEE_iris_code':'N/A'})['INSEE_iris_code']


def builIrisDataFrame(irisCollection):
    """
    returns all ille et vilaine IRISs

    irisCollection : Mongo collection 
        IRIS collection    
    """
    illeEtVilaineIRIS=irisCollection.find({'code_dept':'35'})
    illeEtVilaineIRIS=pd.DataFrame(list(illeEtVilaineIRIS))
    illeEtVilaineIRIS['loc'].apply(lambda x : [y.reverse() if x['type']== 'Polygon' else  [z.reverse() for z in y] for f in  x['coordinates'] for y in f ])
    return illeEtVilaineIRIS

def loadProjectedRawData(coyoteData,limit=None,projection={}):
    """
    coyoteData : Mongo collection 
        coyote collection of logs
        
    limit : int
        the number of records to return
        
    projection : dict
        the projection of features
        
    returns pandas dataframe of th requested collection
    """
    if limit :
        df = pd.DataFrame(list(coyoteData.find({},projection).limit(limit)))
    else : df =  pd.DataFrame(list(coyoteData.find({},projection)))
        
    df.sort_values(by='time',inplace=True)
    if(type(df.time.values[0]) == np.int64):
        transformedTime =  pd.to_datetime(df.time,unit='s')+np.timedelta64(1,'h')
        df=df.assign(time=transformedTime)
    return df

def loadRawData(coyoteData,limit=None):
    """
    coyoteData : Mongo collection 
        coyote collection of logs
        
    limit : int
        the number of records to return
        
    returns pandas dataframe of th requested collection
    """
    if limit :
        df = pd.DataFrame(list(coyoteData.find({}).limit(limit)))
    else : df =  pd.DataFrame(list(coyoteData.find({})))
        
    df.sort_values(by='time',inplace=True)
    if(type(df.time.values[0]) == np.int64):
        transformedTime =  pd.to_datetime(df.time,unit='s')+np.timedelta64(1,'h')
        df=df.assign(time=transformedTime)
    return df

def isResidential(location):
    """
    check if a pont is in residential area
    """
    cur=osmData.find({'tag.k':'landuse',
                      'tag.v':{'$in':["commercial",
                                      "construction",
                                      "industrial",	
                                      "residential",	
                                      "retail"
                                     ]
                              }, 
                        "loc":
                            {
                                 "$near": {
                                    "$geometry": {'type':'point',
                                                  'coordinates':location},
                                 }
                            }
                     }
                    ,{'tag.v':1,'_id':0}
                    ).limit(1)
    return "residential" in [landuse['v'] for landuse in list(cur)[0]['tag']]