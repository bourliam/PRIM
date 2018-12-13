import pandas as pd
from vincenty import vincenty

def correctIRIS(point,maxDistance):
    """
    Affect point to nearest IRIS
    point : Geojason point
    maxDistance : max distance between point and IRIS (meters)
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

def reverseVincenty(a,b):
    """ 
    Vincenty distance
    a,b : array on length 2 (longitude, latitude)
    returns the distance between the two points
    """
    a=a.copy()
    a.reverse()
    b=b.copy()
    b.reverse()
    return vincenty(a,b)

def builIrisDataFrame(irisCollection):
    """ 
    irisCollection : Mongo collection of IRIS
    returns all ille et vilaine IRISs
    """
    illeEtVilaineIRIS=irisCollection.find({'code_dept':'35'})
    illeEtVilaineIRIS=pd.DataFrame(list(illeEtVilaineIRIS))
    illeEtVilaineIRIS['loc'].apply(lambda x : [y.reverse() if x['type']== 'Polygon' else  [z.reverse() for z in y] for f in  x['coordinates'] for y in f ])
    return illeEtVilaineIRIS

def loadRawData(coyoteData,limit=None):
    """
    coyoteData : Mongo collection of logs
    limit : the number of records to return
    returns pandas dataframe of th requested collection
    """
    if limit :
        df = pd.DataFrame(list(coyoteData.find({}).limit(limit)))
    else : df =  pd.DataFrame(list(coyoteData.find({})))
    df.sort_values(by='time',inplace=True)
    return df