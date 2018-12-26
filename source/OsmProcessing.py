import pandas as pd
import numpy as np
import time 
from DataProcessing import reverseVincenty
def getSegments(    osmWays,innerBox=[[[-1.5460, 48.1656], [-1.5460, 48.0632], [-1.7626, 48.0632], [-1.7626,48.1656], [-1.5460, 48.1656]]],
                    innerTags =["motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"],
                    outerBox=[[[-1.4460, 48.2056], [-1.4460, 48.0032], [-1.8626, 48.0032], [-1.8626,48.2056], [-1.4460, 48.2056]]],
                    outerTags= ["motorway", "trunk"]
               ):
    
    """ 
    filter segments by tags in the requested bounding box
    
    osmWays : mongo collection
        OSM ways
        
    innerBox : array of  float length 5 
        an inner rectangle where to fetch segments with tags @innerTags
        
    innerTags : list of str 
        tags filter for segments in the inner box
        
    outerBox : array of float length 5
        an inner rectangle where to fetch segments with tags @outerTags
        
    outerTags : list of str
        tags filter for segments in the outer box    
    """
    
    cur = osmWays.find(    
        {
            "loc"   : {"$geoIntersects": {"$geometry": {"type": "Polygon" ,"coordinates":innerBox }}},    
            "tag.k" : "highway",
            "tag.k":{'$nin':['proposed']},
            "tag.v" : {
                "$in" : innerTags
            }
        }
    )
    
    innerRocadeDF=pd.DataFrame(list(cur))
    
    cur = osmWays.find(    
        {
            "loc"   : {"$geoIntersects": {"$geometry": {"type": "Polygon" ,"coordinates": outerBox}}},    
            "tag.k" : "highway",
            "tag.k":{'$nin':['proposed']},        
            "tag.v" : {
              "$in" : outerTags
            }
        }
    )

    outerRocadeDF=pd.DataFrame(list(cur))
    segments=pd.concat([innerRocadeDF,outerRocadeDF])
    segments.drop_duplicates('_id',inplace=True)
    segments.reset_index(drop=True,inplace=True)
    segments.tag=segments.tag.apply(lambda x : dict([(v['k'],v['v']) for v in x]))
    return segments.set_index('_id')


def closestSegment(myPoint,maxDistance,osm):
    """
    (deprecated)
    find closest segment to myPoint
    """
    res=osm.find(
    {   
        "loc":
        {
         "$near": {
            "$geometry": myPoint,
            "$maxDistance": maxDistance,
         }
       }
    },{'loc':1,'_id':1}
    ).limit(1)
    return next(res,{'_id':'N/A'})['_id']

def parallelClosestSegment(myPoint):
    """
    (deprecated)
    find closest segment to myPoint (parallel)
    """
    res=client.osm.ways.find(
    {   
        "loc":
        {
         "$near": {
            "$geometry": myPoint['loc'],
            "$maxDistance": 20,
         }
       }
    },{'loc':1,'_id':1}
    ).limit(1)
    return next(res,{'_id':'N/A'})['_id']

def buildSegmentsMeta(segments, points):
    """ 
    find all incoming/outgoins segments for each segment
    compute the length of each segment
    assign Max speed for each segment
    compute Point count and car count
    
    segments: pandas dataframe
        Osm ways
        
    points : pandas dataframe
        logs
    """
    noOneWays=segments[segments.oneWay!=1].copy()
    noOneWays.nodes=noOneWays.nodes.apply(lambda x : x[::-1])
    segs=pd.concat([segments,noOneWays])
    ins = segs.nodes.apply(lambda x : segs.index[segs.nodes.apply(lambda y : ((y[len(y)-1] in x[:-1]) or (x[0] in y[1:])) and x!=y and x!=y[::-1])].values)
    outs = segs.nodes.apply(lambda x : segs.index[segs.nodes.apply(lambda y : ((x[len(x)-1] in y[:-1]) or (y[0] in x[1:])) and x!=y and x!=y[::-1])].values)
    ins = ins.groupby('_id').apply(lambda x : np.unique(np.concatenate([*x])))
    outs = outs.groupby('_id').apply(lambda x : np.unique(np.concatenate([*x])))
    
    segments=segments.assign(maxSpeed=segments.tag.apply(lambda x : x['maxspeed'] if 'maxspeed'in x.keys() else '' ))
    length = segments['loc'].apply(lambda x : sum([reverseVincenty(a,b) for a, b in zip(x['coordinates'][:-1],x['coordinates'][1:])]))
    pointCounts=points.groupby(['matching_road']).size()
    carCounts = points.groupby(['matching_road','id']).size().groupby(['matching_road']).size()
    return segments.assign(ins=ins, outs=outs, length = length,pointCounts=pointCounts,carCounts=carCounts)

def setOneWay(segments):
    """
    extract one way tag from Osm data
    
    segments: pandas dataframe
        Osm ways
    """
    return segments.assign(oneWay=segments.tag.apply(lambda x :
                                                                  1 if ('oneway' in x and (x['oneway']=='yes')) 
                                                                      or(x['highway']=='motorway') 
                                                                      or('junction' in x and(x['junction'] in ['circular','roundabout'] ))
                                                            else -1 if  ('oneway' in x and x['oneway']=='-1' ) 
                                                            else  0))