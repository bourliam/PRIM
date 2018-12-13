import pandas as pd
import time 
def getSegments(    osmWays,innerBox=[[[-1.5460, 48.1656], [-1.5460, 48.0632], [-1.7626, 48.0632], [-1.7626,48.1656], [-1.5460, 48.1656]]],
                    innerTags =["motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link"],
                    outerBox=[[[-1.4460, 48.2056], [-1.4460, 48.0032], [-1.8626, 48.0032], [-1.8626,48.2056], [-1.4460, 48.2056]]],
                    outerTags= ["motorway", "trunk"]
               ):
    
    """ 
    filter segments by tags in the requested bounding box
    
    osmWays : mongo collection of OSM ways
    innerBox : an inner rectangle where to fetch segments with tags @innerTags
    innerTags : tags filter for segments in the inner box
    outerBox : an inner rectangle where to fetch segments with tags @outerTags
    outerTags : tags filter for segments in the outer box    
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

def buildSegmentHierarchy(segments):
    """ 
    find all incoming/outgoins segments for each segment
    segments: pandas dataframe of segmenst (OSM way)
    """
    ins =segments.nodes.apply(lambda x : segments.index[ segments.nodes.apply(lambda y : ((y[len(y)-1] in x) or (x[0] in y)) and x!=y )].values)
    outs=segments.nodes.apply(lambda x : segments.index[ segments.nodes.apply(lambda y : ((x[len(x)-1] in y) or (y[0] in x)) and x!=y )].values)
    return segments.assign(ins=ins,outs=outs)