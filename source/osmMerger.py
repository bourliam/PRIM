from mongoConnection import *
import OsmProcessing 
import Plotting
import numpy as np
import pandas as pd
from speed_matrix import SpeedMatrix
from affect_road_to_point_para import get_north_azimut
from CustomUtils import getTimeSpent
WEIGHTS=np.array([1,1,1,1]).reshape(-1,1)
MIN_VALID_DATA=0.8
fmap=Plotting.getFoliumMap()

def profileSim(segment,neighbours,speeds):
    """
    compute similarity between the segment and its neighbours
    """
    segValid = speeds.loc[segment].dropna()
    return np.fromiter(map(lambda x : segPDist(segment,x,speeds),neighbours),np.float)

def segPDist(seg1, seg2, speeds):
    """
    Compute MSE between the normalized speeds of the two segments when there are values + the inverse of the number of NA values over the number of possible values
    """
    
    seg1Valid=speeds.loc[seg1].dropna()
    seg2Valid=speeds[seg1Valid.index].loc[seg2].dropna()/sum(speeds[seg1Valid.index].loc[seg2].dropna(),1)
    seg1Valid=seg1Valid[seg2Valid.index]/sum(seg1Valid[seg2Valid.index],1)
    if len(seg1Valid.values) == 0 or len(seg2Valid.values) == 0 : return 1 + (1-len(seg2Valid.index)/ speeds.columns.size)
    return np.mean((seg1Valid.values - seg2Valid.values)**2) + (1-len(seg2Valid.index)/ speeds.columns.size)

def getCosRateBetweenSegments(seg1, seg2):
    """
    the difference in cosine between the two segments
    """
    return 1-(np.cos(np.deg2rad(get_north_azimut(seg1)-get_north_azimut(seg2)))/2+0.5)

def directtion(segment,neighbours,segmentsMeta):
    """
    Compute the difference between the segement and its neighbours in terms of cosine and sine
    """
    headTail= list(map(lambda x : 'Head' if x in segmentsMeta.at[segment,'outs'] else 'Tail',neighbours ))
    cosSinDiff = np.fromiter(map(lambda x,y :np.exp(abs(segmentsMeta['cos'+y][x]-segmentsMeta['cos'+y][segment])+abs(segmentsMeta['sin'+y][x]-segmentsMeta['sin'+y][segment]))-1 ,neighbours,headTail),np.float)
    #return  np.fromiter(map(lambda x : 1/(1+np.exp(6*x)) ,cosSinDiff),np.float)
    oneCos = cosSinDiff[0] if  cosSinDiff[0] <= np.exp(1) else np.inf
    return [oneCos]

def shareNoEdges(segment,neighbours,segmentsMeta):
    """
    return wether the segment and its neghbours doesn't share an edge
    """
    return np.fromiter(map(lambda x : haveNoSameEdges(segment,x,segmentsMeta),neighbours),np.bool)

def haveNoSameEdges(seg1,seg2,segmentsMeta):
    """
    return true if the two segments doesn't share any edges
    """
    seg1Edges = segmentsMeta['edges'][seg1]
    seg2Edges = segmentsMeta['edges'][seg2]
    return not any(a==b for a in seg1Edges for b in seg2Edges)

def pickBestCandidateToMergeWith(segmentToMerge,neighbours,mergedSegments,updatedSpeed,weights):
    """
    return the neigbour with the smallest criterion score (direction , temporel similarity, edge sharing)
    """
    criteronScores = [
        profileSim(segmentToMerge,neighbours,updatedSpeed),
        directtion(segmentToMerge,neighbours,mergedSegments),
        shareNoEdges(segmentToMerge,neighbours,mergedSegments)
    ]
    return neighbours[sum(criteronScores*weights).argmin()]

def computeCriteria(seg1,seg2,mergedSegments,weights):
    """
    return the weighted sum of criteria  between segments
    """
    criteronScores = [
        profileSim(seg1,[seg2],updatedSpeed),
        directtion(seg1,[seg2],mergedSegments),
        shareNoEdges(seg1,[seg2],mergedSegments)
        ]
    return sum(criteronScores*weights)



def getNeighbours(seg,meta,inversedIndex):
    """
    return the list of neighbours of the segments
    """
    return np.unique(np.fromiter( (inversedIndex[x] for x in np.concatenate([meta.loc[seg]['ins'],meta.loc[seg]['outs']])),dtype=np.int))

def computeCriteria(segment,mergedSegments,updatedSpeed,weights,inversedIndex):
    """
    return the weighted sum of criteria  between segments
    """
    neighbours = getNeighbours(segment,mergedSegments,inversedIndex)
    criteronScores = [
        profileSim(segment,neighbours,updatedSpeed),
        directtion(segment,neighbours,mergedSegments),
        shareNoEdges(segment,neighbours,mergedSegments)
    ]
    resDict=dict(zip(neighbours,sum(criteronScores*weights)))
    if segment in resDict.keys():
        resDict[segment]=np.inf
    return resDict

def lengthCriterion(segment1, segment2, mergedSegments, minValidData, inversedIndex):
    """
    compute the number of satisfied criteria ( non null values > (minValidData/2)  , number of segments (after merge) >10 , the total length(after merge) > 4 KM)
    """
    
    nullCrit = max(mergedSegments.loc[segment1].nonNullProp,mergedSegments.loc[segment2].nonNullProp)>minValidData/2
    nSegsCrit = inversedIndex[inversedIndex.isin([segment1,segment2])].size>10
    lengthCrit = (mergedSegments.loc[segment1].length+mergedSegments.loc[segment2].length)>4

    return np.array([np.sum([nullCrit+nSegsCrit+lengthCrit])])


def insertValue(mx,seg,values):
    """
    insert value in matrix (work arround the fact that you can't assign in a lambda function)
    """
    mx[seg]=values

def getCriteriaMatrix(inversedIndex,mergedSegments,updatedSpeed,weights):
    """
        Computer criteria between each couple of connected segments as a  matrix(dict of dict)
    """
    
    criteriaMatrix = {}
    inversedIndex.apply(lambda segment: insertValue(criteriaMatrix, segment,computeCriteria(segment,mergedSegments,updatedSpeed,weights,inversedIndex)))
    return criteriaMatrix

def customMin(x,mergedSegments, minValidData = 0.8):
    """
    compute the minimum value in a dict of dict (matrix)
    """
    if mergedSegments.loc[x].nonNullProp >= minValidData :  
            return np.inf

    idx = min(criteriaMatrix.get(x),
        key=lambda y : np.inf if y not in inversedIndex.values
                         else criteriaMatrix.get(x).get(y)
    )
    return np.inf if idx not in inversedIndex.values else criteriaMatrix.get(x).get(idx)

def getNeighboursCriteriaIndex(seg,mergedSegments,updatedSpeed,inversedIndex,weights,minValidData):
    """
    Compute the criteria between the segment and its neighbours
    """
    neighbours = getNeighbours(seg,mergedSegments,inversedIndex)
    if len(neighbours) == 0 : return pd.Series(index=[[],[]])

    df = pd.Series(index=[np.array([seg]*len(neighbours)),neighbours])
    return pd.Series(df.index.map( lambda x: computePairCriteria(*x,mergedSegments,updatedSpeed,inversedIndex,weights,minValidData)).values,df.index)

def getInversedCriteria(seg,mergedSegments,updatedSpeed,inversedIndex,weights,minValidData):
    """
    Compute the criteria between neighbours and the segment  (inversed index) 
    """
    
    neighbours = getNeighbours(seg, mergedSegments, inversedIndex)
    if len(neighbours) == 0 : return pd.Series(index=[[],[]])

    df = pd.Series(index=[neighbours, np.array([seg]*len(neighbours))])
    return pd.Series(df.index.map( lambda x: computePairCriteria(*x,mergedSegments,updatedSpeed,inversedIndex,weights,minValidData)).values,df.index)

def computePairCriteria(segment1, segment2, mergedSegments, updatedSpeed, inversedIndex, weights, minValidData,verbose=False):
    """
    Computer the criteria between two segments
    """
    
    segment2 = inversedIndex.loc[segment2]
    
    if mergedSegments.loc[segment1]['tag']['highway']!=mergedSegments.loc[segment2]['tag']['highway'] :
        return np.inf

    if mergedSegments.loc[segment1].nonNullProp >= minValidData :  
        return np.inf
    
    if segment2 == segment1 : return np.inf
    criteronScores = [
        profileSim(segment1,[segment2],updatedSpeed),
        directtion(segment1,[segment2],mergedSegments),
        shareNoEdges(segment1,[segment2],mergedSegments),
        lengthCriterion(segment1, segment2, mergedSegments, minValidData, inversedIndex)
    ]
    if verbose : print(criteronScores)
    return sum(criteronScores*weights)[0]



def linkInsOuts(ins,outs,meta,roundabout):
    """
    connect the ins(outs) segments of a roundabout with the outs(ins) segments of a roundabout
    
    """
    
    
    for onIns in ins :
        res= np.unique(np.concatenate((meta.loc[onIns]['outs'],outs)))
        meta.at[onIns,'outs']=res[res!=roundabout]
    for oneOuts in outs:
        res= np.unique(np.concatenate((meta.loc[oneOuts]['ins'],ins)))
        meta.at[oneOuts,'ins']=res[res!=roundabout]
        

def removeRounabouts(segmentsMeta):
    """
    remove all roundabouts from the segments meta
    """
    roundabouts = segmentsMeta[segmentsMeta.nodes.apply(lambda x : x[0]==x[-1] )]
    roundabouts.apply(lambda x : linkInsOuts(x['ins'],x['outs'],segmentsMeta,x.name) ,axis=1)
    segmentsMeta.drop(roundabouts.index,inplace=True)
    
    
def mergeRoundaboutChunks(segments):
    """
    merge connected small chunks of roundaout 
    """
    
    roundabouts = segments[segments.tag.apply(lambda x : ('junction' in x and(x['junction'] in ['circular','roundabout'] )))]
    roundabouts=roundabouts.nodes.apply(lambda x : pd.Series((x[0],x[-1]),index=['in','out']))
    roundabouts=roundabouts[roundabouts['in']!=roundabouts['out']]
    incopmleteRoundabouts  = roundabouts[~roundabouts['in'].isin(roundabouts['out'])].index.values
    while(len(incopmleteRoundabouts)!=0):
        roundabouts.drop(incopmleteRoundabouts,inplace=True)
        incopmleteRoundabouts  = roundabouts[~roundabouts['in'].isin(roundabouts['out'])].index.values


    def getRoundaboutFromChunks(startSeg,chunks):
        """
        connect roundabout chunks
        """
        
        sequence=[startSeg]
        nextIndex=startSeg
        while len(sequence)<=1 or sequence[0]!=sequence[-1]:
            outValue = chunks.loc[nextIndex]['out']
            nextIndex =  chunks[chunks['in']==outValue].index[0]
            sequence.append(nextIndex)
        return sequence[:-1]

    sequences=roundabouts.index.map(lambda x : getRoundaboutFromChunks(x,roundabouts))

    sequences=pd.Series(sequences)

    sequences = sequences.groupby(lambda x : sorted(sequences.loc[x])[0]).first()

    sequences=sequences.apply(lambda x : np.concatenate([ segments.nodes.loc[seg] for seg in  x]).tolist())
    
    segments.drop(roundabouts.loc[~roundabouts.index.isin(sequences.index)].index,inplace=True)
    for idx in sequences.index:
        segments.at[idx,'nodes']=sequences.loc[idx]


def computeSegmentsAttributes(mergedSegments,updatedSpeed):
    """
    Compute attributes use for computing criteria
    
    """
    
    mergedSegments=mergedSegments.assign(nonNullProp = updatedSpeed.notna().sum(axis=1)/updatedSpeed.columns.size)
    mergedSegments=mergedSegments.assign(   edges = mergedSegments.nodes.apply(lambda x : np.array([x[0],x[-1]])))
    mergedSegments=mergedSegments.assign( cosHead = mergedSegments['loc'].apply(lambda x :np.cos(np.deg2rad(get_north_azimut([x['coordinates'][-2],x['coordinates'][-1]])))))
    mergedSegments=mergedSegments.assign( sinHead = mergedSegments['loc'].apply(lambda x :np.sin(np.deg2rad(get_north_azimut([x['coordinates'][-2],x['coordinates'][-1]])))))
    mergedSegments=mergedSegments.assign( cosTail = mergedSegments['loc'].apply(lambda x :np.cos(np.deg2rad(get_north_azimut([x['coordinates'][0],x['coordinates'][1]])))))
    mergedSegments=mergedSegments.assign( sinTail = mergedSegments['loc'].apply(lambda x :np.sin(np.deg2rad(get_north_azimut([x['coordinates'][0],x['coordinates'][1]])))))
    mergedSegments=mergedSegments.assign(   head  = mergedSegments.nodes.apply(lambda x : x[-1]))
    mergedSegments=mergedSegments.assign(   tail  = mergedSegments.nodes.apply(lambda x : x[0]))
    return mergedSegments

def getCriteriaSeries(mergedSegments,updatedSpeed,inversedIndex,WEIGHTS,minValidData):
    """
    compute criteria between all connected segments in multiIndexed pandas Series
    """
    return pd.concat([getNeighboursCriteriaIndex(x,mergedSegments,updatedSpeed,inversedIndex,WEIGHTS,minValidData=MIN_VALID_DATA) for x in inversedIndex.values])
        
def mergeSpeedAndMeta(segmentToMerge,bestGroup,mergedSegments,updatedSpeed,inversedIndex,criteriaSeries,weights,minValidData):
    """
        merge/update attributes of the merged segments
    """
    mergeWithOutSeg = any( bestGroup == inversedIndex.loc[x] for x  in mergedSegments.at[segmentToMerge,'outs'])

    if mergeWithOutSeg :
        if(len(mergedSegments.at[bestGroup,'outs'])==0 and len(mergedSegments.at[segmentToMerge,'ins'])==0) or(mergedSegments.at[segmentToMerge,'head'] != mergedSegments.loc[bestGroup]['tail']):
            criteriaSeries.at[(segmentToMerge,bestGroup),]=np.inf
            return criteriaSeries
    else :
        if(len(mergedSegments.at[bestGroup,'ins'])==0 and len(mergedSegments.at[segmentToMerge,'outs'])==0) or (mergedSegments.at[segmentToMerge,'tail'] != mergedSegments.loc[bestGroup]['head']):
            criteriaSeries.at[(segmentToMerge,bestGroup),]=np.inf
            return criteriaSeries
        
    if( not mergeWithOutSeg and not any( bestGroup == inversedIndex.loc[x] for x  in mergedSegments.at[segmentToMerge,'ins'])) or ( mergeWithOutSeg and any( bestGroup == inversedIndex.loc[x] for x  in mergedSegments.at[segmentToMerge,'ins'])):
        print('****************************************\n\n*********************************\n\n bestgroup not found \n\n**********************************\n')
        print( not mergeWithOutSeg  ,any( bestGroup == inversedIndex.loc[x] for x  in mergedSegments.at[segmentToMerge,'ins']))
        print(segmentToMerge,bestGroup)
        print( "ins outs")
        print([inversedIndex.loc[x] for x  in mergedSegments.at[segmentToMerge,'ins']])
        print([inversedIndex.loc[x] for x  in mergedSegments.at[segmentToMerge,'outs']])
        print( 'b ins outs')
        print([inversedIndex.loc[x] for x  in mergedSegments.at[bestGroup,'ins']])
        print([inversedIndex.loc[x] for x  in mergedSegments.at[bestGroup,'outs']])
        
        
        
    
    if mergeWithOutSeg :
        
        mergedSegments.at[segmentToMerge,'outs'] =  mergedSegments.loc[bestGroup]['outs']
        mergedSegments.at[segmentToMerge,'cosHead' ] = mergedSegments.at[bestGroup,'cosHead' ]
        mergedSegments.at[segmentToMerge,'sinHead' ] = mergedSegments.at[bestGroup,'sinHead' ]
        mergedSegments.at[segmentToMerge,'head'] =  mergedSegments.loc[bestGroup]['head']

    else :
        mergedSegments.at[segmentToMerge,'ins'] = mergedSegments['ins'][bestGroup]
        mergedSegments.at[segmentToMerge,'cosTail' ] = mergedSegments.at[bestGroup,'cosTail' ]
        mergedSegments.at[segmentToMerge,'sinTail' ] = mergedSegments.at[bestGroup,'sinTail' ]
        mergedSegments.at[segmentToMerge,'tail'] =  mergedSegments.loc[bestGroup]['tail']

    newLength = mergedSegments.loc[segmentToMerge]['length']+  mergedSegments.loc[bestGroup]['length']

    mergedSegments.at[segmentToMerge,'length'] = newLength
    
    mergedSegments.at[segmentToMerge,'nonNullProp'] = (updatedSpeed.loc[segmentToMerge].notna() | updatedSpeed.loc[bestGroup].notna()).sum()/updatedSpeed.columns.size


    #mergedSegments.at[segmentToMerge,'edges' ] = np.unique(np.concatenate((mergedSegments.loc[segmentToMerge]['edges'],mergedSegments.loc[bestGroup]['edges'])))
    #speed update
    mergedSegments.at[segmentToMerge,'edges' ] = np.unique(np.concatenate((mergedSegments.loc[segmentToMerge]['edges'],mergedSegments.loc[bestGroup]['edges'])))
    
    updatedSpeed.loc[segmentToMerge]=updatedSpeed.loc[set([inversedIndex.loc[x] for x in [segmentToMerge,bestGroup]])].mean()
    updatedSpeed.drop(bestGroup, inplace=True)
    mergedSegments.drop(bestGroup, inplace=True)
    inversedIndex.replace(bestGroup, segmentToMerge, inplace=True)
    
    
    criteriaSeries.drop(index=bestGroup, level=0, inplace = True)
    
    segmentCriteria = getNeighboursCriteriaIndex(segmentToMerge, mergedSegments,updatedSpeed,inversedIndex,weights,minValidData)
    
    criteriaSeries.drop(index=segmentToMerge, level=0, inplace=True )
    
    
    
    criteriaSeries = criteriaSeries.append( segmentCriteria )
        
    criteriaSeries.drop(index=bestGroup, level=1, inplace = True)
    criteriaSeries.drop(index=segmentToMerge, level=1, inplace = True)
    
    inverseSegmentCriteria = getInversedCriteria(segmentToMerge,mergedSegments,updatedSpeed,inversedIndex,weights,minValidData)
    criteriaSeries = criteriaSeries.append( inverseSegmentCriteria )

    
    
    return criteriaSeries
    
    

def mergeSegments(minValidData = 0.8,weights=np.array([1,1,1]).reshape(-1,1),speedsMx = None,):
    """
    Main algorithem for merging segments
    
    ** Compute meta/ segments and preprocess the data
    ** Compute the speed matrix and aligne the index
    ** Compute attributes for each segment
    ** Compute criteria of merge for each connected couple of segments
    ** Iteratively merge segments
    
    returns the index of merged segments
    """
    
    print("Computing raw data : ")
    print(getTimeSpent(reset=True))

    print("getting segments and meta :")
    segments=OsmProcessing.getSegments(osmWays)
    segments = OsmProcessing.setOneWay(segments)
    mergeRoundaboutChunks(segments)
    segmentsMeta = OsmProcessing.buildSegmentsMeta(segments,linearOnly=True)
    removeRounabouts(segmentsMeta)
    print(getTimeSpent())

    print("getting speed matrix :")
    
    if type(speedsMx) == type(None):
        sm = SpeedMatrix("geolytics", "coyote", "geolytics", "ways")
        roads_ids = segments.index.values.tolist()
        speeds = sm.get_speed_matrix(15, roads_ids, 14, 19)

    else :
        speeds = speedsMx
    updatedSpeed =speeds.reindex(segmentsMeta.segmentID).set_index(segmentsMeta.index)
    print(getTimeSpent())
    print("computing meta attributs and inversed index :")
    mergedSegments = segmentsMeta.copy()
    inversedIndex = pd.Series(data=mergedSegments.index.values.copy(),index=mergedSegments.index.values.copy(),name='segmentIndex')
    mergedSegments = computeSegmentsAttributes(mergedSegments,updatedSpeed)
    print(getTimeSpent())

    print("computing merging criteria :")
    criteriaSeries=getCriteriaSeries(mergedSegments,updatedSpeed,inversedIndex,WEIGHTS,minValidData)
    print(getTimeSpent())

    print("start merging :")
    itr=0
    maxItr = len(inversedIndex)
    layers = []
    fmap = Plotting.getFoliumMap()

    while mergedSegments.nonNullProp.min()<minValidData and criteriaSeries.min()!= np.inf :
        itr+=1
        """
        what I want is :
        find the segment I want to merge (random, lowest data rate, highest insufficient data rate).
        search for best candidate to merge with :
            - same direction
            - same profile
            - connected edges
        group the two segments to form a group of segment
        update data for this group:
            - update speed to be the mean (may change) speed of the two (group) segments.
            - update the hierarchical data for the merged group
            - update the inverted index to match the indexes of both speed and hierarchy.
        """
        seg1,seg2 = criteriaSeries.idxmin()

        if len(np.intersect1d(mergedSegments.edges[seg1],mergedSegments.edges[seg2],assume_unique=True))>1:
            criteriaSeries.at[(seg1,seg2),]=np.inf
            print(itr,'it happend')
            continue
        if itr %100 == 1 : 
            print('iter : ',itr,' seg 1 : ', seg1,' seg2 : ', seg2,' inv seg2 : ',inversedIndex[seg2],' mean non null ',mergedSegments.nonNullProp.mean())
#        seg2= inversedIndex[seg2]
        criteriaSeries=mergeSpeedAndMeta(seg1,seg2,mergedSegments,updatedSpeed,inversedIndex,criteriaSeries,weights,minValidData)
        if(itr%500 == 0):
            layers.append(Plotting.saveBigMergesMap(inversedIndex,segmentsMeta,fmap,"iter "+str(itr)))
    print(getTimeSpent())

    fmap= Plotting.stackHistotyLayers(layers,fmap)
    fmap.save('RoadsHist.html')
            
    return inversedIndex ,segmentsMeta,mergedSegments
