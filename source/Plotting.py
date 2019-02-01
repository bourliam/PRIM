import ipywidgets as widgets
import folium
import CustomUtils
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
maxV=10
global multipleTrips
global carIdWidget
carIdWidget = widgets.Dropdown(options=list([]))

tripIdWidget=widgets.IntSlider(min=0,max=maxV)
dateWidget=widgets.DatePicker(
    description='Pick a Date',
    disabled=False,
)

def drawOneTrip(trip):
    """
    return a map with the trip plotted
    
    trip : pandas Series
        an entry in the trips data frame
    """
    folium_map = folium.Map(location=[48.10301,-1.65537],
                    zoom_start=13,
                    tiles="OpenStreetMap")
    addTrip(folium_map,trip)
    return folium_map


def getFoliumMap():
    folium_map = folium.Map(location=[48.10301,-1.65537],
                    zoom_start=13,
                    tiles="OpenStreetMap")    
    return folium_map


def drawMultipleTrips(trips,interact):
    """
    (Interactive,deprecated)
    return a map with the trip plotted
    
    trips : pandas Series
        multiple entries in the trips data frame
    """
    global tripIdWidget
    global multipleTrips
    multipleTrips=trips
    tripIdWidget=widgets.IntSlider(min=0,max=len(trips)-1)
    interact(prepareMultipleTrips)

def prepareMultipleTrips(trip=tripIdWidget,All=False):
    
    """
    (Interactive,deprecated)
    prepare data to be plotted
    trip : str 
        trip ID
    """
    global multipleTrips
    trips = multipleTrips
    folium_map = folium.Map(location=[48.10301,-1.65537],
                    zoom_start=13,
                    tiles="OpenStreetMap")
    if All :
        colors=plt.cm.brg([ x/len(trips) for x in range(len(trips))])
        for s,c  in zip(trips.iterrows(),colors):
            addTrip(folium_map,s[1],matplotlib.colors.rgb2hex(c),edges_only=True)
    else :    
        addTrip(folium_map,trips.iloc[trip],'purple')
        print('Start : ',trips.iloc[trip]['begin'],'\nEnd : ',trips.iloc[trip]['end'],'\nDur : '+str(trips.iloc[trip]['dur'])+' mins')
    display(folium_map)

def addTrip(folium_map,trip,color='red',edges_only=False):
    """    
    add a trip to the folium map
    
    trip : pandas series
        the trip to add to map
        
    color : str or hex representation 
        the color to use for the trip
        
    edges_only : bool 
        whether to use all points or edges only in the plot
    """
    locs=[[x['coordinates'][1],x['coordinates'][0]] for x in trip['loc']]
    if edges_only:
        locs=[locs[0],locs[len(locs)-1]]
    
    MarkerCluster(locations=locs,
                  icons = [folium.Icon(color='green', icon='play-circle'),*[folium.Icon(color=color, icon='info-sign') if not edges_only else '' for _ in range(len(locs)-2) ],folium.Icon(color='black', icon='stop') if len(locs)>1 else []],
                  popups=['ID : '+trip.id+'<br>Speed : '+str(s)+'<br>Time : '+ str(t) +'<br> cooRdinates : '+'lat : '+str(l['coordinates'][1])+' lon : '+str(l['coordinates'][0])for s,t,l in zip(trip['speed'],trip['time'],trip['loc'])]).add_to(folium_map)
    folium.PolyLine(locations=locs,color=color).add_to(folium_map)

    
def addMultipleTrips(folium_map, carTrips,edges_only=False,colors=None):
    if type(colors)==type(None) :
        colors=plt.cm.brg([ x/len(carTrips) for x in range(len(carTrips))])
    for s,c  in zip(carTrips.iterrows(),colors):
        addTrip(folium_map,s[1],matplotlib.colors.rgb2hex(c),edges_only)
        
def addCarTrips(folium_map, trips, carsID):
    carTrips=trips[trips.id==carsID]
    colors=plt.cm.brg([ x/len(carTrips) for x in range(len(carTrips))])
    for s,c  in zip(carTrips.iterrows(),colors):
        addTrip(folium_map,s[1],matplotlib.colors.rgb2hex(c))
        
def printmap(carID=carIdWidget,byTrip=False,trip=tripIdWidget,byDate=False,date=dateWidget,trips=None):
    """
    (Interactive)
    plot trips with the disired options
    
    carID : str
        the id of the device
        
    byTrips : bool
        whether to plot all trips at once or one by one
        
    trip : int
        tripID used if plotting trips one by one
        
    byDate : bool
        whether to plot all days at once or day by day
        
    date : datetime
        the date to plot 
        
    trips : pandas dataFrame
        dataFrame of trips
    """
    carTrips=trips[trips.id==carID]    
    maxV= len(carTrips)-1
    if(byDate):
        carTrips=carTrips[carTrips.day ==date]
        maxV= len(carTrips)-1 if (len(carTrips)-1)>0 else 0
    folium_map = folium.Map(location=[48.10301,-1.65537],
                        zoom_start=13,
                        tiles="OpenStreetMap")
    if byTrip and len(carTrips)>0:
        if(trip>maxV):
            trip=0
        addTrip(folium_map,carTrips.iloc[trip],'purple')
        print('Start : ',carTrips.iloc[trip]['begin'],'\nEnd : ',carTrips.iloc[trip]['end'],'\nDur : '+str(carTrips.iloc[trip]['dur'])+' mins')
    else :
        addCarTrips(folium_map,carTrips,carID)
    display(folium_map)
    if(date == None or date>carTrips.day.max()):
        dateWidget.value=carTrips.day.max()
    if(date == None or date<carTrips.day.min()):
        dateWidget.value=carTrips.day.min()
    tripIdWidget.max=maxV
    
    
def dataFrameAsImage(df,cmap=plt.cm.hot):
    """
    returns an image represntation of the dataFrame df using colormap cmap
    
    df : pandas dataFrame
    """
    plt.figure(figsize=(18,18))
    plt.imshow(df,cmap=cmap)
    
def plotUserRegionsOfInterst(userEdges,folium_map=None,show_outliers=True):
    """
    plot user regions of interst
    
    userEdges : pandas series
        user edges
    
    nClusters : int
        number of clusters
        
    clusters : array
        label for each position in user edges
        
    folium_map : folium.map optional
        the map to plot the regions on
    """
    if not folium_map :
        folium_map=getFoliumMap()
    
    validIds=np.where(userEdges.clusters_begin>=-show_outliers)[0]
    colors=CustomUtils.getClustersColors(userEdges.clusters_begin[validIds])    
    beginLayer = getLayerWithPositions(userEdges.edges_begin[validIds],colors,name='begin',fmap=folium_map)
    beginLayer.add_to(folium_map)
    
    validIds=np.where(userEdges.clusters_end>=-show_outliers)[0]
    colors=CustomUtils.getClustersColors(userEdges.clusters_end[validIds])
    endLayer = getLayerWithPositions(userEdges.edges_end[validIds],colors, name='end', fmap=folium_map)
    endLayer.add_to(folium_map)
    
    folium.LayerControl(collapsed=False).add_to(folium_map)
    return folium_map

def getLayerWithPositions(positions,colors,fmap,name='map'):
    layer =    folium.plugins.FeatureGroupSubGroup(fmap,name=name)
    [folium.CircleMarker(location=positions[i][::-1],
                                  color=matplotlib.colors.rgb2hex(colors[i])
                                 ).add_to(layer) 
    for i in range(len(positions))]
    return layer

def plotRoads(roads,colors=None):
    """
    plot roads with colors
    
    roads: pandas Series
        Roads cooredinates
    
    Colors: Array or None
        color for each road
    """
    folium_map = folium.Map(location=[48.14301,-1.69537],
                        zoom_start=13,
                        tiles="OpenStreetMap")
    if type(colors)==type(None) :
        [folium.PolyLine(locations=[lo[::-1] for lo in x['coordinates']]).add_to(folium_map) for x in roads]
    else : 
        [folium.PolyLine(locations=[lo[::-1] for lo in x['coordinates']],color=color).add_to(folium_map) for x,color in zip(roads,colors)]

    return folium_map

def plotUserTripsClusters(folium_map,trips,userEdges,edges_only=False):
    if folium_map == None :
        folium_map=Plotting.getFoliumMap()
    colors = CustomUtils.getClustersColors(userEdges.trip_clusters)
    
    for cluster in set(userEdges.trip_clusters):
        layer = folium.plugins.FeatureGroupSubGroup(folium_map,name='cluster_'+str(cluster))
        tripColors = colors[np.where(userEdges.trip_clusters==cluster)[0]]
        tripIds = userEdges.edges_trip_id[np.where(userEdges.trip_clusters==cluster)[0]]
        [addTrip(layer,trips.loc[idt],color,edges_only) for idt,color in zip(tripIds,tripColors)]
        layer.add_to(folium_map)
    folium.LayerControl(collapsed=False).add_to(folium_map)

    return folium_map