from multiprocessing import Pool
import sys
# MONGODB 
from pymongo import MongoClient
# Ploting
import matplotlib
import matplotlib.pyplot as plt
from vincenty import vincenty
# Data and processing
import datetime
import numpy as np
import pandas as pd
import time
# Printing
import pprint

import traceback
import logging

import config_trajet

from datetime import date, datetime, timedelta

def get_logger(name):
	formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
	handler = logging.StreamHandler(stream=sys.stdout)
	handler.setFormatter(formatter)
	logger = logging.getLogger(name)
	logger.setLevel(logging.DEBUG)
	logger.addHandler(handler)

	return logger


def reverseVincenty(x, y):
	return vincenty(x[::-1], y[::-1])
	
def timeSpentSince(startTime):
	endTime = time.time()
	logger.info('took {:.3f} ms'.format((endTime-startTime)*1000.0))

def temporelTripsFilter(st,thresh= 15):
	# index of records where the difference between the two successive values is higher than thresh(default 15mins)
	time=np.array(st['time'])
	timeDiff=time[1:]-time[:-1]
	return  np.where(timeDiff >np.timedelta64(thresh,'m'))[0]+1 

def stopsFilter(st, tempFilter, thresh= 10):
	# maximum stop time filter
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
	# segments
	return [[locs[cl][i:j] for cl in locs.index] for i,j in zip(np.append(0,idx),np.append(idx,len(locs['loc'])))]

def buildTrips(df,adhocCoef= 1.24, minPointsDuration= 15,stopsDuration= 10):
	startTime = time.time()
	logger.info('grouping data points by car : ....')
	carsDF=df.groupby(by='id').agg(lambda x:[*x.values])
	#logger.info('grouping data points by car : Done')
	timeSpentSince(startTime)
	startTime = time.time()
	logger.info('extracting trips : ....')
	trips =carsDF.apply(lambda x : list(zip(*segmentsOn(x,stopsFilter(x,temporelTripsFilter(x,thresh=minPointsDuration),thresh=stopsDuration)))),axis=1)
	#logger.info('extracting trips : Done')
	timeSpentSince(startTime)
	startTime = time.time()
	logger.info('splitting trips by car : ....')
	trips=pd.DataFrame([[trips.index[idx],*row] for idx in range(len(trips)) for row in zip(*trips.iloc[idx])],columns=['id',*carsDF.columns])
	#logger.info('splitting trips by car : Done')
	timeSpentSince(startTime)
	#return trips
	startTime = time.time()
	logger.info('adding columns (day,begin point,end point, duration : ....')
	# trips=trips.assign(day=trips.time.apply(lambda x:pd.to_datetime(x[0]).date()))
	trips=trips.assign(day=trips.time.apply(lambda x:x[0].item() // 10000000000))
	trips=trips.assign(begin=trips.apply(lambda x : dict([(c,x[c][0]) for c in ['_id', 'loc','time','heading','speed','INSEE_iris_code']]),axis=1))
	trips=trips.assign(end=trips.apply(lambda x : dict([(c,x[c][len(x[c])-1]) for c in ['_id', 'loc','time','heading','speed','INSEE_iris_code']]),axis=1))
	trips=trips.assign(dur=trips.time.apply(lambda x :x[len(x)-1]-x[0] ))
	trips=trips.assign(time_difference_seconds = trips.time.apply(lambda loc : np.array([(y-x)/np.timedelta64(1, 's') for x,y in zip(loc[:-1],loc[1:])])))
	#logger.info('adding columns (day,begin point,end point, duration : Done')
	timeSpentSince(startTime)
	startTime = time.time()
	logger.info('Calculating distances: ...')
	trips=trips.assign(pairs_distances_km=trips['loc'].apply(lambda loc : np.array([reverseVincenty(x['coordinates'],y['coordinates'])*adhocCoef for x,y in zip(loc[:-1],loc[1:])])))
	trips=trips.assign(trip_distance_km=trips.pairs_distances_km.apply(sum))
	trips=trips.assign(trip_distance_km_vo = trips['loc'].apply(lambda x: reverseVincenty(x[0]['coordinates'],x[len(x)-1]['coordinates'])))
	#logger.info('Calculating distances: Done')
	timeSpentSince(startTime)
	startTime = time.time()
	return trips

#Filtering trips
def filterTrips(trips,cnd='ALL',irisFilter=[],maxOverallSpeed=0.2,minDuration=60,minDistance=0.2,maxSpeed=0,maxJumpSpeed=0.05)							  :
	# Max overall speed (km/second)
	globalSpeedFilter = ((trips.trip_distance_km/trips.dur.apply(lambda x : x/np.timedelta64(1, 's')))<maxOverallSpeed)
	trips=trips[globalSpeedFilter]
	# minimumTripDistance (km)
	minimumTripDistance = (trips.trip_distance_km>=minDistance)
	trips=trips[minimumTripDistance]
	# minimum trip duration (seconds)
	minTripDuration=(trips.dur>=np.timedelta64(minDuration, 's'))
	trips=trips[minTripDuration]
	# maximum Start point speed (default 0)
	maxStartSpeed = (trips.begin.apply(lambda x : x['speed']<=maxSpeed))
	trips=trips[maxStartSpeed]

	if len(trips) <= 0:
		return trips
	
	# maximum end point speed	
	maxEndSpeed = (trips.end.apply(lambda x : x['speed']<=maxSpeed))
	trips=trips[maxEndSpeed]

	if len(trips) <= 0:
		return trips
	# geographique jump on start
	geoJump=(trips.pairs_distances_km.apply(lambda x : x[0])/trips.time_difference_seconds.apply(lambda x : x[0]))<=maxJumpSpeed
	trips=trips[geoJump]
	
	return trips


def get_point_iris_parallel(loc):
	from pymongo import MongoClient
	#import config_trajet	
	remote_client = MongoClient(host='srv142.it4pme.fr', port=27017)
	res = remote_client.geo4cast.iris_geo_coords.find_one({
		'loc': {
			'$geoIntersects': {
				'$geometry': loc
			}
		}
	}, {
		'INSEE_iris_code': True
	})
	if res is None:
		return 'N/A'
	return res['INSEE_iris_code']

def multi_trips_point_iris_parallel(trips):
	_remote_client = MongoClient(host='srv142.it4pme.fr', port=27017)
	IrisCodes=[]
	for trip in trips :
		tripIris=[]
		for loc in trip['loc']:
			res = _remote_client.geo4cast.iris_geo_coords.find_one({
				'loc': {
					'$geoIntersects': {
						'$geometry': loc
					}
				}
			}, {
				'INSEE_iris_code': True
			})
			if res is None:
				tripIris.append('N/A')
			else:
				tripIris.append(res['INSEE_iris_code'])
		IrisCodes.append(tripIris)
	return IrisCodes



def get_point_iris(loc, irisCollection):
	res = irisCollection.find_one({
		'loc': {
			'$geoIntersects': {
				'$geometry': loc
			}
		}
	}, {
		'INSEE_iris_code': True
	})
	if res is None:
		return 'N/A'
	return res['INSEE_iris_code']
	
	
def get_yesterday_collection(yesterday, collection_name):
	yesterday = (date.today() - timedelta(1))
	return collection_name + '_{y}{m}{d}' . format(y=yesterday.year, m=yesterday.month, d=yesterday.day)



# rc = ipp.Client()
# lb_view = rc.load_balanced_view()
# lb_view.map_sync(setupIpyparallelClusters,range(len(lb_view)))
# #dview = rc[:]		

#collection_name = get_yesterday_collection(config_trajet.Mongo.data_collection)
collection_name = 'coyote_data_20181225'
#ride_collection_name = get_yesterday_collection(config_trajet.Mongo.ride_collection)
ride_collection_name = 'trajet_20181225'

logger = get_logger('trajets')

script_date_start = datetime.today()

# mongo client
# client = MongoClient()
client = MongoClient(host=config_trajet.Mongo.Host, port=config_trajet.Mongo.Port)
remote_client = MongoClient(host=config_trajet.Geo4Cast.Host, port=config_trajet.Geo4Cast.Port)
# db name  'coyote'
# db = client.congestion
db = client.coyote
rem_db = remote_client.geo4cast
# iris collection "db.iris_geo_coords"
irisCollection = rem_db.iris_geo_coords
# coyote data "db.coyote"
coyoteData = db[collection_name]
# coyoteData = db.coyote
# Main
startTime = time.time()
script_start_time = time.time()
server_name = 'srv060'
# loading Data
nbDevices=1000
p_counter = 0
p_max = 5000
nWorkers = 4


logger.info('loading Data (batch ' + str(nbDevices) + ')')
startTime = time.time()
distIds=coyoteData.distinct("id")
timeSpentSince(startTime)
all_results = []
for group in zip(range(0,len(distIds),nbDevices),list(range(0,len(distIds),nbDevices))[1:]+[len(distIds)]):
	logger.info('getting data by Id')
	startTime = time.time()
	ids=coyoteData.find({"id":{'$in':distIds[slice(*group,1)]}}, no_cursor_timeout=True)
	timeSpentSince(startTime)
	logger.info('Listing data into data frame ' + str(ids.count()))
	startTime = time.time()
	df=pd.DataFrame(list(ids))
	ids.close()
	timeSpentSince(startTime)
	#df = pd.DataFrame(list(coyoteData.find({})))
	# Sort data by time
	logger.info('Sort data by time')
	df.sort_values(by='time',inplace=True)
	if type(df.time.values[0]) == np.int64 :
		transTime=pd.to_datetime(df.time,unit='s')
		df = df.assign(time=transTime)
	logger.info('building trips')
	# Creating trips-
	trips= buildTrips(df)

	try:
		logger.info("filtering trips")
		filterdTrips=filterTrips(trips,irisFilter=[])
		if len(filterdTrips) > 0:
			filterdTrips.begin = filterdTrips.begin.map(lambda x: {'loc': x['loc'], 'INSEE_iris_code': get_point_iris(x['loc'], irisCollection), 'time': x['time'].item() // 10000000000})
			filterdTrips.end = filterdTrips.end.map(lambda x: {'loc': x['loc'], 'INSEE_iris_code': get_point_iris(x['loc'], irisCollection), 'time': x['time'].item() // 10000000000})
			
			filterdTrips.dur = filterdTrips.dur.apply(lambda x : int(x.total_seconds()))
			filterdTrips = filterdTrips.assign(pointsId = filterdTrips._id.apply(lambda x : x))
			filterdTrips = filterdTrips.assign(server_name=server_name)
			filterdTrips = filterdTrips.assign(indexed_iris=0)
			filterdTrips.time = filterdTrips.time.apply(lambda x: [y.item() // 10000000000 for y in x])
			filterdTrips = filterdTrips.drop(['_id'], axis=1)
			filterdTrips = filterdTrips.drop(['speed'], axis=1)
			# filterdTrips = filterdTrips.drop(['loc'], axis=1)
			filterdTrips = filterdTrips.drop(['pairs_distances_km'], axis=1)
			# filterdTrips = filterdTrips.drop(['trip_distance_km'], axis=1)
			# filterdTrips = filterdTrips.drop(['trip_distance_km_vo'], axis=1)
			filterdTrips = filterdTrips.drop(['time'], axis=1)
			filterdTrips = filterdTrips.drop(['road'], axis=1)
			filterdTrips = filterdTrips.drop(['type'], axis=1)
			filterdTrips = filterdTrips.drop(['day'], axis=1)
			filterdTrips = filterdTrips.drop(['heading'], axis=1)
			filterdTrips = filterdTrips.drop(['country'], axis=1)
			filterdTrips = filterdTrips.drop(['time_difference_seconds'], axis=1)
			allTrips = filterdTrips.to_dict('records')
			logger.info('PREPARING BATCH FOR INSERTION ' + str(len(allTrips)))

			startTime = time.time()
			with Pool(processes=nWorkers) as pool:
				res=pool.map(multi_trips_point_iris_parallel, np.array_split(allTrips,nWorkers))
			res=np.concatenate(res)
			for it, trip in enumerate(allTrips):
				trip.pop('loc', None)
				trip['INSEE_iris_code'] = res[it]
				
			all_results += allTrips
			len_res = len(all_results)
			logger.info('All results length : ' + str(len_res))
			if len_res >= nbDevices:
				p_counter += len_res
				logger.info('INSERTING BATCH')
				db[ride_collection_name].insert_many(all_results, ordered=False)
				if p_counter >= p_max:
					logger.warn('*** Reached ' + str(p_max) + ' documents ***')
					p_counter = 0
				all_results = []
			timeSpentSince(startTime)
	except:
		logger.info('Started at ' + script_date_start.isoformat())
		traceback.print_exc()
		#logger.fatal('filterdTrips : {}' . format(filterdTrips.to_dict('records')))
		break
		

if len(all_results) > 0:
	logger.info('INSERTING LAST BATCH OF ' + str(len(all_results)))
	db.trajets.insert_many(all_results)

logger.info('THE END ' + str(len(all_results)))
timeSpentSince(script_start_time)
script_date_end = datetime.today()

logger.info('Started at ' + script_date_start.isoformat() + ' - Ended at ' + script_date_end.isoformat())



