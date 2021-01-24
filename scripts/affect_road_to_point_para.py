
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import pprint
import time
import math
import os

from threading import Thread
from multiprocessing import Process


"""
Usage :

client = MongoClient()
collection = client.geo4cast.subset_rennes
osm_roads = client['osm-rennes'].roads

affect = Affect_road_to_point(geo4cast, collection, osm, osm_roads, 10)
affect.affect_para(os.cpu_count())
"""



def get_north_azimut(coords):
  """
  Get the north azimut of a road by using its geometry.   
  @params   
  coords: The array of coordinates from the osm data.
  """
  degs = math.degrees(math.atan2((coords[-1][0] - coords[0][0]),(coords[-1][1] - coords[0][1])))
  if degs < 0:
      degs += 360.0
  return degs


def get_headings(road):
  """
  Get the headings of a road by using it's geometry.   
  @params   
  road: The object road from the osm data. (we need the coordinates, the keys and the tags)
  """
  coords = road['loc']['coordinates']
  headings = []
  heading = get_north_azimut(coords)
  if 'oneway' in road['key']:
      for tag in road['tag']:
          if tag['k'] == 'oneway':
              oneway=tag['v']
              break
      if oneway == '-1':
          headings.append((heading + 180)%360)
      else:
          headings.append(heading)
  else:
      headings.append(heading)
      headings.append((heading + 180)%360)
  
  return headings



class Worker(Process):
  """
  Worker class. The thread that updates some of the points.
  """

  def __init__(self, number, limit, db_name, collection_name, db_osm_name, roads_name, maxDistance):
    Process.__init__(self)
    
    self.db_name = db_name
    self.collection_name = collection_name
    self.db_osm_name = db_osm_name
    self.roads_name = roads_name
    self.number = number
    self.limit = limit
    self.maxDistance = maxDistance

  def run(self):
    time.sleep(1*self.number+3)
    client =  MongoClient(host = "mongodb://mbouchouia:cbf20Li34!@mongodb-tp.enst.fr", port=27017,connect=False)

    client = MongoClient(host = "mongodb://mbouchouia:cbf20Li34!@mongodb-tp.enst.fr", port=27017)

    self.collection = client[self.db_name][self.collection_name]
    self.osm_roads = client[self.db_osm_name][self.roads_name]

    requests = []
    count = 0
    points = self.collection.find({},{'loc':1,'heading':1},no_cursor_timeout=True,batch_size=1000).skip(self.number*self.limit).limit(self.limit)
    try :
        for point in points:
          if count%1000==0 and count<3000 :
          
            print("{:} done {:}, {:}".format(self.number,count,time.time()))
          road = self.osm_roads.find_one({
            "loc": {
              "$near": {
                "$geometry": {
                  "type": "Point" ,
                  "coordinates": point['loc']['coordinates']
                },
                "$maxDistance": self.maxDistance
              }
            }
          },
          { "_id": 1, "loc": 1, "key": 1, "tag": 1}
          )

          heading = 0

          if road == None:
            road = {"_id": ""}
          else:
            headings = get_headings(road)
            if len(headings) == 2:
              if 180 - abs(abs(headings[0] - point['heading']) - 180) > 90:
                heading = 1


          requests.append(UpdateOne({ '_id': point['_id'] }, { '$set': { 'matching_road': road['_id'], 'heading_road': heading } }))

          count += 1

          if count % 1000 == 0:
            try:
              self.collection.bulk_write(requests,ordered=False)

              requests = []
            except BulkWriteError as bwe:
              pprint(bwe.details)

          if self.number == 0 and count % 10000 == 0:
            print("Worker", self.number, ":", count, "points modified /", self.limit, end='\r', flush=True)

        if count % 1000 != 0:
          self.collection.bulk_write(requests)

        print("Worker", self.number, "done:", count, "points modified")
    except:
        points.close()
        print("closing cursor due to error")
        raise


class Affect_road_to_point:
  """
  Class to affect osm roads to the points.    
  params:     
  @collection: the pymongo collection containing the points.     
  @osm_roads: the pymongo collection with the osm roads.      
  @maxDistance: the max distance between the point and the road (meters).    
  """

  def __init__(self, db_name, collection_name, db_osm_name, roads_name, maxDistance):

    self.maxDistance = maxDistance
    self.db_name = db_name
    self.collection_name = collection_name
    self.db_osm_name = db_osm_name
    self.roads_name = roads_name



  def affect_para(self, nb_workers):
    """
    Affect the roads using nb_workers threads.
    """
    start = time.time()
    
    client =  MongoClient(host = "mongodb://mbouchouia:cbf20Li34!@mongodb-tp.enst.fr", port=27017)
    self.collection = client[self.db_name][self.collection_name]

    nbPoints = self.collection.count()

    limit = math.ceil(nbPoints / nb_workers)

    threads = [ Worker(k, limit, self.db_name, self.collection_name, self.db_osm_name, self.roads_name, self.maxDistance) for k in range(nb_workers) ]

    for t in threads:
      t.start()

    for t in threads:
      t.join()
    
    end = time.time()
    print("Done in", round(end - start, 3) ,"seconds")





  def affect(self):
    """
    (deprecated)   
    Affect the roads using only one main thread.
    """

    client =  MongoClient(host = "mongodb://mbouchouia:cbf20Li34!@mongodb-tp.enst.fr", port=27017)

    self.collection = client[self.db_name][self.collection_name]
    self.osm_roads = client[self.db_osm_name][self.roads_name]

    start = time.time()
    requests = []
    count = 0
    nbPoints = self.collection.count()
    for point in self.collection.find():
      road = self.osm_roads.find_one({
        "loc": {
          "$near": {
            "$geometry": {
              "type": "Point" ,
              "coordinates": point['loc']['coordinates']
            },
            "$maxDistance": self.maxDistance
          }
        }
      },
      { "_id": 1}
      )
      
      if road == None:
        road = {"_id": ""}

      requests.append(UpdateOne({ '_id': point['_id'] }, { '$set': { 'matching_road': road['_id'] } }))
      
      count += 1

      if count % 1000 == 0:
        try:
          self.collection.bulk_write(requests)
          print(count, "points modified", end='\r')
          requests = []
        except BulkWriteError as bwe:
          pprint(bwe.details)

    if count % 1000 != 0:
      self.collection.bulk_write(requests)
    
    end = time.time()
    print("\nDone.")
    print(count, "points modified /", nbPoints, "in", round(end - start, 3) ,"seconds")
