
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
import pprint
import time
import math
import os

from threading import Thread


"""
Usage :

client = MongoClient()
collection = client.geo4cast.subset_rennes
osm_roads = client['osm-rennes'].roads

affect = Affect_road_to_point(collection,osm_roads, 10)
affect.affect_para(os.cpu_count())
"""

class Worker(Thread):
  """
  Worker class. The thread that updates some of the points.
  """

  def __init__(self, number, limit, collection, osm_roads, maxDistance):
    Thread.__init__(self)
    self.number = number
    self.limit = limit
    self.collection = collection
    self.osm_roads = osm_roads
    self.maxDistance = maxDistance

  def run(self):
    requests = []
    count = 0
    
    for point in self.collection.find().skip(self.number*self.limit).limit(self.limit):
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
          
          requests = []
        except BulkWriteError as bwe:
          pprint(bwe.details)
    
    if count % 1000 != 0:
      self.collection.bulk_write(requests)

    print("Worker", self.number, "done:", count, "points modified")



class Affect_road_to_point:
  """
  Class to affect osm roads to the points.    
  params:     
  @collection: the pymongo collection containing the points.     
  @osm_roads: the pymongo collection with the osm roads.      
  @maxDistance: the max distance between the point and the road (meters).    
  """

  def __init__(self, collection, osm_roads, maxDistance):
    self.collection = collection
    self.osm_roads = osm_roads
    self.maxDistance = maxDistance




  def affect_para(self, nb_workers):
    """
    Affect the roads using nb_workers threads.
    """
    start = time.time()
    nbPoints = self.collection.count()

    limit = math.ceil(nbPoints / nb_workers)


    threads = [ Worker(k, limit, self.collection, self.osm_roads, self.maxDistance) for k in range(nb_workers) ]

    for t in threads:
      t.start()

    for t in threads:
      t.join()
    
    end = time.time()
    print("Done")
    print(count, "points modified /", nbPoints, "in", round(end - start, 3) ,"seconds")





  def affect(self):
    """
    (deprecated)   
    Affect the roads using only one main thread.
    """
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
