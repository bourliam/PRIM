
from pymongo import MongoClient, UpdateOne
from pymongo.errors import BulkWriteError
from pprint import pprint
import time


class Affect_road_to_point:
  """
  Class to affect osm roads to the points.
  params:
  @collection: the pymongo collection containing the points.
  @osm_roads: the pymongo collection with the osm roads.
  @maxDistance: the max distance between the point and the road (meters).
  """

  def __init__(self, collection, osm_roads, maxDistance):
    #client = MongoClient()
    #geo4cast = client.geo4cast
    #subset_rennes = geo4cast.subset_rennes

    #osm_rennes = client["osm-rennes"]
    #roads = osm_rennes.roads

    self.collection = collection
    self.osm_roads = osm_roads
    self.maxDistance = maxDistance

  def affect(self):
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

      # result = self.collection.update_one({'_id': point['_id']}, {"$set": {'matching_road': road['_id']}})
      # count += result.modified_count
      # print(count, "/", nbPoints, end='\r')

    if count % 1000 != 0:
      self.collection.bulk_write(requests)
    
    end = time.time()
    print("\nDone.")
    print(count, "points modified /", nbPoints, "in", round(end - start, 3) ,"seconds")
  

if __name__ == "__main__":
    

    client = MongoClient()
    collection = client.geo4cast.subset_rennes
    osm_roads = client['osm-rennes'].roads
   
    affect = Affect_road_to_point(collection,osm_roads, 10)
    
    affect.affect()
