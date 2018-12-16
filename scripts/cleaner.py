from pymongo import MongoClient
import numpy as np

def remove_aberrant(data:int, timeframe = 15, nb_points = 5):
  """[A function to remove aberrant points in a collection.]
  
  A point is abberant when we have nb_points in timeframe minutes with speed=0 and the same user id.

  Arguments:
    data {int} -- [The collection to clean]
  
  Keyword Arguments:
    timeframe {int} -- [The timeframe we want to use] (default: {15})
    nb_points {int} -- [The min number of points to consider] (default: {5})
  """
 
  grouped = list(data.aggregate([
    { "$match": { "speed": 0 } },
    { "$group": {
      "_id": {
        "matching_road": "$matching_road",
        "id": "$id",
        "time": {
          "$toDate": {
            "$subtract": [
              { "$toLong": "$time"},
              { "$mod": [ { "$toLong": "$time"}, 1000 * 60 * timeframe] }
            ]
          }
        }
      },
      "ids": { "$push": "$_id"},
      "count": {"$sum": 1} 
    }},
    { "$match": { "count": { "$gt": nb_points } } }
  ]))


  ids = []
  for g in grouped:
    ids += g['ids'][1:]
  ids 

  res = data.delete_many({'_id': { "$in": ids}})
  print(res.deleted_count, "points deleted")
