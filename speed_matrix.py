import pandas as pd
from pymongo import MongoClient


class SpeedMatrix:
  """
  The class to compute the speed matrix.   
  @params   
  db_name: the name of the database to use.    
  collection_name: the name of the collection to use.  
  """

  def __init__(self, db_name, collection_name):
    client = MongoClient()
    self.collection = client[db_name][collection_name]
    self.avg_speed = pd.DataFrame()


  def get_speed_matrix_headings(self, timeframe):
    """
    The function to get the speed matrix   
    @params:    
    timeframe: the time interval between two speeds.
    """

    self.avg_speed = pd.DataFrame(
      list(self.collection.aggregate(
        [{
          "$group" : {
            "_id" : {
              "matching_road": "$matching_road",
              "heading": "$heading_road",
              "time": {
                "$toDate": {
                  "$subtract": [
                    { "$toLong": "$time"},
                    { "$mod": [ { "$toLong": "$time"}, 1000 * 60 * timeframe] }
                  ]
                }
              }
            },
            "avg_speed": {"$avg": "$speed"},
            "count": {"$sum": 1},
            "matching_road": {"$first": {"$concat": [{"$toString":"$matching_road"}, "_", {"$toString": "$heading_road"}]}},
          }
        }]
      ))
    )
    self.avg_speed['time'] = self.avg_speed['_id'].apply(lambda x: x['time'])
    self.avg_speed = self.avg_speed.pivot(index='matching_road', columns='time', values='avg_speed')
    return self.avg_speed
