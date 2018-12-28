from pymongo import MongoClient
import numpy as np
from multiprocessing import Process, Manager



def remove_aberrant(data, nb_workers, timeframe = 15, nb_points = 5):
  """[A function to remove aberrant points in a collection.]
  
  A point is abberant when we have nb_points in timeframe minutes with speed=0 and the same user id.

  Arguments:
    data {int} -- [The collection to clean]
    nb_workers {int} -- [The number of process to make for the deletion]
  
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
  ], allowDiskUse=True))


  ids = []
  for g in grouped:
    ids += g['ids'][1:]

  nb_ids = len(ids)
  bulk_size = int(np.ceil(nb_ids / nb_workers))
  
  def delete_para(proc_num, ids):
    res = data.delete_many({'_id': { "$in": ids}})
    return_dict[proc_num] = res.deleted_count

  manager = Manager()
  return_dict = manager.dict()

  processes = [ Process(target=delete_para, args=(k, ids[k*bulk_size:(k+1)*bulk_size])) for k in range(nb_workers) ]

  for p in processes:
    p.start()

  for p in processes:
    p.join()

  print(sum(return_dict.values()), "points deleted")
