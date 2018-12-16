from pymongo import MongoClient
import pprint
import numpy as np
import pandas as pd

client020 = MongoClient('srv020.it4pme.fr')
client142 = MongoClient('srv142.it4pme.fr')

departements_geo_coords = client142.geo4cast.departements_geo_coords

geomIlleEtVillaine = departements_geo_coords.find({'INSEE_code_dept': '35'})[0]['loc']

coyote = client020.coyote
collections = coyote.list_collection_names()
print(collections)

test = coyote.coyote_data_20181025.find(
    {'loc': {'$geoWithin': {'$geometry': {'type': "MultiPolygon", 'coordinates': geomIlleEtVillaine }}}}
)

pprint.pprint(test.next())
