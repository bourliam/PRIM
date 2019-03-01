from pymongo import MongoClient
# mongo client
client = MongoClient("mongodb://mbouchouia:cbf20Li34!@mongodb-tp.enst.fr")
# coyote data "db.coyote"
coyoteData = client.geolytics.coyote
# OsmWaysData
osmWays= client.geolytics.ways
