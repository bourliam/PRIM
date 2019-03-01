import pandas as pd
from pymongo import MongoClient


class SpeedMatrix:
    """
    The class to compute the speed matrix.   
     Arguments:
        db_name {str} -- The name of the database
        collection_name {str} -- The name of the collection
    Keyword Arguments:
        db_osm_name {str} -- The name of the osm database  (default: {"osm"})
        roads_name {str} -- The name of the roads collection (default: {"roads"}) 
    """

    def __init__(self, db_name, collection_name, db_osm_name="osm", roads_name="roads"):
        """Class to compute the speed matrix
        Arguments:
            db_name {str} -- The name of the database
            collection_name {str} -- The name of the collection
        Keyword Arguments:
            db_osm_name {str} -- The name of the osm database  (default: {"osm"})
            roads_name {str} -- The name of the roads collection (default: {"roads"})
        """

        client = MongoClient()
        self.collection = client[db_name][collection_name]
        self.osm_roads = client[db_osm_name][roads_name]
        self.avg_speed = pd.DataFrame()

    def get_main_roads(self):
        """Deprecated : Use the function in source/OsmProcessing.
        Function to get the main roads.
        Returns:
            list -- The list of road ids.
        """

        curs = self.osm_roads.find(
            {
                "loc": {
                    "$geoIntersects": {
                        "$geometry": {
                            "type": "Polygon",
                            "coordinates": [
                                [[-1.5460, 48.1656],
                                 [-1.5460, 48.0632],
                                    [-1.7626, 48.0632],
                                 [-1.7626, 48.1656],
                                 [-1.5460, 48.1656]]
                            ]
                        }
                    }
                },
                "tag.k": "highway",
                "tag.v": {
                    "$in": [
                        "motorway",
                        "trunk",
                        "primary",
                        "secondary",
                        "tertiary",
                        "motorway_link",
                        "trunk_link",
                        "primary_link",
                        "secondary_link",
                        "tertiary_link"
                    ]
                }
            },
            {'_id': 1}
        )

        main_roads = [r['_id'] for r in curs]

        return main_roads

    def get_speed_matrix(self, timeframe, road_ids, hour_start=17, hour_end=20):
        """The function to get the speed matrix
        Arguments:
            timeframe {int} -- The length of the timeframe, ie the time between two columns.
            road_ids {list} -- List of the roads to build the matrix.
        Keyword Arguments:
            hour_start {int} -- The start hour each day (default: {17})
            hour_end {int} -- The end hour each day (default: {20})
        Returns:
            pandas.DataFrame -- The speed matrix. A row is for a road. A column is for a time t.
        """

        self.avg_speed = pd.DataFrame(
            list(self.collection.aggregate(
                [
                    {"$addFields": {
                        "hour": {"$hour": {"$toDate": {"$multiply": ["$time", 1000]}}}}},
                    {"$match": {"matching_road": {"$in": road_ids},
                                "hour": {"$gte": hour_start, "$lt": hour_end}}},

                    {
                        "$group": {
                            "_id": {
                                "matching_road": "$matching_road",
                                "heading": "$heading_road",
                                "time": {
                                    "$toDate": {

                                        "$multiply": [
                                            {"$subtract": [
                                                {"$toLong": "$time"},
                                                {"$mod": [
                                                    {"$toLong": "$time"}, 1 * 60 * 15]}
                                            ]}, 1000]

                                    }
                                }
                            },
                            "avg_speed": {"$avg": "$speed"},
                            "count": {"$sum": 1},
                            "matching_road": {"$first": {"$concat": [{"$toString": "$matching_road"}, "_", {"$toString": "$heading_road"}]}},
                        }
                    }], allowDiskUse=True
            ))
        )
        self.avg_speed['time'] = self.avg_speed['_id'].apply(
            lambda x: x['time'])
        countsDF=self.avg_speed.pivot(index='matching_road', columns='time', values='count')

        self.avg_speed = self.avg_speed.pivot(
            index='matching_road', columns='time', values='avg_speed')
        return self.avg_speed,countsDF