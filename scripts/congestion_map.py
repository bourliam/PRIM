from folium.plugins import MarkerCluster
from folium.plugins import FastMarkerCluster
import folium

from pymongo import MongoClient
import numpy as np
import pandas as pd


class CongestionMap:
    """The class to print or save the map of the congestion

    Returns:
        folimum_map -- A folium map of the congestion.
    """

    def __init__(self):
        """Initialistion of the class, fetching the osm collection.
        """

        client = MongoClient()
        self.roads = client.osm.roads

    def get_speed_limit(self, row):
        """Function to get the speed limit of a row.

        Arguments:
            row {pandas Series} -- The row of the matrix for which we look for the speed limit.

        Returns:
            string -- The speed limit of the road. If no speed limit returns False.
        """

        road_id = int(row.name.split('_')[0])
        r = self.roads.find_one({"_id": road_id})
        if 'maxspeed' in r['key']:
            return [item.get('v') for item in r['tag'] if item.get('k') == 'maxspeed'][0]

        return False

    def compute_congestion(self, row):
        """Function to compute the congestion indice

        Arguments:
            row {pandas Series} -- The row of the matrix for which we compute the congestion

        Returns:
            float -- The indice of congestion. Returns -1 if not possible to compute.
        """

        speed, limit = row[0], row[1]
        if np.isnan(speed):
            return -1.0
        if limit == False:
            return -1.0
        try:
            c = float(speed) / float(limit)
        except ValueError:
            c = -1.0
        return c

    def congestion_level(self, C):
        """Function to compute the congestion level

        Arguments:
            C {float} -- The congestion indice

        Returns:
            int -- Returns -1, 0, 1 or 2, depending on the congestion.
        """

        if C == -1:    # We don't know => gray
            return -1
        if C < 0.25:   # Speed is 25% of limit => red
            return 0
        if C < 0.70:   # Speed is in [25%, 70%] => orange
            return 1
        return 2       # Speed is high enough => green

    def roads_by_congestion(self, M, congestion):
        """Returns the list of the roads' ids depending on the wanted congestion level

        Arguments:
            M {pandas DataFrame} -- The DataFrame we are working on. 
            congestion {int} -- The level of congestion we want for the roads.

        Returns:
            list -- The list of the roads ids
        """

        return list(set([r.split('_')[0] for r in list(M[M['Congestion_level'] == congestion].index.values)]))

    def coords_lists(self, road_list):
        """Function to get the lists of coordinates of roads.

        Arguments:
            road_list {list} -- The list of roads ids for which we want the coordinates.

        Returns:
            list -- A list of lists of coordinates. 
        """

        return [self.roads.find({'_id': int(i)})[0]['loc']['coordinates'] for i in road_list]

    def add_color_roads_to_map(self, f_map, coords, color):
        """Function to add the color roads to the map

        Arguments:
            f_map {folium.Map} -- The map on which we want to add the roads.
            coords {list} -- The list of lists of coordinates to plot.
            color {string} -- The color to use.
        """

        [folium.PolyLine(locations=[[lo[1], lo[0]] for lo in x],
                         color=color).add_to(f_map) for x in coords]

    def get_congestion_map(self, matrix, save=False, save_name="congestion_map"):
        """Function to create a map of the congestion

        Arguments:
            matrix {pandas Series/DataFrame} -- The speeds at the time we want to create the map.

        Keyword Arguments:
            save {bool} -- If True saves the map. (default: {False})
            save_name {str} -- The name of the map if saved. (default: {"congestion_map"})

        Returns:
            folium.Map -- The map of the congestion
        """

        M = pd.DataFrame(matrix)

        M['limit'] = M.apply(self.get_speed_limit, axis=1)
        print("Routes sans vitesse limite:", (M['limit'] == False).sum())

        M['C'] = M.apply(self.compute_congestion, axis=1)

        M['Congestion_level'] = M['C'].apply(self.congestion_level)

        print(M.describe())
        print("\n===========\n")
        print(M['Congestion_level'].value_counts())

        coords_green = self.coords_lists(self.roads_by_congestion(M, 2))
        coords_orange = self.coords_lists(self.roads_by_congestion(M, 1))
        coords_red = self.coords_lists(self.roads_by_congestion(M, 0))
        coords_gray = self.coords_lists(self.roads_by_congestion(M, -1))

        folium_map = folium.Map(
            location=[48.111952, -1.679330], zoom_start=12, tiles="OpenStreetMap")

        self.add_color_roads_to_map(folium_map, coords_green, 'green')
        self.add_color_roads_to_map(folium_map, coords_orange, 'orange')
        self.add_color_roads_to_map(folium_map, coords_red, 'red')
        self.add_color_roads_to_map(folium_map, coords_gray, 'gray')

        if save:
            folium_map.save(save_name + '.html')

        return folium_map
