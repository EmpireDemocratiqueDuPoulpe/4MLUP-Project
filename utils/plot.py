# #################################################################################################################### #
#       plot.py                                                                                                        #
#           Various functions to plot.                                                                                 #
# #################################################################################################################### #

import os
import time
import webbrowser
import pandas
import geopandas
import folium
from folium import plugins

folium_data = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"


class Map:
    TILE_LAYERS = [
        {"name": "openstreetmap", "display_name": "Open Street Map"},
        {"name": "stamentoner", "display_name": "Stamen toner"},
        {"name": "cartodbpositron", "display_name": "CartoDB (Light)"},
        {"name": "cartodbdark_matter", "display_name": "CartoDB (Dark)"},
    ]

    def __init__(self, crs: str = "EPSG:3857", **kwargs):
        self.crs = crs
        self.map = folium.Map(tiles=None, crs=self.crs.replace(":", ""), **kwargs)
        self.layers = []

    def _add_tile_layers(self):
        for layer in Map.TILE_LAYERS:
            folium.TileLayer(layer["name"], name=layer["display_name"]).add_to(self.map)

    def _add_map_layers(self):
        for layer in self.layers:
            for sublayer in layer.get_layers():
                sublayer.add_to(self.map)

    def _register_layer(self, layer):
        if isinstance(layer, MapLayer):
            self.layers.append(layer)

    def fit_bounds(self, south_west, north_east):
        self.map.fit_bounds([south_west, north_east])

    def open(self, notebook: bool = False, output_dir: str = "./temp", filename: str = None):
        self._add_tile_layers()
        self._add_map_layers()
        folium.LayerControl().add_to(self.map)

        if notebook:
            return self.map

        path = os.path.join(output_dir, (filename if filename else f"map-{time.time()}.html"))

        self.map.save(path)
        webbrowser.open(path)


class MapLayer:
    DATA_TYPES = {"DataFrame": 0, "Geo[GCS]": 1, "TimedGeo[GCS]": 2}

    def __init__(self, name: str, show_default: bool = False):
        self.name = name
        self.parent_map = None
        self.feature_group = folium.FeatureGroup(self.name, overlay=True, show=show_default)
        self.layers = []

        self.data = None
        self.data_type = False

    def get_layers(self):
        return self.layers

    def add_to(self, m: Map):
        if isinstance(m, Map):
            self.parent_map = m
            # noinspection PyProtectedMember
            self.parent_map._register_layer(self)

        return self

    def load_dataframe(self, data: pandas.DataFrame):
        self.data = data
        self.data_type = MapLayer.DATA_TYPES["DataFrame"]

        return self

    def load_gcs_data(self, data: pandas.DataFrame, col_names: dict = None, time_column: str = None):
        if time_column is None:
            col_names = col_names if col_names else {"lat": "Latitude", "lon": "Longitude"}
            data_coords = data[[col_names["lat"], col_names["lon"]]].dropna(axis=0, how="any")
            geometry = geopandas.points_from_xy(data_coords[col_names["lon"]], data_coords[col_names["lat"]])

            self.data = geopandas.GeoDataFrame(data, geometry=geometry, crs=self.parent_map.crs)
            self.data_type = MapLayer.DATA_TYPES["Geo[GCS]"]
        else:
            data_coords = data[[time_column, col_names["lat"], col_names["lon"]]].dropna(axis=0, how="any")
            data_dates = data[time_column].drop_duplicates()
            data_timed = {}

            for _, d in data_dates.iteritems():
                data_timed[d.date().__str__()] = data_coords.loc[data_coords[time_column] == d][
                    [col_names["lat"], col_names["lon"]]].values.tolist()

            self.data = data_timed
            self.data_type = MapLayer.DATA_TYPES["TimedGeo[GCS]"]

        return self

    def _add_to_layer(self, item):
        item.add_to(self.feature_group)
        self.layers.append(item)

    def to_choropleth(self, key_on: str = None, fill_color: str = None, **kwargs):
        if not self.data_type == MapLayer.DATA_TYPES["DataFrame"]:
            raise RuntimeError("MapLayer: to_choropleth() is only available for pandas dataframes.")

        full_kwargs = {
            "key_on": "feature.id", "fill_color": "YlOrRd", "fill_opacity": 0.7, "line_opacity": 0.2, **kwargs
        }

        choropleth = folium.Choropleth(data=self.data, **full_kwargs)
        self._add_to_layer(choropleth)

        return self