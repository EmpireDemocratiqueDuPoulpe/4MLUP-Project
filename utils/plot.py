# #################################################################################################################### #
#       plot.py                                                                                                        #
#           Various functions to plot.                                                                                 #
# #################################################################################################################### #

import os
import time
import webbrowser
import numpy
import pandas
import geopandas
import folium
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from wordcloud import WordCloud
from matplotlib import pyplot
from scipy.cluster import hierarchy

folium_data = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data"


# #### Map & MapLayer ##################################################################################################
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


# #### PCA #############################################################################################################
def pca(data: pandas.DataFrame, cluster_col: str, title: str = None):
    data_pca = pandas.DataFrame(PCA(n_components=2).fit_transform(data), columns=["pca1", "pca2"])
    pyplot.scatter(data_pca["pca1"], data_pca["pca2"], c=data[cluster_col], s=50)

    if title is not None:
        pyplot.title(title)

    pyplot.show()


# #### Wordcloud #######################################################################################################
def generate_wordcloud(word_list: list, title: str = "Word cloud", language: str = "english"):
    words = " ".join(word for word in word_list)
    cloud = WordCloud(
        background_color="white", colormap="winter",
        width=1600, height=800,
        max_words=50, max_font_size=200,
        stopwords=stopwords.words(language), normalize_plurals=True
    ).generate(words)

    pyplot.figure(figsize=(12, 10))
    pyplot.title(title)
    pyplot.axis("off")
    pyplot.imshow(cloud, interpolation="bilinear")
    pyplot.show()


# #### Dendrogram ######################################################################################################
def dendrogram(model):
    counts = numpy.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        curr_count = 0

        for child_i in merge:
            if child_i < n_samples:
                curr_count += 1
            else:
                curr_count += counts[child_i - n_samples]

        counts[i] = curr_count

    linkage_matrix = numpy.column_stack([model.children_, model.distances_, counts]).astype(float)
    hierarchy.dendrogram(linkage_matrix)
    pyplot.show()

