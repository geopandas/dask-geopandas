from dask.dataframe.core import get_parallel_type, make_meta

from shapely.geometry.collection import GeometryCollection
from shapely.geometry.base import BaseGeometry
import geopandas

from .core import GeoSeries, GeoDataFrame

get_parallel_type.register(geopandas.GeoDataFrame, lambda _: GeoDataFrame)
get_parallel_type.register(geopandas.GeoSeries, lambda _: GeoSeries)


@make_meta.register((GeometryCollection, BaseGeometry))
def make_meta_shapely_geometry(x, index=None):
    return x
