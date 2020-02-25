from dask.dataframe.core import get_parallel_type, make_meta
from dask.dataframe.extensions import make_array_nonempty, make_scalar
import numpy as np
import shapely.geometry
from shapely.geometry.collection import GeometryCollection
from shapely.geometry.base import BaseGeometry
import geopandas
from geopandas.array import GeometryDtype, from_shapely

from .core import GeoSeries, GeoDataFrame

get_parallel_type.register(geopandas.GeoDataFrame, lambda _: GeoDataFrame)
get_parallel_type.register(geopandas.GeoSeries, lambda _: GeoSeries)


@make_meta.register((GeometryCollection, BaseGeometry))
def make_meta_shapely_geometry(x, index=None):
    return x


@make_array_nonempty.register(GeometryDtype)
def _(dtype):
    a = np.array([shapely.geometry.Point(i, i) for i in range(2)], dtype=object)
    return from_shapely(a)


@make_scalar.register(GeometryDtype.type)
def _(x):
    return shapely.geometry.Point(0, 0)
