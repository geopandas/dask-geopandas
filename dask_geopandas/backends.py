from dask.dataframe.core import get_parallel_type, make_meta
from dask.dataframe.utils import (
    meta_nonempty,
    _nonempty_index,
    meta_nonempty_dataframe,
)
from dask.dataframe.extensions import make_array_nonempty, make_scalar
import numpy as np
import shapely.geometry
from shapely.geometry.base import BaseGeometry
import geopandas
from geopandas.array import GeometryDtype, from_shapely

from .core import GeoSeries, GeoDataFrame

get_parallel_type.register(geopandas.GeoDataFrame, lambda _: GeoDataFrame)
get_parallel_type.register(geopandas.GeoSeries, lambda _: GeoSeries)


@make_meta.register(BaseGeometry)
def make_meta_shapely_geometry(x, index=None):
    return x


@make_array_nonempty.register(GeometryDtype)
def _(dtype):
    a = np.array(
        [shapely.geometry.LineString([(i, i), (i, i + 1)]) for i in range(2)],
        dtype=object,
    )
    return from_shapely(a)


@make_scalar.register(GeometryDtype.type)
def _(x):
    return shapely.geometry.Point(0, 0)


@meta_nonempty.register(geopandas.GeoSeries)
def _nonempty_geoseries(x, idx=None):
    if idx is None:
        idx = _nonempty_index(x.index)
    data = make_array_nonempty(x.dtype)
    return geopandas.GeoSeries(data, name=x.name, crs=x.crs)


@meta_nonempty.register(geopandas.GeoDataFrame)
def _nonempty_geodataframe(x):
    df = meta_nonempty_dataframe(x)
    return geopandas.GeoDataFrame(df, crs=x.crs)


@make_meta.register((geopandas.GeoSeries, geopandas.GeoDataFrame))
def make_meta_geodataframe(df, index=None):
    return df.head(0)
