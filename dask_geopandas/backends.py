import uuid

from dask.dataframe.core import get_parallel_type
from dask.dataframe.utils import (
    meta_nonempty,
    meta_nonempty_dataframe,
)
from dask.dataframe.extensions import make_array_nonempty, make_scalar
from dask.base import normalize_token

try:
    from dask.dataframe.utils import make_meta_util as make_meta
except ImportError:
    from dask.dataframe.core import make_meta

try:
    from dask.dataframe.backends import _nonempty_index
except:
    from dask.dataframe.utils import _nonempty_index

import shapely.geometry
from shapely.geometry.base import BaseGeometry
import geopandas
from geopandas.array import GeometryArray, GeometryDtype, from_shapely

from .core import GeoSeries, GeoDataFrame

get_parallel_type.register(geopandas.GeoDataFrame, lambda _: GeoDataFrame)
get_parallel_type.register(geopandas.GeoSeries, lambda _: GeoSeries)


@make_meta.register(BaseGeometry)
def make_meta_shapely_geometry(x, index=None):
    return x


@make_array_nonempty.register(GeometryDtype)
def _(dtype):
    return from_shapely(
        [shapely.geometry.LineString([(i, i), (i, i + 1)]) for i in range(2)]
    )


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


@normalize_token.register(GeometryArray)
def tokenize_geometryarray(x):
    # TODO if we can find an efficient hashing function (eg hashing integer
    # pointers on the C level?), we could replace this random uuid
    return uuid.uuid4().hex
