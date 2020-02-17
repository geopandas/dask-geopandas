from dask.dataframe.core import get_parallel_type

import geopandas

from .dataframe import GeoSeries, GeoDataFrame

get_parallel_type.register(geopandas.GeoDataFrame, lambda _: GeoDataFrame)
get_parallel_type.register(geopandas.GeoSeries, lambda _: GeoSeries)
