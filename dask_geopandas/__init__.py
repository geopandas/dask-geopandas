from ._version import get_versions

from . import backends
from .clip import clip
from .core import (
    points_from_xy,
    GeoDataFrame,
    GeoSeries,
    from_geopandas,
    from_dask_dataframe,
)
from .io.parquet import read_parquet, to_parquet
from .sjoin import sjoin


__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "points_from_xy",
    "GeoDataFrame",
    "GeoSeries",
    "from_geopandas",
    "from_dask_dataframe",
    "read_parquet",
    "to_parquet",
    "clip",
    "sjoin",
]
