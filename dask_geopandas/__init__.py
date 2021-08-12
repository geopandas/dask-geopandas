from ._version import get_versions

from . import backends
from .core import (
    points_from_xy,
    GeoDataFrame,
    GeoSeries,
    from_geopandas,
    from_dask_dataframe,
)
from .io.parquet import read_parquet, to_parquet
from .io.arrow import read_feather, to_feather
from .sjoin import sjoin


__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "points_from_xy",
    "GeoDataFrame",
    "GeoSeries",
    "from_geopandas",
    "from_dask_dataframe",
    "read_feather",
    "read_parquet",
    "to_feather" "to_parquet",
    "sjoin",
]
