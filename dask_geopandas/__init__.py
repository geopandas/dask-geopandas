from ._version import get_versions

from . import backends
from .core import (
    points_from_xy,
    GeoDataFrame,
    GeoSeries,
    from_geopandas,
    from_dask_dataframe,
)

__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "points_from_xy",
    "GeoDataFrame",
    "GeoSeries",
    "from_geopandas",
    "from_dask_dataframe",
]
