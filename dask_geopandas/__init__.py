from ._version import get_versions

from .expr import (
    points_from_xy,
    from_wkt,
    from_wkb,
    GeoDataFrame,
    GeoSeries,
    from_geopandas,
    from_dask_dataframe,
)
from .io.file import read_file
from .io.parquet import read_parquet, to_parquet
from .io.arrow import read_feather, to_feather
from .clip import clip
from .sjoin import sjoin
from . import backends as _  # needed to register dispatch functions with dask


__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "GeoDataFrame",
    "GeoSeries",
    "clip",
    "from_dask_dataframe",
    "from_geopandas",
    "from_wkb",
    "from_wkt",
    "points_from_xy",
    "read_feather",
    "read_file",
    "read_parquet",
    "sjoin",
    "to_feather",
    "to_parquet",
]

from . import _version

__version__ = _version.get_versions()["version"]
