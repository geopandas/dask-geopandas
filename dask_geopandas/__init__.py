from ._version import get_versions

from . import backends

if backends.QUERY_PLANNING_ON:
    from .expr import (
        points_from_xy,
        from_wkt,
        from_wkb,
        GeoDataFrame,
        GeoSeries,
        from_geopandas,
        from_dask_dataframe,
    )
else:
    from .core import (
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
from .overlay import overlay


__version__ = get_versions()["version"]
del get_versions

__all__ = [
    "points_from_xy",
    "from_wkt",
    "from_wkb",
    "GeoDataFrame",
    "GeoSeries",
    "from_geopandas",
    "from_dask_dataframe",
    "read_file",
    "read_feather",
    "read_parquet",
    "to_feather",
    "to_parquet",
    "clip",
    "sjoin",
    "overlay",
]

from . import _version

__version__ = _version.get_versions()["version"]

from . import _version

__version__ = _version.get_versions()["version"]
