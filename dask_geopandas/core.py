import warnings

from .expr import *  # noqa: F403

warnings.warn(
    "dask_geopandas.core is deprecated and will be removed in a future version.",
    category=FutureWarning,
    stacklevel=1,
)
