from functools import partial
import json

import geopandas
import shapely.geometry

import dask.dataframe as dd

from .arrow import GeoDatasetEngine

try:
    # pyarrow is imported here, but is an optional dependency
    from dask.dataframe.io.parquet.arrow import ArrowEngine
except ImportError:
    ArrowEngine = object


def _get_partition_bounds(part):
    """
    Based on the part information gathered by dask, get the partition bounds
    if available.

    """
    from pyarrow.parquet import read_metadata

    # read the metadata from the actual file (this is again file IO, but
    # we can't rely on the schema metadata, because this is only the
    # metadata of the first piece)
    pq_metadata = None
    if "piece" in part:
        path = part["piece"][0]
        if isinstance(path, str):
            pq_metadata = read_metadata(path)
    if pq_metadata is None:
        return None

    metadata_str = pq_metadata.metadata.get(b"geo", None)
    if metadata_str is None:
        return None

    metadata = json.loads(metadata_str.decode("utf-8"))

    # for now only check the primary column (TODO generalize this to follow
    # the logic of geopandas to fallback to other geometry columns)
    geometry = metadata["primary_column"]
    bbox = metadata["columns"][geometry].get("bbox", None)
    if bbox is None:
        return None
    return shapely.geometry.box(*bbox)


class GeoArrowEngine(GeoDatasetEngine, ArrowEngine):
    @classmethod
    def read_metadata(cls, *args, **kwargs):
        meta, stats, parts, index = super().read_metadata(*args, **kwargs)

        # Update meta to be a GeoDataFrame
        # TODO convert columns based on GEO metadata (this will now only work
        # for a default "geometry" column)
        meta = geopandas.GeoDataFrame(meta)

        # get spatial partitions if available
        regions = geopandas.GeoSeries([_get_partition_bounds(part) for part in parts])
        if regions.notna().all():
            # a bit hacky, but this allows us to get this passed through
            meta.attrs["spatial_partitions"] = regions

        return (meta, stats, parts, index)


to_parquet = partial(dd.to_parquet, engine=GeoArrowEngine)
to_parquet.__doc__ = dd.to_parquet.__doc__


def read_parquet(*args, **kwargs):
    result = dd.read_parquet(*args, engine=GeoArrowEngine, **kwargs)
    # check if spatial partitioning information was stored
    spatial_partitions = result._meta.attrs.get("spatial_partitions", None)
    result.spatial_partitions = spatial_partitions
    return result


read_parquet.__doc__ = dd.read_parquet.__doc__
