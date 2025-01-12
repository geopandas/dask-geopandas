from functools import partial

import dask.dataframe as dd

import geopandas

from .arrow import (
    GeoDatasetEngine,
    _get_partition_bounds,
    _update_meta_to_geodataframe,
)

try:
    # pyarrow is imported here, but is an optional dependency
    from dask.dataframe.io.parquet.arrow import (
        ArrowDatasetEngine as DaskArrowDatasetEngine,
    )
except ImportError:
    DaskArrowDatasetEngine = object


def _get_partition_bounds_parquet(part, fs):
    """
    Based on the part information gathered by dask, get the partition bounds
    if available.

    """
    from pyarrow.parquet import ParquetFile

    # read the metadata from the actual file (this is again file IO, but
    # we can't rely on the schema metadata, because this is only the
    # metadata of the first piece)
    pq_metadata = None
    if "piece" in part:
        path = part["piece"][0]
        if isinstance(path, str):
            with fs.open(path, "rb") as f:
                pq_metadata = ParquetFile(f).metadata
    if pq_metadata is None:
        return None

    return _get_partition_bounds(pq_metadata.metadata)


def _convert_to_list(column) -> list | None:
    if column is None or isinstance(column, list):
        pass
    elif isinstance(column, tuple):
        column = list(column)
    elif hasattr(column, "dtype"):
        column = column.tolist()
    else:
        column = [column]
    return column


class GeoArrowEngine(GeoDatasetEngine, DaskArrowDatasetEngine):
    """
    Engine for reading geospatial Parquet datasets. Subclasses dask's
    ArrowEngine for Parquet, but overriding some methods to ensure we
    correctly read/write GeoDataFrames.

    """

    @classmethod
    def _update_meta(cls, meta, schema):
        """
        Convert meta to a GeoDataFrame and update with potential GEO metadata
        """
        return _update_meta_to_geodataframe(meta, schema.metadata)

    @classmethod
    def _create_dd_meta(cls, dataset_info):
        meta = super()._create_dd_meta(dataset_info)
        schema = dataset_info["schema"]
        if not schema.names and not schema.metadata:
            if len(list(dataset_info["ds"].get_fragments())) == 0:
                raise ValueError(
                    "No dataset parts discovered. Use dask.dataframe.read_parquet "
                    "to read it as an empty DataFrame"
                )
        meta = cls._update_meta(meta, schema)

        if dataset_info["kwargs"].get("gather_spatial_partitions", True):
            fs = dataset_info["fs"]
            parts, _, _ = cls._construct_collection_plan(dataset_info)
            regions = geopandas.GeoSeries(
                [_get_partition_bounds_parquet(part, fs) for part in parts],
                crs=meta.crs,
            )
            if regions.notna().all():
                # a bit hacky, but this allows us to get this passed through
                meta.attrs["spatial_partitions"] = regions

        return meta


to_parquet = partial(dd.to_parquet, engine=GeoArrowEngine)
to_parquet.__doc__ = dd.to_parquet.__doc__


def read_parquet(*args, **kwargs):
    from dask.dataframe import read_parquet

    result = read_parquet(*args, engine=GeoArrowEngine, **kwargs)
    # check if spatial partitioning information was stored
    spatial_partitions = result._meta.attrs.get("spatial_partitions", None)

    result = dd.from_graph(
        result.dask,
        result._meta,
        result.divisions,
        result.__dask_keys__(),
        "read_parquet",
    )

    result.spatial_partitions = spatial_partitions
    return result


read_parquet.__doc__ = dd.read_parquet.__doc__
