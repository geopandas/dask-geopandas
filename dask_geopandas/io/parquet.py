from functools import partial

import geopandas

import dask.dataframe as dd

from .arrow import (
    DASK_2022_12_0_PLUS,
    DASK_2023_03_2_DEV,
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


class GeoArrowEngine(GeoDatasetEngine, DaskArrowDatasetEngine):
    """
    Engine for reading geospatial Parquet datasets. Subclasses dask's
    ArrowEngine for Parquet, but overriding some methods to ensure we
    correctly read/write GeoDataFrames.

    """

    @classmethod
    def read_metadata(cls, fs, paths, **kwargs):
        meta, stats, parts, index = super().read_metadata(fs, paths, **kwargs)

        gather_spatial_partitions = kwargs.pop("gather_spatial_partitions", True)

        if gather_spatial_partitions:
            regions = geopandas.GeoSeries(
                [_get_partition_bounds_parquet(part, fs) for part in parts],
                crs=meta.crs,
            )
            if regions.notna().all():
                # a bit hacky, but this allows us to get this passed through
                meta.attrs["spatial_partitions"] = regions

        return (meta, stats, parts, index)

    @classmethod
    def _update_meta(cls, meta, schema):
        """
        Convert meta to a GeoDataFrame and update with potential GEO metadata
        """
        return _update_meta_to_geodataframe(meta, schema.metadata)

    @classmethod
    def _generate_dd_meta(cls, schema, index, categories, partition_info):
        """Overriding private method for dask < 2021.10.0"""
        meta, index_cols, categories, index, partition_info = super()._generate_dd_meta(
            schema, index, categories, partition_info
        )
        meta = cls._update_meta(meta, schema)
        return meta, index_cols, categories, index, partition_info

    @classmethod
    def _create_dd_meta(cls, dataset_info, use_nullable_dtypes=False):
        """Overriding private method for dask >= 2021.10.0"""
        if DASK_2022_12_0_PLUS and not DASK_2023_03_2_DEV:
            meta = super()._create_dd_meta(dataset_info, use_nullable_dtypes)
        else:
            meta = super()._create_dd_meta(dataset_info)

        schema = dataset_info["schema"]
        if not schema.names and not schema.metadata:
            if len(list(dataset_info["ds"].get_fragments())) == 0:
                raise ValueError(
                    "No dataset parts discovered. Use dask.dataframe.read_parquet "
                    "to read it as an empty DataFrame"
                )
        meta = cls._update_meta(meta, schema)
        return meta


to_parquet = partial(dd.to_parquet, engine=GeoArrowEngine)
to_parquet.__doc__ = dd.to_parquet.__doc__


def read_parquet(*args, **kwargs):
    result = dd.read_parquet(*args, engine=GeoArrowEngine, **kwargs)
    # check if spatial partitioning information was stored
    spatial_partitions = result._meta.attrs.get("spatial_partitions", None)
    result.spatial_partitions = spatial_partitions
    return result


read_parquet.__doc__ = dd.read_parquet.__doc__
