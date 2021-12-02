from functools import partial
import json
from typing import TYPE_CHECKING

import pandas as pd

import geopandas
import shapely.geometry

import dask.dataframe as dd

try:
    # pyarrow is imported here, but is an optional dependency
    from dask.dataframe.io.parquet.arrow import ArrowEngine
except ImportError:
    ArrowEngine = object

if TYPE_CHECKING:
    import pyarrow


def _get_partition_bounds(part, fs):
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


class GeoArrowEngine(ArrowEngine):
    @classmethod
    def read_metadata(cls, fs, paths, **kwargs):
        meta, stats, parts, index = super().read_metadata(fs, paths, **kwargs)

        # get spatial partitions if available
        regions = geopandas.GeoSeries(
            [_get_partition_bounds(part, fs) for part in parts], crs=meta.crs
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
        if schema.metadata and b"geo" in schema.metadata:
            geo_meta = json.loads(schema.metadata[b"geo"])
            geometry_column_name = geo_meta["primary_column"]
            crs = geo_meta["columns"][geometry_column_name]["crs"]
            geometry_columns = geo_meta["columns"]
        else:
            # TODO we could allow the user to pass those explicitly if not
            # stored in the metadata
            geometry_column_name = None
            crs = None
            geometry_columns = {}

        # Update meta to be a GeoDataFrame
        meta = geopandas.GeoDataFrame(meta, geometry=geometry_column_name, crs=crs)
        for col, item in geometry_columns.items():
            if not col == meta._geometry_column_name:
                meta[col] = geopandas.GeoSeries(meta[col], crs=item["crs"])

        return meta

    @classmethod
    def _generate_dd_meta(cls, schema, index, categories, partition_info):
        """Overriding private method for dask < 2021.10.0"""
        meta, index_cols, categories, index, partition_info = super()._generate_dd_meta(
            schema, index, categories, partition_info
        )
        meta = cls._update_meta(meta, schema)
        return meta, index_cols, categories, index, partition_info

    @classmethod
    def _create_dd_meta(cls, dataset_info):
        """Overriding private method for dask >= 2021.10.0"""
        meta = super()._create_dd_meta(dataset_info)
        schema = dataset_info["schema"]
        meta = cls._update_meta(meta, schema)
        return meta

    @classmethod
    def _arrow_table_to_pandas(
        cls, arrow_table: "pyarrow.Table", categories, **kwargs
    ) -> pd.DataFrame:
        from geopandas.io.arrow import _arrow_to_geopandas

        _kwargs = kwargs.get("arrow_to_pandas", {})
        _kwargs.update({"use_threads": False, "ignore_metadata": False})

        # TODO support additional keywords
        try:
            return _arrow_to_geopandas(arrow_table)
        except ValueError as err:
            # when no geometry column is selected, the above will error.
            # We want to fallback to reading it as a plain dask object, because
            # the column selection can be an automatic pushdown (eg `ddf['col']`)
            # TODO more robust detection of when to fall back?
            if "No geometry columns are included" in str(err):
                return super()._arrow_table_to_pandas(
                    arrow_table, categories=categories, **kwargs
                )
            else:
                raise

    @classmethod
    def _pandas_to_arrow_table(
        cls, df: pd.DataFrame, preserve_index=False, schema=None
    ) -> "pyarrow.Table":
        from geopandas.io.arrow import _geopandas_to_arrow

        # TODO add support for schema
        if schema is not None:
            raise NotImplementedError("Passing 'schema' is not yet supported")

        table = _geopandas_to_arrow(df, index=preserve_index)
        return table


to_parquet = partial(dd.to_parquet, engine=GeoArrowEngine)
to_parquet.__doc__ = dd.to_parquet.__doc__


def read_parquet(*args, **kwargs):
    result = dd.read_parquet(*args, engine=GeoArrowEngine, **kwargs)
    # check if spatial partitioning information was stored
    spatial_partitions = result._meta.attrs.get("spatial_partitions", None)
    result.spatial_partitions = spatial_partitions
    return result


read_parquet.__doc__ = dd.read_parquet.__doc__
