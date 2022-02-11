import copy
import json
import math
from typing import TYPE_CHECKING

from dask.base import compute_as_if_collection, tokenize
from dask.dataframe.core import new_dd_object, Scalar
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key, apply

from dask.dataframe.io.utils import _get_pyarrow_dtypes, _meta_from_dtypes

import pandas as pd
import geopandas
import shapely.geometry

from fsspec.core import get_fs_token_paths


if TYPE_CHECKING:
    import pyarrow


def _update_meta_to_geodataframe(meta, schema_metadata):
    """
    Convert meta to a GeoDataFrame and update with potential GEO metadata
    """
    if schema_metadata and b"geo" in schema_metadata:
        geo_meta = json.loads(schema_metadata[b"geo"])
        geometry_column_name = geo_meta["primary_column"]
        crs = geo_meta["columns"][geometry_column_name]["crs"]
        geometry_columns = geo_meta["columns"]
    else:
        # TODO we could allow the user to pass those explicitly if not
        # stored in the metadata
        raise ValueError(
            "Missing geo metadata in the Parquet/Feather file. "
            "Use dask.dataframe.read_parquet/read_feather() instead."
        )

    # Update meta to be a GeoDataFrame
    meta = geopandas.GeoDataFrame(meta, geometry=geometry_column_name, crs=crs)
    for col, item in geometry_columns.items():
        if not col == meta._geometry_column_name:
            meta[col] = geopandas.GeoSeries(meta[col], crs=item["crs"])

    return meta


def _get_partition_bounds(schema_metadata):
    """
    Get the partition bounds, if available, for the dataset fragment.
    """
    if not (schema_metadata and b"geo" in schema_metadata):
        return None

    metadata = json.loads(schema_metadata[b"geo"].decode("utf-8"))

    # for now only check the primary column (TODO generalize this to follow
    # the logic of geopandas to fallback to other geometry columns)
    geometry = metadata["primary_column"]
    bbox = metadata["columns"][geometry].get("bbox", None)
    if bbox is None or all(math.isnan(val) for val in bbox):
        return None
    return shapely.geometry.box(*bbox)


class ArrowDatasetEngine:

    file_format: str

    @classmethod
    def read_metadata(cls, fs, paths, columns, filters, index):

        import pyarrow.dataset as ds
        from pyarrow.parquet import _filters_to_expression

        # dataset discovery
        # TODO support filesystems
        if len(paths) == 1:
            # list of 1 directory path is not supported
            paths = paths[0]
        dataset = ds.dataset(
            paths, partitioning="hive", filesystem=None, format=cls.file_format
        )

        # Get all (filtered) fragments
        if filters is not None:
            filter = _filters_to_expression(filters)
        else:
            filter = None

        fragments = list(dataset.get_fragments(filter=filter))

        # numeric rather than glob ordering
        # TODO how does this handle different partitioned directories?
        fragments = sorted(fragments, key=lambda f: natural_sort_key(f.path))

        # TODO potential splitting / aggregating of fragments

        # Create dask meta
        schema = dataset.schema
        # TODO add support for `categories`keyword
        dtypes = _get_pyarrow_dtypes(schema, categories=None)
        if columns is None:
            columns = list(dtypes)
        else:
            ex = set(columns) - set(dtypes)
            if ex:
                raise ValueError(f"Requested columns {ex} not in schema {set(dtypes)}")
        index = [index] if isinstance(index, str) else index
        meta = _meta_from_dtypes(columns, dtypes, index, [])
        return fragments, meta, schema, filter

    @classmethod
    def _arrow_table_to_pandas(
        cls, arrow_table: "pyarrow.Table", categories, **kwargs
    ) -> pd.DataFrame:
        _kwargs = kwargs.get("arrow_to_pandas", {})
        _kwargs.update({"use_threads": False, "ignore_metadata": False})

        return arrow_table.to_pandas(categories=categories, **_kwargs)

    @classmethod
    def read_partition(cls, fs, fragment, schema, columns, filter, **kwargs):

        table = fragment.to_table(
            schema=schema, columns=columns, filter=filter, use_threads=False
        )
        df = cls._arrow_table_to_pandas(table, None)
        return df

    @classmethod
    def write_partition(cls, df, path, fs, filename, **kwargs):
        from pyarrow import feather

        table = cls._pandas_to_arrow_table(df, preserve_index=None)
        # TODO using the datasets API could automatically support partitioning
        # on columns
        with fs.open(fs.sep.join([path, filename]), "wb") as f:
            feather.write_feather(table, f)


class GeoDatasetEngine:
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


class FeatherDatasetEngine(GeoDatasetEngine, ArrowDatasetEngine):
    file_format = "feather"


class FeatherFunctionWrapper:
    """
    Feather Function-Wrapper Class
    Reads Feather data from disk to produce a partition.
    """

    def __init__(self, engine, fs, columns, filter, schema, index):
        self.engine = engine
        self.fs = fs
        self.columns = columns
        self.filter = filter
        self.schema = schema
        self.index = index

    def project_columns(self, columns):
        """Return a new FeatherFunctionWrapper object with
        a sub-column projection.
        """
        if columns == self.columns:
            return self
        func = copy.deepcopy(self)
        func.columns = columns
        return func

    def __call__(self, parts):
        _df = self.engine.read_partition(
            self.fs, parts, self.schema, self.columns, self.filter
        )
        if self.index:
            _df.set_index(self.index, inplace=True)
        return _df


def read_feather(
    path,
    columns=None,
    filters=None,
    index=None,
    storage_options=None,
):
    """Read a Feather dataset into a Dask-GeoPandas DataFrame.

    Parameters
    ----------
    path: str or list(str)
        Source directory for data, or path(s) to individual Feather files.
        Paths can be a full URL with protocol specifier, and may include
        glob character if a single string.
    columns: None or list(str)
        Columns to load. If None, loads all.
    index: str
        Column name to set as index.
    storage_options: None or dict
        Further parameters to pass to the bytes backend.

    Returns
    -------
    dask_geopandas.GeoDataFrame (even if there is only one column)

    """
    # Get engine
    engine = FeatherDatasetEngine

    # Process file path(s)
    storage_options = storage_options or {}
    fs, _, paths = get_fs_token_paths(path, mode="rb", storage_options=storage_options)
    paths = sorted(paths, key=natural_sort_key)  # numeric rather than glob ordering

    # Let backend engine generate a list of parts from the dataset metadata
    parts, meta, schema, filter = engine.read_metadata(
        fs,
        paths,
        columns,
        filters,
        index,
    )

    # Update meta to be a GeoDataFrame
    meta = _update_meta_to_geodataframe(meta, schema.metadata)

    # Construct spatial partitioning information, if available
    spatial_partitions = geopandas.GeoSeries(
        [_get_partition_bounds(frag.physical_schema.metadata) for frag in parts],
        crs=meta.crs,
    )
    if spatial_partitions.isna().any():
        spatial_partitions = None

    # Construct and return a Blockwise layer
    label = "read-feather-"
    output_name = label + tokenize(path, columns, filters, index)
    layer = DataFrameIOLayer(
        output_name,
        columns,
        parts,
        FeatherFunctionWrapper(engine, fs, columns, filter, schema, index),
        label=label,
    )
    graph = HighLevelGraph({output_name: layer}, {output_name: set()})
    result = new_dd_object(graph, output_name, meta, [None] * (len(parts) + 1))
    result.spatial_partitions = spatial_partitions
    return result


def to_feather(
    df,
    path,
    write_index=True,
    storage_options=None,
    compute=True,
    compute_kwargs=None,
):
    """Store Dask.dataframe to Feather files

    Notes
    -----
    Each partition will be written to a separate file.

    Parameters
    ----------
    df : dask_geopandas.GeoDataFrame
    path : string or pathlib.Path
        Destination directory for data.  Prepend with protocol like ``s3://``
        or ``hdfs://`` for remote data.
    write_index : boolean, default True
        Whether or not to write the index. Defaults to True.
    storage_options : dict, default None
        Key/value pairs to be passed on to the file-system backend, if any.
    compute : bool, default True
        If True (default) then the result is computed immediately. If False
        then a ``dask.delayed`` object is returned for future computation.
    compute_kwargs : dict, default True
        Options to be passed in to the compute method

    See Also
    --------
    dask_geopandas.read_feather: Read Feather data to dask.dataframe
    """
    # based on the to_orc function from dask

    # Get engine
    engine = FeatherDatasetEngine

    # Process file path
    storage_options = storage_options or {}
    fs, _, _ = get_fs_token_paths(path, mode="wb", storage_options=storage_options)
    # Trim any protocol information from the path before forwarding
    path = fs._strip_protocol(path)

    if not write_index:
        # Not writing index - might as well drop it
        df = df.reset_index(drop=True)

    # Use df.npartitions to define file-name list
    fs.mkdirs(path, exist_ok=True)
    filenames = [f"part.{i}.feather" for i in range(df.npartitions)]

    # Construct IO graph
    dsk = {}
    name = "to-feather-" + tokenize(df, fs, path, write_index, storage_options)
    part_tasks = []
    for d, filename in enumerate(filenames):
        dsk[(name, d)] = (
            apply,
            engine.write_partition,
            [
                (df._name, d),
                path,
                fs,
                filename,
            ],
        )
        part_tasks.append((name, d))
    dsk[name] = (lambda x: None, part_tasks)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[df])

    # Compute or return future
    if compute:
        if compute_kwargs is None:
            compute_kwargs = dict()
        from dask_geopandas import GeoDataFrame

        return compute_as_if_collection(
            GeoDataFrame, graph, part_tasks, **compute_kwargs
        )
    return Scalar(graph, name, "")
