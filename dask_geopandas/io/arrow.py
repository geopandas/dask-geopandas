import copy
from typing import TYPE_CHECKING

from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.utils import natural_sort_key

from dask.dataframe.io.utils import _get_pyarrow_dtypes, _meta_from_dtypes

import pandas as pd
import geopandas

from fsspec.core import get_fs_token_paths


if TYPE_CHECKING:
    import pyarrow


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
        if columns is not None:
            ex = set(columns) - set(dtypes)
            if ex:
                raise ValueError(
                    "Requested columns (%s) not in schema (%s)" % (ex, set(dtypes))
                )
        columns = list(dtypes) if columns is None else columns
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
    # TODO convert columns based on GEO metadata (this will now only work
    # for a default "geometry" column)
    meta = geopandas.GeoDataFrame(meta)

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
    return new_dd_object(graph, output_name, meta, [None] * (len(parts) + 1))
