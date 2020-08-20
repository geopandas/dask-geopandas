from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

import geopandas

import dask.dataframe as dd

try:
    # pyarrow is imported here, but is an optional dependency
    from dask.dataframe.io.parquet.arrow import ArrowEngine
except ImportError:
    ArrowEngine = object

if TYPE_CHECKING:
    import pyarrow


class GeoArrowEngine(ArrowEngine):
    @classmethod
    def read_metadata(cls, *args, **kwargs):
        meta, stats, parts, index = super().read_metadata(*args, **kwargs)

        # Update meta to be a GeoDataFrame
        # TODO convert columns based on GEO metadata (this will now only work
        # for a default "geometry" column)
        meta = geopandas.GeoDataFrame(meta)

        return (meta, stats, parts, index)

    @classmethod
    def _arrow_table_to_pandas(
        cls, arrow_table: "pyarrow.Table", categories, **kwargs
    ) -> pd.DataFrame:
        from geopandas.io.arrow import _arrow_to_geopandas

        _kwargs = kwargs.get("arrow_to_pandas", {})
        _kwargs.update({"use_threads": False, "ignore_metadata": False})

        # TODO support additional keywords
        return _arrow_to_geopandas(arrow_table)

    @classmethod
    def _pandas_to_arrow_table(
        cls, df: pd.DataFrame, preserve_index=False, schema=None
    ) -> "pyarrow.Table":
        from geopandas.io.arrow import _geopandas_to_arrow

        # TODO add support for schema
        table = _geopandas_to_arrow(df, index=preserve_index)
        # table = pa.Table.from_pandas(df, preserve_index=preserve_index,
        #                              schema=schema)
        return table

    # --------------------------------------------------
    # TEMP: can be removed once https://github.com/dask/dask/pull/6505 is
    # merged and released

    @classmethod
    def write_partition(
        cls,
        df,
        path,
        fs,
        filename,
        partition_on,
        return_metadata,
        fmd=None,
        compression=None,
        index_cols=None,
        schema=None,
        **kwargs,
    ):
        from dask.dataframe.io.parquet.arrow import _index_in_schema, _write_partitioned
        import pyarrow.parquet as pq

        _meta = None
        preserve_index = False
        if _index_in_schema(index_cols, schema):
            df.set_index(index_cols, inplace=True)
            preserve_index = True
        else:
            index_cols = []

        t = cls._pandas_to_arrow_table(df, preserve_index, schema)

        if partition_on:
            md_list = _write_partitioned(
                t,
                path,
                filename,
                partition_on,
                fs,
                index_cols=index_cols,
                compression=compression,
                **kwargs,
            )
            if md_list:
                _meta = md_list[0]
                for i in range(1, len(md_list)):
                    _meta.append_row_groups(md_list[i])
        else:
            md_list = []
            with fs.open(fs.sep.join([path, filename]), "wb") as fil:
                pq.write_table(
                    t,
                    fil,
                    compression=compression,
                    metadata_collector=md_list,
                    **kwargs,
                )
            if md_list:
                _meta = md_list[0]
                _meta.set_file_path(filename)
        # Return the schema needed to write the metadata
        if return_metadata:
            return [{"schema": t.schema, "meta": _meta}]
        else:
            return []

    # --------------------------------------------------


to_parquet = partial(dd.to_parquet, engine=GeoArrowEngine)
read_parquet = partial(dd.read_parquet, engine=GeoArrowEngine)
