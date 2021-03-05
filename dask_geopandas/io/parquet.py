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

read_parquet = partial(dd.read_parquet, engine=GeoArrowEngine)
read_parquet.__doc__ = dd.read_parquet.__doc__
