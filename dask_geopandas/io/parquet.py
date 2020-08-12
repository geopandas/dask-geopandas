from functools import partial
from typing import TYPE_CHECKING

import pandas as pd

import geopandas
from geopandas.io.arrow import _geopandas_to_arrow, _arrow_to_geopandas

import dask.dataframe as dd
from dask.dataframe.io.parquet.arrow import ArrowEngine

if TYPE_CHECKING:
    import pyarrow


class GeoArrowEngine(ArrowEngine):

    @classmethod
    def read_metadata(cls, *args, **kwargs):
        meta, stats, parts, index = super().read_metadata(*args, **kwargs)

        # Update meta to be a GeoDataFrame
        # TODO convert columns based on GEO metadata (this will now only work for a default "geometry" column)
        meta = geopandas.GeoDataFrame(meta)

        return (meta, stats, parts, index)

    @classmethod
    def _arrow_table_to_pandas(
        cls, arrow_table: "pyarrow.Table", categories, **kwargs
    ) -> pd.DataFrame:
        _kwargs = kwargs.get("arrow_to_pandas", {})
        _kwargs.update({"use_threads": False, "ignore_metadata": False})

        # TODO support additional keywords
        return _arrow_to_geopandas(arrow_table)

    @classmethod
    def _pandas_to_arrow_table(
        cls, df: pd.DataFrame, preserve_index=False, schema=None
    ) -> "pyarrow.Table":
        # TODO add support for schema
        table = _geopandas_to_arrow(df, index=preserve_index)
        # table = pa.Table.from_pandas(df, preserve_index=preserve_index, schema=schema)
        return table


to_parquet = partial(dd.to_parquet, engine=GeoArrowEngine)
read_parquet = partial(dd.read_parquet, engine=GeoArrowEngine)
