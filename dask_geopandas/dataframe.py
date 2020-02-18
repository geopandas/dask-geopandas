from functools import partialmethod
import numpy as np
import dask
import dask.dataframe as dd
from dask.utils import M, OperatorMethodMixin, derived_from

import geopandas


class _Frame(dd.core._Frame, OperatorMethodMixin):
    """ Superclass for DataFrame and Series

    Parameters
    ----------
    dsk : dict
        The dask graph to compute this DataFrame
    name : str
        The key prefix that specifies which keys in the dask comprise this
        particular DataFrame / Series
    meta : geopandas.GeoDataFrame, geopandas.GeoSeries
        An empty cudf object with names, dtypes, and indices matching the
        expected output.
    divisions : tuple of index values
        Values along which we partition our blocks on the index
    """
    def __repr__(self):
        s = "<dask_geopandas.%s | %d tasks | %d npartitions>"
        return s % (type(self).__name__, len(self.dask), self.npartitions)

    def to_dask_dataframe(self):
        """Create a dask.dataframe object from a dask_geopandas object"""
        return self.map_partitions(M.to_pandas)

    @property
    @derived_from(geopandas.base.GeoPandasBase)
    def bounds(self):
        return self.map_partitions(
            getattr,
            "bounds",
            token=self._name + "-bounds",
            meta=self._meta.bounds
        )

    @property
    @derived_from(geopandas.base.GeoPandasBase)
    def total_bounds(self):
        def agg(concatted):
            return np.array(
            (
                concatted[0::4].min(),  # minx
                concatted[1::4].min(),  # miny
                concatted[2::4].max(),  # maxx
                concatted[3::4].max(),  # maxy
            )
        )

        return self.reduction(
            lambda x: getattr(x, "total_bounds"),
            token=self._name + "-total_bounds",
            meta=self._meta.total_bounds,
            aggregate=agg,
        )


class GeoSeries(_Frame, dd.core.Series):
    _partition_type = geopandas.GeoSeries


class GeoDataFrame(_Frame, dd.core.DataFrame):
    _partition_type = geopandas.GeoDataFrame


from_geopandas = dd.from_pandas


def from_dask_dataframe(df):
    return df.map_partitions(geopandas.GeoDataFrame)


for name in []:
    meth = getattr(geopandas.GeoDataFrame, name)
    GeoDataFrame._bind_operator_method(name, meth)

    meth = getattr(geopandas.GeoSeries, name)
    GeoSeries._bind_operator_method(name, meth)

for name in [
    "set_geometry",
    "rename_geometry",
    "iterfeatures",
    "merge",
    "plot",
    "dissolve",
    "explode",
    "astype",
]:
    meth = getattr(geopandas.GeoDataFrame, name)
    GeoDataFrame._bind_operator_method(name, meth)

for name in []:

    meth = getattr(geopandas.GeoSeries, name)
    GeoSeries._bind_comparison_method(name, meth)
