from functools import partial
import numpy as np
import dask
import dask.dataframe as dd
from dask.utils import M, OperatorMethodMixin, derived_from, ignore_warning

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
    _partition_type = geopandas.base.GeoPandasBase

    def __repr__(self):
        s = "<dask_geopandas.%s | %d tasks | %d npartitions>"
        return s % (type(self).__name__, len(self.dask), self.npartitions)

    def to_dask_dataframe(self):
        """Create a dask.dataframe object from a dask_geopandas object"""
        return self.map_partitions(M.to_pandas)

    @classmethod
    def _bind_property(cls, attr):
        """Map property to partitions and bind to class"""
        def prop(self):
            meta = getattr(self._meta, attr)
            token = "%s-%s" % (self._name, attr)
            return self.map_partitions(
                getattr,
                attr,
                token=token,
                meta=meta
            )
        doc = getattr(cls._partition_type, attr).__doc__
        # Insert disclaimer that this is a copied docstring note that
        # malformed docs will not get the disclaimer (see #4746).
        if doc:
            doc = ignore_warning(doc, cls._partition_type, attr)
        setattr(cls, name, property(fget=prop, doc=doc))

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


for name in [
    "is_valid",
    "bounds"
]:
    _Frame._bind_property(name)

