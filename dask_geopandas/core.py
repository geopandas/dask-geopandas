import numpy as np

import dask.dataframe as dd
from dask.dataframe.core import _emulate, map_partitions, elemwise
from dask.utils import M, OperatorMethodMixin, derived_from, ignore_warning

import geopandas
from shapely.geometry.collection import GeometryCollection


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
            token = f"{self._name}-{attr}"
            return self.map_partitions(getattr, attr, token=token, meta=meta)

        doc = getattr(cls._partition_type, attr).__doc__
        # Insert disclaimer that this is a copied docstring note that
        # malformed docs will not get the disclaimer (see #4746).
        if doc:
            doc = ignore_warning(doc, cls._partition_type, attr)
        setattr(cls, name, property(fget=prop, doc=doc))

    @classmethod
    def _bind_elemwise_comparison_method(
        cls, name, comparison, original, *args, **kwargs
    ):
        """ bind comparison method like GeoSeries.contains to this class """

        def meth(self, other, *args, **kwargs):
            return elemwise(comparison, self, other, *args, **kwargs)

        meth.__name__ = name
        setattr(cls, name, derived_from(original)(meth))

    @classmethod
    def _bind_elemwise_operator_method(cls, name, op, original, *args, **kwargs):
        """ bind operator method like GeoSeries.distance to this class """
        # name must be explicitly passed for div method whose name is truediv
        def meth(self, other, *args, **kwargs):
            meta = _emulate(op, self, other)
            return map_partitions(
                op, self, other, meta=meta, enforce_metadata=False, *args, **kwargs
            )

        meth.__name__ = name
        setattr(cls, name, derived_from(original)(meth))

    @property
    def crs(self):
        """
        The Coordinate Reference System (CRS) represented as a ``pyproj.CRS``
        object.

        Returns None if the CRS is not set, and to set the value it
        :getter: Returns a ``pyproj.CRS`` or None. When setting, the value
        can be anything accepted by :meth:`pyproj.CRS.from_user_input`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
        """
        return self._meta.crs

    @crs.setter
    def crs(self, value):
        """Sets the value of the crs"""
        self.dask = self.set_crs(value).dask

    def set_crs(self, value):
        self._meta.crs = value

        def set_crs(df, crs):
            df = df.copy(deep=False)
            df.crs = crs
            return df

        return self.map_partitions(
            set_crs, value, meta=self._meta, enforce_metadata=False
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

    @property
    def sindex(self):
        """Need to figure out how to concatenate spatial indexes"""
        raise NotImplementedError

    @property
    @derived_from(geopandas.base.GeoPandasBase)
    def unary_union(self):
        attr = "unary_union"
        meta = GeometryCollection()

        return self.reduction(
            lambda x: getattr(x, attr),
            token=f"{self._name}-{attr}",
            aggregate=lambda x: getattr(geopandas.GeoSeries(x), attr),
            meta=meta,
        )

    def representative_point(self):
        raise NotImplementedError

    @derived_from(geopandas.base.GeoPandasBase)
    def project(self, other, normalized=False):
        """Objects must have the same number or rows"""
        meth = getattr(self._partition_type, "project")
        meta = _emulate(meth, self, other, normalized=normalized)
        return map_partitions(
            meth, self, other, normalized=normalized, meta=meta, enforce_metadata=False,
        )


class GeoSeries(_Frame, dd.core.Series):
    _partition_type = geopandas.GeoSeries


class GeoDataFrame(_Frame, dd.core.DataFrame):
    _partition_type = geopandas.GeoDataFrame


from_geopandas = dd.from_pandas


def from_dask_dataframe(df):
    return df.map_partitions(geopandas.GeoDataFrame)


def points_from_xy(df, x="x", y="y", z="z"):
    """Convert dask.dataframe of x and y (and optionally z) values to a GeoSeries."""

    def func(data, x, y, z):
        return geopandas.GeoSeries(
            geopandas.points_from_xy(
                data[x], data[y], data[z] if z in df.columns else None
            )
        )

    return df.map_partitions(func, x, y, z, meta=geopandas.GeoSeries())


for name in [
    "area",
    "geom_type",
    "type",
    "length",
    "is_valid",
    "is_empty",
    "is_simple",
    "is_ring",
    "has_z",
    "boundary",
    "centroid",
    "convex_hull",
    "envelope",
    "exterior",
    "interiors",
    "bounds",
]:
    _Frame._bind_property(name)

for name in [
    "geometry",
    "x",
    "y",
]:
    GeoSeries._bind_property(name)

for name in [
    "contains",
    "geom_equals",
    "crosses",
    "disjoint",
    "intersects",
    "overlaps",
    "touches",
    "within",
]:
    meth = getattr(geopandas.base.GeoPandasBase, name)
    GeoSeries._bind_elemwise_comparison_method(
        name, meth, original=geopandas.base.GeoPandasBase
    )
    GeoDataFrame._bind_elemwise_comparison_method(
        name, meth, original=geopandas.base.GeoPandasBase
    )


for name in [
    "distance",
    "difference",
    "symmetric_difference",
    "union",
    "intersection",
    "relate",
]:
    meth = getattr(geopandas.base.GeoPandasBase, name)
    GeoSeries._bind_elemwise_operator_method(
        name, meth, original=geopandas.base.GeoPandasBase
    )
    GeoDataFrame._bind_elemwise_operator_method(
        name, meth, original=geopandas.base.GeoPandasBase
    )
