import dask
from dask.utils import M, OperatorMethodMixin
import dask.array as da
from dask.array.utils import meta_from_array

import geopandas
from geopandas.array import _points_from_xy


def points_from_xy(x, y, z=None):
    """Convert arrays of x and y values to a GeometryArray of points."""
    x = da.asarray(x)
    y = da.asarray(y)
    if z is not None:
        z = da.asarray(z)
    out = _points_from_xy(x, y, z)
    aout = da.empty(len(x))
    aout[:] = out
    return GeometryArray.from_array(aout)


class GeometryArray(da.core.Array, OperatorMethodMixin):
    """
    Class wrapping a dask array of geopandas.array.GeometryArray object

    Parameters
    ----------
    dsk : dict
        The dask graph to compute this Array
    name : str
        The key prefix that specifies which keys in the dsk comprise this
        particular Array
    meta : geopandas.array.GeometryArray
        An empty geopandas object with names, dtypes, and indices matching the
        expected output.
    divisions : tuple of index values
        Values along which we partition our blocks on the index
    """
    _partition_type = geopandas.array.GeometryArray

    def __repr__(self):
        s = "<dask_geopandas.%s | %d tasks | %d npartitions>"
        return s % (type(self).__name__, len(self.dask), self.npartitions)

    def to_dask_array(self):
        """Create a dask.array object from a dask_geopandas object"""
        return self.map_partitions(M.to_array)


from_array = da.from_array
