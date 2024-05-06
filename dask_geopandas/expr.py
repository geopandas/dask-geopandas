from packaging.version import Version
import warnings

import numpy as np
import pandas as pd

import dask
import dask.dataframe as dd
import dask.array as da
from dask.dataframe import map_partitions
from dask.highlevelgraph import HighLevelGraph
from dask.utils import M, OperatorMethodMixin, derived_from, ignore_warning
from dask.base import tokenize

import dask_expr as dx
from dask_expr import (
    elemwise,
    from_graph,
    get_collection_type,
)
from dask_expr._collection import new_collection
from dask_expr._expr import _emulate, ApplyConcatApply

import geopandas
import shapely
from shapely.geometry import box

from .hilbert_distance import _hilbert_distance
from .morton_distance import _morton_distance
from .geohash import _geohash

import dask_geopandas


DASK_2022_8_1 = Version(dask.__version__) >= Version("2022.8.1")
GEOPANDAS_0_12 = Version(geopandas.__version__) >= Version("0.12.0")
PANDAS_2_0_0 = Version(pd.__version__) >= Version("2.0.0")


if Version(shapely.__version__) < Version("2.0"):
    raise ImportError(
        "Running dask-geopandas requires Shapely >= 2.0 to be "
        "installed. However, Shapely is only at "
        f"version {shapely.__version__}"
    )


class UnaryUnion(ApplyConcatApply):
    @classmethod
    def chunk(cls, df, **kwargs):
        return df.unary_union

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        s = geopandas.GeoSeries(inputs)
        return s.unary_union


class TotalBounds(ApplyConcatApply):
    @classmethod
    def chunk(cls, df, **kwargs):
        return df.total_bounds

    @classmethod
    def aggregate(cls, inputs, **kwargs):
        concatted = np.stack(inputs)
        if len(concatted) == 0:
            # numpy 'min' cannot handle empty arrays
            # TODO with numpy >= 1.15, the 'initial' argument can be used
            return np.array([np.nan, np.nan, np.nan, np.nan])
        with warnings.catch_warnings():
            # if all rows are empty geometry / none, nan is expected
            warnings.filterwarnings(
                "ignore", r"All-NaN slice encountered", RuntimeWarning
            )
            return np.array(
                (
                    np.nanmin(concatted[:, 0]),  # minx
                    np.nanmin(concatted[:, 1]),  # miny
                    np.nanmax(concatted[:, 2]),  # maxx
                    np.nanmax(concatted[:, 3]),  # maxy
                )
            )

    @property
    def _meta(self):
        return self.frame._meta.total_bounds


def _set_crs(df, crs, allow_override):
    """Return a new object with crs set to ``crs``"""
    return df.set_crs(crs, allow_override=allow_override)


class _Frame(dx.FrameBase, OperatorMethodMixin):
    """Superclass for DataFrame and Series

    Parameters
    ----------
    dsk : dict
        The dask graph to compute this DataFrame
    name : str
        The key prefix that specifies which keys in the dask comprise this
        particular DataFrame / Series
    meta : geopandas.GeoDataFrame, geopandas.GeoSeries
        An empty geopandas object with names, dtypes, and indices matching the
        expected output.
    divisions : tuple of index values
        Values along which we partition our blocks on the index
    """

    _partition_type = geopandas.base.GeoPandasBase

    def __init__(self, expr, spatial_partitions=None):
        super().__init__(expr)
        self.spatial_partitions = spatial_partitions

    def to_dask_dataframe(self):
        """Create a dask.dataframe object from a dask_geopandas object"""
        return self.map_partitions(pd.DataFrame)

    def __dask_postpersist__(self):
        func, args = super().__dask_postpersist__()

        return self._rebuild, (func, args)

    def _rebuild(self, graph, func, args):
        collection = func(graph, *args)
        collection.spatial_partitions = self.spatial_partitions
        return collection

    def optimize(self, fuse: bool = True):
        result = new_collection(self.expr.optimize(fuse=fuse))
        result = self._propagate_spatial_partitions(result)
        return result

    @property
    def spatial_partitions(self):
        """
        The spatial extent of each of the partitions of the dask GeoDataFrame.
        """
        return self._spatial_partitions

    @spatial_partitions.setter
    def spatial_partitions(self, value):
        if value is not None:
            if not isinstance(value, geopandas.GeoSeries):
                raise TypeError(
                    "Expected a geopandas.GeoSeries for the spatial_partitions, "
                    f"got {type(value)} instead."
                )
            if len(value) != self.npartitions:
                raise ValueError(
                    f"Expected spatial partitions of length {self.npartitions}, "
                    f"got {len(value)} instead."
                )
        self._spatial_partitions = value

    @property
    def _args(self):
        # Ensure we roundtrip through pickle correctly
        # https://github.com/geopandas/dask-geopandas/issues/237
        return super()._args + (self.spatial_partitions,)

    def __setstate__(self, state):
        *dask_state, spatial_partitions = state
        super().__setstate__(dask_state)
        self.spatial_partitions = spatial_partitions

    @classmethod
    def _bind_property(cls, attr, preserve_spatial_partitions=False):
        """Map property to partitions and bind to class"""

        def prop(self):
            meta = getattr(self._meta, attr)
            result = self.map_partitions(getattr, attr, token=attr, meta=meta)
            if preserve_spatial_partitions:
                result = self._propagate_spatial_partitions(result)
            return result

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
        """bind comparison method like GeoSeries.contains to this class"""

        def meth(self, other, *args, **kwargs):
            return elemwise(comparison, self, other, *args, **kwargs)

        meth.__name__ = name
        setattr(cls, name, derived_from(original)(meth))

    @classmethod
    def _bind_elemwise_operator_method(cls, name, op, original, *args, **kwargs):
        """bind operator method like GeoSeries.distance to this class"""

        # name must be explicitly passed for div method whose name is truediv
        def meth(self, other, *args, **kwargs):
            meta = _emulate(op, self, other)
            return map_partitions(
                op, self, other, meta=meta, enforce_metadata=False, *args, **kwargs
            )

        meth.__name__ = name
        setattr(cls, name, derived_from(original)(meth))

    def calculate_spatial_partitions(self):
        """Calculate spatial partitions"""
        # TEMP method to calculate spatial partitions for testing, need to
        # add better methods (set_partitions / repartition)
        parts = geopandas.GeoSeries(
            self.map_partitions(
                lambda part: shapely.convex_hull(
                    shapely.geometrycollections(np.asarray(part.geometry))
                )
            ).compute(),
            crs=self.crs,
        )
        self.spatial_partitions = parts

    def _propagate_spatial_partitions(self, new_object):
        """
        We need to override several dask methods to ensure the spatial
        partitions are properly propagated.
        This is a helper method to set this.
        """
        new_object.spatial_partitions = self.spatial_partitions
        return new_object

    @property
    @derived_from(geopandas.GeoSeries)
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
        # When using setter, Geopandas always overrides the CRS
        new = self.set_crs(value, allow_override=True)
        self._expr = new.expr

    @derived_from(geopandas.GeoSeries)
    def set_crs(self, value, allow_override=False):
        """Set the value of the crs on a new object"""
        new = self.map_partitions(
            _set_crs, value, allow_override, enforce_metadata=False
        )
        if self.spatial_partitions is not None:
            new.spatial_partitions = self.spatial_partitions.set_crs(
                value, allow_override=allow_override
            )
        return new

    @derived_from(geopandas.GeoSeries)
    def to_crs(self, crs=None, epsg=None):
        return self.map_partitions(M.to_crs, crs=crs, epsg=epsg)

    def copy(self):
        """Make a copy of the dataframe

        Creates shallow copies of the computational graph and spatial partitions.
        Does not affect the underlying data.
        """
        self_copy = super().copy()
        if self.spatial_partitions is not None:
            self_copy.spatial_partitions = self.spatial_partitions.copy()
        return self_copy

    @property
    @derived_from(geopandas.base.GeoPandasBase)
    def total_bounds(self):
        total_bounds = TotalBounds(self.expr)
        graph = total_bounds.lower_completely().__dask_graph__()
        return da.Array(
            graph,
            list(graph.keys())[0][0],
            chunks=((4,),),
            dtype=np.dtype("float64"),
        )

    @property
    def sindex(self):
        """Need to figure out how to concatenate spatial indexes"""
        raise NotImplementedError

    @property
    @derived_from(geopandas.base.GeoPandasBase)
    def unary_union(self):
        return new_collection(UnaryUnion(self.expr))

    @derived_from(geopandas.base.GeoPandasBase)
    def representative_point(self):
        return self.map_partitions(
            self._partition_type.representative_point, enforce_metadata=False
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def geom_equals_exact(self, other, tolerance):
        comparison = self._partition_type.geom_equals_exact
        return elemwise(
            comparison, self, other, meta=pd.Series(dtype=bool), tolerance=tolerance
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def buffer(self, distance, resolution=16, **kwargs):
        return self.map_partitions(
            self._partition_type.buffer,
            distance,
            resolution=resolution,
            enforce_metadata=False,
            **kwargs,
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def simplify(self, *args, **kwargs):
        return self.map_partitions(
            self._partition_type.simplify, *args, enforce_metadata=False, **kwargs
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def interpolate(self, distance, normalized=False):
        return self.map_partitions(
            self._partition_type.interpolate,
            distance,
            normalized=normalized,
            enforce_metadata=False,
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def affine_transform(self, matrix):
        return self.map_partitions(
            self._partition_type.affine_transform, matrix, enforce_metadata=False
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def translate(self, xoff=0.0, yoff=0.0, zoff=0.0):
        return self.map_partitions(
            self._partition_type.translate,
            xoff=xoff,
            yoff=yoff,
            zoff=zoff,
            enforce_metadata=False,
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def rotate(self, angle, origin="center", use_radians=False):
        return self.map_partitions(
            self._partition_type.rotate,
            angle,
            origin=origin,
            use_radians=use_radians,
            enforce_metadata=False,
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def scale(self, xfact=1.0, yfact=1.0, zfact=1.0, origin="center"):
        return self.map_partitions(
            self._partition_type.scale,
            xfact=xfact,
            yfact=yfact,
            zfact=zfact,
            origin=origin,
            enforce_metadata=False,
        )

    @derived_from(geopandas.base.GeoPandasBase)
    def skew(self, xs=0.0, ys=0.0, origin="center", use_radians=False):
        return self.map_partitions(
            self._partition_type.skew,
            xs=xs,
            ys=ys,
            origin=origin,
            use_radians=use_radians,
            enforce_metadata=False,
        )

    @derived_from(geopandas.geodataframe.GeoDataFrame)
    def explode(self):
        return self.map_partitions(self._partition_type.explode, enforce_metadata=False)

    @property
    @derived_from(geopandas.geodataframe.GeoDataFrame)
    def cx(self):
        """
        Coordinate based indexer to select by intersection with bounding box.

        Format of input should be ``.cx[xmin:xmax, ymin:ymax]``. Any of
        ``xmin``, ``xmax``, ``ymin``, and ``ymax`` can be provided, but input
        must include a comma separating x and y slices. That is, ``.cx[:, :]``
        will return the full series/frame, but ``.cx[:]`` is not implemented.
        """
        return _CoordinateIndexer(self)

    def hilbert_distance(self, total_bounds=None, level=16):
        """
        Calculate the distance along a Hilbert curve.

        The distances are calculated for the midpoints of the geometries in the
        GeoDataFrame, and using the total bounds of the GeoDataFrame.

        The Hilbert distance can be used to spatially partition Dask-GeoPandas
        objects, by mapping two dimensional geometries along the Hilbert curve.

        Parameters
        ----------
        total_bounds : 4-element array, optional
            The spatial extent in which the curve is constructed (used to
            rescale the geometry midpoints). By default, the total bounds
            of the full dask GeoDataFrame will be computed (from the spatial
            partitions, if available, otherwise computed from the full
            dataframe). If known, you can pass the total bounds to avoid this
            extra computation.
        level : int (1 - 16), default 16
            Determines the precision of the curve (points on the curve will
            have coordinates in the range [0, 2^level - 1]).

        Returns
        -------
        dask.Series
            Series containing distances for each partition

        """
        # Compute total bounds of all partitions rather than each partition
        if total_bounds is None:
            if self.spatial_partitions is not None:
                total_bounds = self.spatial_partitions.total_bounds
            else:
                total_bounds = self.total_bounds

        # Calculate hilbert distances for each partition
        distances = self.map_partitions(
            _hilbert_distance,
            total_bounds=total_bounds,
            level=level,
            meta=pd.Series([], name="hilbert_distance", dtype="uint32"),
        )

        return distances

    def morton_distance(self, total_bounds=None, level=16):
        """
        Calculate the distance of geometries along the Morton curve

        The Morton curve is also known as Z-order https://en.wikipedia.org/wiki/Z-order.

        The Morton distance can be used to spatially partition Dask-GeoPandas objects,
        by mapping two-dimensional geometries along the Morton space-filing curve.

        Each geometry is represented by the midpoint of its bounds and linked to the
        Morton curve. The function returns a distance from the beginning
        of the curve to the linked point.

        Morton distance is more performant than ``hilbert_distance`` but can result in
        less optimal partitioning.

        Parameters
        ----------

        total_bounds : 4-element array, optional
            The spatial extent in which the curve is constructed (used to
            rescale the geometry midpoints). By default, the total bounds
            of the full dask GeoDataFrame will be computed (from the spatial
            partitions, if available, otherwise computed from the full
            dataframe). If known, you can pass the total bounds to avoid this
            extra computation.
        level : int (1 - 16), default 16
            Determines the precision of the Morton curve.

        Returns
        -------
        dask.Series
            Series containing distances along the Morton curve

        """
        # Compute total bounds of all partitions rather than each partition
        if total_bounds is None:
            if self.spatial_partitions is not None:
                total_bounds = self.spatial_partitions.total_bounds
            else:
                total_bounds = self.total_bounds

        # Calculate Morton distances for each partition
        distances = self.map_partitions(
            _morton_distance,
            total_bounds=total_bounds,
            level=level,
            meta=pd.Series([], name="morton_distance", dtype="uint32"),
        )

        return distances

    def geohash(self, as_string=True, precision=12):
        """
        Calculate geohash based on the middle points of the geometry bounds
        for a given precision.
        Only geographic coordinates (longitude, latitude) are supported.

        Parameters
        ----------
        as_string : bool, default True
            To return string or int Geohash.
        precision : int (1 - 12), default 12
            Precision of the string geohash values. Only used when
            ``as_string=True``.

        Returns
        -------
        type : pandas.Series
            Series containing Geohash
        """

        if precision not in range(1, 13):
            raise ValueError(
                "The Geohash precision only accepts an integer value between 1 and 12"
            )

        if as_string is True:
            dtype = object
        else:
            dtype = np.uint64

        geohashes = self.map_partitions(
            _geohash,
            as_string=as_string,
            precision=precision,
            meta=pd.Series([], name="geohash", dtype=dtype),
        )

        return geohashes

    @derived_from(geopandas.GeoDataFrame)
    def clip(self, mask, keep_geom_type=False):
        return dask_geopandas.clip(self, mask=mask, keep_geom_type=keep_geom_type)

    @derived_from(geopandas.GeoDataFrame)
    def to_wkt(self, **kwargs):
        meta = self._meta.to_wkt(**kwargs)
        return self.map_partitions(M.to_wkt, **kwargs, meta=meta)

    @derived_from(geopandas.GeoDataFrame)
    def to_wkb(self, hex=False, **kwargs):
        meta = self._meta.to_wkb(hex=hex, **kwargs)
        return self.map_partitions(M.to_wkb, hex=hex, **kwargs, meta=meta)


class GeoSeries(_Frame, dd.Series):
    """Parallel GeoPandas GeoSeries

    Do not use this class directly. Instead use functions like
    :func:`dask_geopandas.read_parquet`,or :func:`dask_geopandas.from_geopandas`.
    """

    _partition_type = geopandas.GeoSeries

    @derived_from(geopandas.GeoSeries)
    def explode(self, ignore_index=False, index_parts=None):
        return self.map_partitions(
            M.explode,
            ignore_index=ignore_index,
            index_parts=index_parts,
            enforce_metadata=False,
        )


class GeoDataFrame(_Frame, dd.DataFrame):
    """Parallel GeoPandas GeoDataFrame

    Do not use this class directly. Instead use functions like
    :func:`dask_geopandas.read_parquet`,or :func:`dask_geopandas.from_geopandas`.
    """

    _partition_type = geopandas.GeoDataFrame

    @property
    def geometry(self):
        geometry_column_name = self._meta._geometry_column_name
        if geometry_column_name not in self.columns:
            raise AttributeError(
                "No geometry data set yet (expected in"
                " column '%s'." % geometry_column_name
            )
        return self[geometry_column_name]

    @geometry.setter
    def geometry(self, col):
        """Sets the geometry column"""
        new = self.set_geometry(col)
        self._expr = new.expr

    @derived_from(dd.DataFrame)
    def set_index(self, *args, **kwargs):
        """Override to ensure we get GeoDataFrame with set geometry column"""
        ddf = super().set_index(*args, **kwargs)
        return ddf.set_geometry(self._meta.geometry.name)

    @derived_from(geopandas.GeoDataFrame)
    def set_geometry(self, col):
        # calculate ourselves to use meta and not meta_nonempty, which would
        # raise an error if meta is an invalid GeoDataFrame (e.g. geometry
        # column name not yet set correctly)
        if isinstance(col, GeoSeries):
            meta = self._meta.set_geometry(col._meta)
        else:
            meta = self._meta.set_geometry(col)
        return self.map_partitions(M.set_geometry, col, meta=meta)

    @derived_from(geopandas.GeoDataFrame)
    def rename_geometry(self, col):
        meta = self._meta.rename_geometry(col)
        return self.map_partitions(M.rename_geometry, col, meta=meta)

    def __getitem__(self, key):
        """
        If the result is a new dask_geopandas.GeoDataFrame/GeoSeries (automatically
        determined by dask based on the meta), then pass through the spatial
        partitions information.
        """
        result = super().__getitem__(key)
        if isinstance(result, _Frame):
            result = self._propagate_spatial_partitions(result)
        return result

    def _repr_html_(self):
        output = super()._repr_html_()
        return output.replace(
            "Dask DataFrame Structure", "Dask-GeoPandas GeoDataFrame Structure"
        )

    @derived_from(dd.DataFrame)
    def to_parquet(self, path, *args, **kwargs):
        """See dask_geopadandas.to_parquet docstring for more information"""
        from .io.parquet import to_parquet

        return to_parquet(self, path, *args, **kwargs)

    def to_feather(self, path, *args, **kwargs):
        """See dask_geopadandas.to_feather docstring for more information"""
        from .io.arrow import to_feather

        return to_feather(self, path, *args, **kwargs)

    def dissolve(self, by=None, aggfunc="first", split_out=1, **kwargs):
        """Dissolve geometries within ``groupby`` into a single geometry.

        Parameters
        ----------
        by : string, default None
            Column whose values define groups to be dissolved. If None,
            whole GeoDataFrame is considered a single group.
        aggfunc : function,  string or dict, default "first"
            Aggregation function for manipulation of data associated
            with each group. Passed to dask ``groupby.agg`` method.
            Note that ``aggfunc`` needs to be applicable to all columns (i.e. ``"mean"``
            cannot be used with string dtype). Select only required columns before
            ``dissolve`` or pass a dictionary mapping to ``aggfunc`` to specify the
            aggregation function for each column separately.
        split_out : int, default 1
            Number of partitions of the output

        **kwargs
            keyword arguments passed to ``groupby``

        Examples
        --------
        >>> ddf.dissolve("foo", split_out=12)

        >>> ddf[["foo", "bar", "geometry"]].dissolve("foo", aggfunc="mean")

        >>> ddf.dissolve("foo", aggfunc={"bar": "mean", "baz": "first"})

        """
        if by is None:
            by = lambda x: 0
            drop = [self.geometry.name]
        else:
            drop = [by, self.geometry.name]

        def union(block):
            block = geopandas.GeoSeries(block)
            merged_geom = block.unary_union
            return merged_geom

        merge_geometries = dd.Aggregation(
            "merge_geometries", lambda s: s.agg(union), lambda s0: s0.agg(union)
        )
        if isinstance(aggfunc, dict):
            data_agg = aggfunc
        else:
            data_agg = {col: aggfunc for col in self.columns.drop(drop)}
        data_agg[self.geometry.name] = merge_geometries
        # dask 2022.8.1 added shuffle keyword, enabled by default if split_out > 1
        # starting with dask 2022.9.1, but geopandas doesn't yet work with shuffle
        # https://github.com/geopandas/dask-geopandas/pull/229
        agg_kwargs = {"shuffle": False} if DASK_2022_8_1 else {}
        aggregated = self.groupby(by=by, **kwargs).agg(
            data_agg, split_out=split_out, **agg_kwargs
        )
        return aggregated.set_crs(self.crs)

    def sjoin(self, df, how="inner", predicate="intersects"):
        """
        Spatial join of two GeoDataFrames.

        Parameters
        ----------
        df : geopandas or dask_geopandas GeoDataFrame
            If a geopandas.GeoDataFrame is passed, it is considered as a
            dask_geopandas.GeoDataFrame with 1 partition (without spatial
            partitioning information).
        how : string, default 'inner'
            The type of join. Currently only 'inner' is supported.
        predicate : string, default 'intersects'
            Binary predicate how to match corresponding rows of the left and right
            GeoDataFrame. Possible values: 'contains', 'contains_properly',
            'covered_by', 'covers', 'crosses', 'intersects', 'overlaps',
            'touches', 'within'.

        Returns
        -------
        dask_geopandas.GeoDataFrame

        Notes
        -----
        If both the left and right GeoDataFrame have spatial partitioning
        information available (the ``spatial_partitions`` attribute is set),
        the output partitions are determined based on intersection of the
        spatial partitions. In all other cases, the output partitions are
        all combinations (cartesian/cross product) of all input partition
        of the left and right GeoDataFrame.
        """
        return dask_geopandas.sjoin(self, df, how=how, predicate=predicate)

    def spatial_shuffle(
        self,
        by="hilbert",
        level=None,
        calculate_partitions=True,
        npartitions=None,
        divisions=None,
        **kwargs,
    ):
        """
        Shuffle the data into spatially consistent partitions.

        This realigns the dataset to be spatially sorted, i.e. geometries that are
        spatially near each other will be within the same partition. This is
        useful especially for overlay operations like a spatial join as it reduces the
        number of interactions between individual partitions.

        The spatial information is stored in the index and will replace the existing
        index.

        Note that ``spatial_shuffle`` uses ``set_index`` under the hood and comes with
        all its potential performance drawbacks.

        Parameters
        ----------
        by : string (default 'hilbert')
            Spatial sorting method, one of {'hilbert', 'morton', 'geohash'}. See
            ``hilbert_distance``, ``morton_distance`` and ``geohash`` methods for
            details.
        level : int (default None)
            Level (precision) of the  Hilbert and Morton
            curves used as a sorting method. Defaults to 16. Does not have an effect for
            the ``'geohash'`` option.
        calculate_partitions : bool (default True)
            Calculate new spatial partitions after shuffling
        npartitions : int, None, or 'auto'
            The ideal number of output partitions. If None, use the same as the input.
            If 'auto' then decide by memory use. Only used when divisions is not given.
            If divisions is given, the number of output partitions will be
            len(divisions) - 1.
        divisions: list, optional
            The “dividing lines” used to split the new index into partitions. Needs to
            match the values returned by the sorting method.
        **kwargs
            Keyword arguments passed to ``set_index``.

        Returns
        -------
        dask_geopandas.GeoDataFrame

        Notes
        -----
        This method, similarly to ``calculate_spatial_partitions``, is computed
        partially eagerly as it needs to calculate the distances for all existing
        partitions before it can determine the divisions for the new
        spatially-shuffled partitions.
        """
        if level is None:
            level = 16
        if by == "hilbert":
            by = self.hilbert_distance(level=level)
        elif by == "morton":
            by = self.morton_distance(level=level)
        elif by == "geohash":
            by = self.geohash(as_string=False)
        else:
            raise ValueError(
                f"'{by}' is not supported. Use one of ['hilbert', 'morton, 'geohash']."
            )

        sorted_ddf = self.set_index(
            by,
            sorted=False,
            npartitions=npartitions,
            divisions=divisions,
            inplace=False,
            **kwargs,
        )

        if calculate_partitions:
            sorted_ddf.calculate_spatial_partitions()

        return sorted_ddf

    @derived_from(geopandas.GeoDataFrame)
    def explode(self, column=None, ignore_index=False, index_parts=None):
        return self.map_partitions(
            M.explode,
            column=column,
            ignore_index=ignore_index,
            index_parts=index_parts,
            enforce_metadata=False,
        )


from_geopandas = dx.from_pandas


def from_dask_dataframe(df, geometry=None):
    """
    Create GeoDataFrame from dask DataFrame.

    Parameters
    ----------
    df : dask DataFrame
    geometry : str or array-like, optional
        If a string, the column to use as geometry. By default, it will look
        for a column named "geometry". If array-like or dask (Geo)Series,
        the values will be set as 'geometry' column on the GeoDataFrame.

    """
    # If the geometry column is already a partitioned `_Frame`, we can't refer to
    # it via a keyword-argument due to https://github.com/dask/dask/issues/8308.
    # Instead, we assign the geometry column using regular dataframe operations,
    # then refer to that column by name in `map_partitions`.
    if isinstance(geometry, dx.Series):
        name = geometry.name if geometry.name is not None else "geometry"
        return df.assign(**{name: geometry}).map_partitions(
            geopandas.GeoDataFrame, geometry=name
        )
    return df.map_partitions(geopandas.GeoDataFrame, geometry=geometry)


@derived_from(geopandas)
def points_from_xy(df, x="x", y="y", z="z", crs=None):
    """Convert dask.dataframe of x and y (and optionally z) values to a GeoSeries."""

    def func(data, x, y, z):
        return geopandas.GeoSeries(
            geopandas.points_from_xy(
                data[x], data[y], data[z] if z in df.columns else None, crs=crs
            ),
            index=data.index,
        )

    return df.map_partitions(
        func, x, y, z, meta=geopandas.GeoSeries(), token="points_from_xy"
    )


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
    "interiors",
    "bounds",
]:
    _Frame._bind_property(name)


for name in [
    "boundary",
    "centroid",
    "convex_hull",
    "envelope",
    "exterior",
]:
    # TODO actually calculate envelope / convex_hull of the spatial partitions
    # for some of those
    _Frame._bind_property(name, preserve_spatial_partitions=True)


for name in [
    "geometry",
    "x",
    "y",
    "z",
]:
    GeoSeries._bind_property(name)

for name in [
    "contains",
    "geom_equals",
    "geom_almost_equals",
    "crosses",
    "disjoint",
    "intersects",
    "overlaps",
    "touches",
    "within",
    "covers",
    "covered_by",
]:
    meth = getattr(geopandas.base.GeoPandasBase, name)
    _Frame._bind_elemwise_comparison_method(
        name, meth, original=geopandas.base.GeoPandasBase
    )


for name in [
    "distance",
    "difference",
    "symmetric_difference",
    "union",
    "intersection",
    "relate",
    "project",
]:
    meth = getattr(geopandas.base.GeoPandasBase, name)
    _Frame._bind_elemwise_operator_method(
        name, meth, original=geopandas.base.GeoPandasBase
    )


dd.DataFrame.set_geometry = GeoDataFrame.set_geometry


# Coodinate indexer (.cx)


def _cx_part(df, bbox):
    idx = df.intersects(bbox)
    return df[idx]


class _CoordinateIndexer(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, key):
        obj = self.obj
        xs, ys = key
        # handle numeric values as x and/or y coordinate index
        if type(xs) is not slice:
            xs = slice(xs, xs)
        if type(ys) is not slice:
            ys = slice(ys, ys)
        if xs.step is not None or ys.step is not None:
            raise ValueError("Slice step not supported.")

        if self.obj.spatial_partitions is not None:
            xmin, ymin, xmax, ymax = obj.spatial_partitions.total_bounds
            bbox = box(
                xs.start if xs.start is not None else xmin,
                ys.start if ys.start is not None else ymin,
                xs.stop if xs.stop is not None else xmax,
                ys.stop if ys.stop is not None else ymax,
            )
            partition_idx = np.nonzero(
                np.asarray(self.obj.spatial_partitions.intersects(bbox))
            )[0]
        else:
            raise NotImplementedError(
                "Not yet implemented if the GeoDataFrame has no known spatial "
                "partitions (you can call the 'calculate_spatial_partitions' method "
                "to set it)"
            )

        name = "cx-%s" % tokenize(key, self.obj)

        if len(partition_idx):
            # construct graph (based on LocIndexer from dask)
            dsk = {}
            for i, part in enumerate(partition_idx):
                dsk[name, i] = (_cx_part, (self.obj._name, part), bbox)

            divisions = [self.obj.divisions[i] for i in partition_idx] + [
                self.obj.divisions[partition_idx[-1] + 1]
            ]
        else:
            # TODO can a dask dataframe have 0 partitions?
            dsk = {(name, 0): self.obj._meta.head(0)}
            divisions = [None, None]

        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[self.obj])
        return from_graph(graph, self.obj._meta, divisions, dsk.keys(), "cx")


get_collection_type.register(geopandas.GeoDataFrame, lambda _: GeoDataFrame)
get_collection_type.register(geopandas.GeoSeries, lambda _: GeoSeries)
