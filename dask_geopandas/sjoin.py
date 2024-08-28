import warnings

import numpy as np

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

import geopandas

from . import backends

if backends.QUERY_PLANNING_ON:
    from .expr import from_geopandas
else:
    from .core import from_geopandas


def sjoin(left, right, how="inner", predicate="intersects", **kwargs):
    """
    Spatial join of two GeoDataFrames.

    Parameters
    ----------
    left, right : geopandas or dask_geopandas GeoDataFrames
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
    if "op" in kwargs:
        predicate = kwargs.pop("op")
        deprecation_message = (
            "The `op` parameter is deprecated and will be removed"
            " in a future release. Please use the `predicate` parameter"
            " instead."
        )
        warnings.warn(deprecation_message, FutureWarning, stacklevel=2)
    if how != "inner":
        raise NotImplementedError("Only how='inner' is supported right now")

    if isinstance(left, geopandas.GeoDataFrame):
        left = from_geopandas(left, npartitions=1)
    if isinstance(right, geopandas.GeoDataFrame):
        right = from_geopandas(right, npartitions=1)

    if backends.QUERY_PLANNING_ON:
        # We call optimize on the inputs to ensure that any optimizations
        # done by dask-expr (which might change the expression, and thus the
        # name of the DataFrame) *before* we build the HighLevelGraph.
        # https://github.com/dask/dask-expr/issues/1129
        left = left.optimize()
        right = right.optimize()

    name = "sjoin-" + tokenize(left, right, how, predicate)
    meta = geopandas.sjoin(left._meta, right._meta, how=how, predicate=predicate)

    if left.spatial_partitions is not None and right.spatial_partitions is not None:
        # Spatial partitions are known -> use them to trim down the list of
        # partitions that need to be joined
        parts = geopandas.sjoin(
            left.spatial_partitions.to_frame("geometry"),
            right.spatial_partitions.to_frame("geometry"),
            how="inner",
            predicate="intersects",
        )
        parts_left = np.asarray(parts.index).tolist()
        parts_right = np.asarray(parts["index_right"].values).tolist()
        using_spatial_partitions = True
    else:
        # Unknown spatial partitions -> full cartesian (cross) product of all
        # combinations of the partitions of the left and right dataframe
        n_left = left.npartitions
        n_right = right.npartitions
        parts_left = np.repeat(np.arange(n_left), n_right)
        parts_right = np.tile(np.arange(n_right), n_left)
        using_spatial_partitions = False

    dsk = {}
    new_spatial_partitions = []
    for i, (part_left, part_right) in enumerate(zip(parts_left, parts_right)):
        dsk[(name, i)] = (
            geopandas.sjoin,
            (left._name, part_left),
            (right._name, part_right),
            how,
            predicate,
        )
        # TODO preserve spatial partitions of the output if only left has spatial
        # partitions
        if using_spatial_partitions:
            lr = left.spatial_partitions.iloc[part_left]
            rr = right.spatial_partitions.iloc[part_right]
            # extent = lr.intersection(rr).buffer(buffer).intersection(lr.union(rr))
            extent = lr.intersection(rr)
            new_spatial_partitions.append(extent)

    divisions = [None] * (len(dsk) + 1)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[left, right])
    if using_spatial_partitions:
        new_spatial_partitions = geopandas.GeoSeries(
            data=new_spatial_partitions, crs=left.crs
        )
    else:
        new_spatial_partitions = None

    if backends.QUERY_PLANNING_ON:
        from dask_expr import from_graph

        result = from_graph(graph, meta, divisions, dsk.keys(), "sjoin")
        result.spatial_partitions = new_spatial_partitions
        return result
    else:
        from .core import GeoDataFrame

        return GeoDataFrame(graph, name, meta, divisions, new_spatial_partitions)
