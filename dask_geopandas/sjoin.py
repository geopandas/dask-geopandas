import numpy as np
import geopandas

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from .core import from_geopandas, GeoDataFrame


def sjoin(left, right, how="inner", op="intersects"):
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
    op : string, default 'intersects'
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
    if how != "inner":
        raise NotImplementedError("Only how='inner' is supported right now")

    if isinstance(left, geopandas.GeoDataFrame):
        left = from_geopandas(left, npartitions=1)
    if isinstance(right, geopandas.GeoDataFrame):
        right = from_geopandas(right, npartitions=1)

    name = "sjoin-" + tokenize(left, right, how, op)
    meta = geopandas.sjoin(left._meta, right._meta, how=how, op=op)

    if left.spatial_partitions is not None and right.spatial_partitions is not None:
        # Spatial partitions are known -> use them to trim down the list of
        # partitions that need to be joined
        parts = geopandas.sjoin(
            left.spatial_partitions.to_frame("geometry"),
            right.spatial_partitions.to_frame("geometry"),
            how="inner",
            op="intersects",
        )
        parts_left = parts.index.values
        parts_right = parts["index_right"].values
        # Sub-select just the partitions from each input we need
        left_sub = left.partitions[parts_left]
        right_sub = right.partitions[parts_right]

        joined = left_sub.map_partitions(
            geopandas.sjoin,
            right_sub,
            how,
            op,
            enforce_metadata=False,
            transform_divisions=False,
            align_dataframes=False,
            meta=meta,
        )

        # TODO preserve spatial partitions of the output if only left has spatial
        # partitions
        joined.spatial_partitions = [
            left.spatial_partitions.iloc[l].intersection(
                right.spatial_partitions.iloc[r]
            )
            for l, r in zip(parts_left, parts_right)
        ]
        return joined

    # Unknown spatial partitions -> full cartesian (cross) product of all
    # combinations of the partitions of the left and right dataframe
    n_left = left.npartitions
    n_right = right.npartitions
    parts_left = np.repeat(np.arange(n_left), n_right)
    parts_right = np.tile(np.arange(n_right), n_left)

    dsk = {}
    for i, (l, r) in enumerate(zip(parts_left, parts_right)):
        dsk[(name, i)] = (geopandas.sjoin, (left._name, l), (right._name, r), how, op)

    divisions = [None] * (len(dsk) + 1)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[left, right])
    return GeoDataFrame(graph, name, meta, divisions, None)
