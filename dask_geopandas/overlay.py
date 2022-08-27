import warnings

import numpy as np
import geopandas

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph

from .core import from_geopandas, GeoDataFrame


def overlay(df1, df2, how="intersection", **kwargs):
    """
    Overlay of two GeoDataFrames.

    Parameters
    ----------
    df1, df2 : geopandas or dask_geopandas GeoDataFrames
        If a geopandas.GeoDataFrame is passed, it is considered as a
        dask_geopandas.GeoDataFrame with 1 partition (without spatial
        partitioning information).
    how : string, default 'intersection'
        Method of spatial overlay: ‘intersection’, ‘union’, ‘identity’,
        ‘symmetric_difference’ or ‘difference’.
    keep_geom_type : bool
        If True, return only geometries of the same geometry type as df1 has,
        if False, return all resulting geometries. Default is None, which will
        set keep_geom_type to True but warn upon dropping geometries.
    make_validbool : default True
        If True, any invalid input geometries are corrected with a call to buffer(0),
        if False, a ValueError is raised if any input geometries are invalid.

    Returns
    -------
    dask_geopandas.GeoDataFrame

    Notes
    -----
    If both the df1 and df2 GeoDataFrame have spatial partitioning
    information available (the ``spatial_partitions`` attribute is set),
    the output partitions are determined based on intersection of the
    spatial partitions. In all other cases, the output partitions are
    all combinations (cartesian/cross product) of all input partition
    of the df1 and df2 GeoDataFrame.
    """

    if how != "inner":
        raise NotImplementedError("Only how='inner' is supported df2 now")

    if isinstance(df1, geopandas.GeoDataFrame):
        df1 = from_geopandas(df1, npartitions=1)
    if isinstance(df2, geopandas.GeoDataFrame):
        df2 = from_geopandas(df2, npartitions=1)

    name = "overlay-" + tokenize(df1, df2, how)
    meta = geopandas.overlay(df1._meta, df2._meta, how=how)

    if df1.spatial_partitions is not None and df2.spatial_partitions is not None:
        # Spatial partitions are known -> use them to trim down the list of
        # partitions that need to be joined
        parts = geopandas.sjoin(
            df1.spatial_partitions.to_frame("geometry"),
            df2.spatial_partitions.to_frame("geometry"),
            how="inner",
            predicate="intersects",
        )
        parts_df1 = np.asarray(parts.index)
        parts_df2 = np.asarray(parts["index_df2"].values)
        using_spatial_partitions = True
    else:
        # Unknown spatial partitions -> full cartesian (cross) product of all
        # combinations of the partitions of the df1 and df2 dataframe
        n_df1 = df1.npartitions
        n_df2 = df2.npartitions
        parts_df1 = np.repeat(np.arange(n_df1), n_df2)
        parts_df2 = np.tile(np.arange(n_df2), n_df1)
        using_spatial_partitions = False

    dsk = {}
    new_spatial_partitions = []
    for i, (l, r) in enumerate(zip(parts_df1, parts_df2)):
        dsk[(name, i)] = (
            geopandas.overlay,
            (df1._name, l),
            (df2._name, r),
            how,
        )
        # TODO preserve spatial partitions of the output if only df1 has spatial
        # partitions
        if using_spatial_partitions:
            lr = df1.spatial_partitions.iloc[l]
            rr = df2.spatial_partitions.iloc[r]
            # extent = lr.intersection(rr).buffer(buffer).intersection(lr.union(rr))
            extent = lr.intersection(rr)
            new_spatial_partitions.append(extent)

    divisions = [None] * (len(dsk) + 1)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[df1, df2])
    if not using_spatial_partitions:
        new_spatial_partitions = None
    return GeoDataFrame(graph, name, meta, divisions, new_spatial_partitions)
