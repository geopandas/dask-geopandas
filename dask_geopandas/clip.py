import numpy as np
import geopandas

from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from
from dask.base import tokenize

from . import backends


@derived_from(geopandas.tools)
def clip(gdf, mask, keep_geom_type=False):

    if backends.QUERY_PLANNING_ON:
        from .expr import GeoDataFrame, GeoSeries
    else:
        from .core import GeoDataFrame, GeoSeries

    if isinstance(mask, (GeoDataFrame, GeoSeries)):
        raise NotImplementedError("Mask cannot be a Dask GeoDataFrame or GeoSeries.")

    if gdf.spatial_partitions is None:
        return gdf.map_partitions(
            lambda partition: geopandas.clip(
                gdf=partition, mask=mask, keep_geom_type=keep_geom_type
            ),
            token="clip",
            meta=gdf._meta,
        )

    new_spatial_partitions = geopandas.clip(
        gdf=gdf.spatial_partitions,
        mask=mask,
        # keep_geom_type is always false for clipping the spatial partitions
        # otherwise we'd be falsely creating new partition(s)
        keep_geom_type=False,
    )
    intersecting_partitions = np.asarray(new_spatial_partitions.index)

    name = f"clip-{tokenize(gdf, mask, keep_geom_type)}"
    dsk = {
        (name, i): (geopandas.clip, (gdf._name, l), mask, keep_geom_type)
        for i, l in enumerate(intersecting_partitions)
    }
    divisions = [None] * (len(dsk) + 1)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[gdf])
    if backends.QUERY_PLANNING_ON:
        from dask_expr import from_graph

        result = from_graph(graph, gdf._meta, tuple(divisions), dsk.keys(), "clip")
    else:
        from .core import GeoDataFrame, GeoSeries

        if isinstance(gdf, GeoDataFrame):
            result = GeoDataFrame(graph, name, gdf._meta, tuple(divisions))
        elif isinstance(gdf, GeoSeries):
            result = GeoSeries(graph, name, gdf._meta, tuple(divisions))

    result.spatial_partitions = new_spatial_partitions
    return result
