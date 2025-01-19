import numpy as np

from dask.base import tokenize
from dask.dataframe import from_graph
from dask.highlevelgraph import HighLevelGraph
from dask.utils import derived_from

import geopandas


@derived_from(geopandas.tools)
def clip(gdf, mask, keep_geom_type=False):
    from dask_geopandas import GeoDataFrame, GeoSeries

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
        (name, i): (geopandas.clip, (gdf._name, part), mask, keep_geom_type)
        for i, part in enumerate(intersecting_partitions)
    }
    divisions = [None] * (len(dsk) + 1)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[gdf])

    result = from_graph(graph, gdf._meta, tuple(divisions), dsk.keys(), "clip")

    result.spatial_partitions = new_spatial_partitions
    return result
