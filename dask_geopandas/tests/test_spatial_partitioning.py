import numpy as np
import pytest

import geopandas
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from shapely.geometry import Point

import dask_geopandas
from dask_geopandas.hilbert_distance import _hilbert_distance


def test_propagate_on_geometry_access():
    # ensure the spatial_partitioning information is preserved in GeoSeries
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    ddf.calculate_spatial_partitions()
    spatial_partitions = ddf.spatial_partitions.copy()

    # geometry attribute
    gs = ddf.geometry
    assert gs.spatial_partitions is not None
    assert_geoseries_equal(gs.spatial_partitions, spatial_partitions)

    # column access
    gs = ddf["geometry"]
    assert gs.spatial_partitions is not None
    assert_geoseries_equal(gs.spatial_partitions, spatial_partitions)

    # subset geodataframe
    subset = ddf[["continent", "geometry"]]
    assert subset.spatial_partitions is not None
    assert_geoseries_equal(subset.spatial_partitions, spatial_partitions)


@pytest.mark.parametrize(
    "attr", ["boundary", "centroid", "convex_hull", "envelope", "exterior"]
)
def test_propagate_geoseries_properties(attr):
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    ddf.calculate_spatial_partitions()
    spatial_partitions = ddf.spatial_partitions.copy()

    result = getattr(ddf, attr)
    assert result.spatial_partitions is not None
    assert_geoseries_equal(result.spatial_partitions, spatial_partitions)
    assert_geoseries_equal(result.compute(), getattr(df, attr))


def test_cx():
    # test cx using spatial partitions
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    ddf.calculate_spatial_partitions()

    subset = ddf.cx[-180:-70, 0:-80]
    assert len(subset) == 8
    expected = df.cx[-180:-70, 0:-80]
    assert_geodataframe_equal(subset.compute(), expected)

    # empty slice
    subset = ddf.cx[-200:-190, 300:400]
    assert len(subset) == 0
    expected = df.cx[-200:-190, 300:400]
    assert_geodataframe_equal(subset.compute(), expected)


def test_geopandas_handles_large_hilbert_distances():
    df = geopandas.GeoDataFrame(
        {'geometry': [Point(-103152.516, -8942.156), Point(118914.500, 1010032.562)]}
    )

    # make sure we have values greater than 32bits
    dist = _hilbert_distance(df)
    assert ((dist > np.iinfo(np.int32).max) | (dist < np.iinfo(np.int32).min)).any()

    ddf = dask_geopandas.from_geopandas(df, npartitions=1)
    ddf.spatial_shuffle()
