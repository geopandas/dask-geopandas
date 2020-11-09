import pytest

import geopandas
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal

import dask_geopandas


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
