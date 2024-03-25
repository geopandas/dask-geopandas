import geopandas
from geopandas.testing import assert_geodataframe_equal

import dask_geopandas


def test_overlay_dask_geopandas():
    world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    capitals = geopandas.read_file(geopandas.datasets.get_path("naturalearth_cities"))

    # Select South America and some columns
    countries = world[world["continent"] == "South America"]
    countries = countries[["geometry", "name"]]

    # Project to crs that uses meters as distance measure
    countries = countries.to_crs("epsg:3395")
    capitals = capitals.to_crs("epsg:3395")

    ddf_countries = dask_geopandas.from_geopandas(countries, npartitions=4)
    ddf_capitals = dask_geopandas.from_geopandas(capitals, npartitions=4)

    expected = geopandas.overlay(capitals, countries)
    expected = expected.sort_values("name_1").reset_index(drop=True)

    # dask / geopandas
    result = dask_geopandas.overlay(ddf_capitals, countries)
    assert_geodataframe_equal(
        expected, result.compute().sort_values("name_1").reset_index(drop=True)
    )

    # geopandas / dask
    result = dask_geopandas.overlay(capitals, ddf_countries)
    assert_geodataframe_equal(
        expected, result.compute().sort_values("name_1").reset_index(drop=True)
    )

    # dask / dask
    result = dask_geopandas.overlay(ddf_capitals, ddf_countries)
    assert_geodataframe_equal(
        expected, result.compute().sort_values("name_1").reset_index(drop=True)
    )

    # with spatial_partitions
    ddf_countries.calculate_spatial_partitions()
    ddf_capitals.calculate_spatial_partitions()
    result = dask_geopandas.overlay(ddf_capitals, ddf_countries)
    assert isinstance(result.spatial_partitions, geopandas.GeoSeries)
    assert_geodataframe_equal(
        expected, result.compute().sort_values("name_1").reset_index(drop=True)
    )
