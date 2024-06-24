import geopandas

import dask_geopandas

from .test_core import geodf_points  # noqa: F401

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


def test_clip(naturalearth_lowres, naturalearth_cities):
    cities = geopandas.read_file(naturalearth_cities)
    dask_obj = dask_geopandas.from_geopandas(cities, npartitions=4)
    dask_obj.calculate_spatial_partitions()
    mask = geopandas.read_file(naturalearth_lowres).query("continent == 'Africa'")
    expected = geopandas.clip(cities, mask)
    clipped = dask_geopandas.clip(dask_obj, mask)

    assert isinstance(clipped.spatial_partitions, geopandas.GeoSeries)

    result = clipped.compute()
    assert_geodataframe_equal(expected.sort_index(), result.sort_index())


def test_clip_no_spatial_partitions(geodf_points):  # noqa: F811
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    mask = geodf_points.iloc[:1]
    mask["geometry"] = mask["geometry"].buffer(2)
    expected = geodf_points.iloc[:2]
    result = dask_geopandas.clip(dask_obj, mask).compute()
    assert_geodataframe_equal(expected, result)


def test_clip_dask_mask(geodf_points):  # noqa: F811
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    mask = dask_geopandas.from_geopandas(geodf_points.iloc[:1], npartitions=1)
    with pytest.raises(
        NotImplementedError, match=r"Mask cannot be a Dask GeoDataFrame or GeoSeries."
    ):
        dask_geopandas.clip(dask_obj, mask)


def test_clip_geoseries(geodf_points):  # noqa: F811
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    dask_obj.calculate_spatial_partitions()
    mask = geodf_points.iloc[:1]
    mask["geometry"] = mask["geometry"].buffer(2)
    expected = geopandas.clip(geodf_points.geometry, mask)
    result = dask_geopandas.clip(dask_obj.geometry, mask).compute()
    assert_geoseries_equal(expected, result)
