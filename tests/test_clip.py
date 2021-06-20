import geopandas
from geopandas.testing import assert_geodataframe_equal
import dask_geopandas
from .test_core import geodf_points  # noqa: F401


def test_clip(geodf_points):  # noqa: F811
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    dask_obj.calculate_spatial_partitions()
    mask = geodf_points.iloc[:1]
    mask["geometry"] = mask["geometry"].buffer(2)
    expected = dask_obj.map_partitions(lambda gdf: geopandas.clip(gdf, mask)).compute()
    result = dask_geopandas.clip(dask_obj, mask).compute()
    assert_geodataframe_equal(expected, result)


def test_clip_no_spatial_partitions(geodf_points):  # noqa: F811
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    mask = geodf_points.iloc[:1]
    mask["geometry"] = mask["geometry"].buffer(2)
    expected = geodf_points.iloc[:2]
    result = dask_geopandas.clip(dask_obj, mask).compute()
    assert_geodataframe_equal(expected, result)
