import geopandas
import shapely

import dask_geopandas

import pytest
from geopandas.testing import assert_geodataframe_equal


def test_sjoin_dask_geopandas(naturalearth_lowres, naturalearth_cities):
    df_points = geopandas.read_file(naturalearth_cities)
    ddf_points = dask_geopandas.from_geopandas(df_points, npartitions=4)

    df_polygons = geopandas.read_file(naturalearth_lowres)
    ddf_polygons = dask_geopandas.from_geopandas(df_polygons, npartitions=4)

    expected = geopandas.sjoin(df_points, df_polygons, predicate="within", how="inner")
    expected = expected.sort_index()

    # dask / geopandas
    result = dask_geopandas.sjoin(
        ddf_points, df_polygons, predicate="within", how="inner"
    )
    assert_geodataframe_equal(expected, result.compute().sort_index())

    # geopandas / dask
    result = dask_geopandas.sjoin(
        df_points, ddf_polygons, predicate="within", how="inner"
    )
    assert_geodataframe_equal(expected, result.compute().sort_index())

    # dask / dask
    result = dask_geopandas.sjoin(
        ddf_points, ddf_polygons, predicate="within", how="inner"
    )
    assert_geodataframe_equal(expected, result.compute().sort_index())

    # with spatial_partitions
    ddf_points.calculate_spatial_partitions()
    ddf_polygons.calculate_spatial_partitions()
    result = dask_geopandas.sjoin(
        ddf_points, ddf_polygons, predicate="within", how="inner"
    )
    assert isinstance(result.spatial_partitions, geopandas.GeoSeries)
    assert_geodataframe_equal(expected, result.compute().sort_index())

    # check warning
    with pytest.warns(FutureWarning, match="The `op` parameter is deprecated"):
        dask_geopandas.sjoin(df_points, ddf_polygons, op="within", how="inner")


def test_sjoin_lsuffix_rsuffix():
    left_gdf = geopandas.GeoDataFrame(
        {"dup_col": [1], "geometry": [shapely.geometry.box(0, 0, 1, 1)]},
    )
    right_gdf = geopandas.GeoDataFrame(
        {"dup_col": [2], "geometry": [shapely.geometry.box(0.5, 0.5, 1.5, 1.5)]},
    )
    lsuffix, rsuffix = "_L", "_R"
    expected = geopandas.sjoin(
        left_gdf,
        right_gdf,
        how="inner",
        predicate="intersects",
        lsuffix=lsuffix,
        rsuffix=rsuffix,
    ).sort_index()

    ddf_left = dask_geopandas.from_geopandas(left_gdf, npartitions=2)
    ddf_right = dask_geopandas.from_geopandas(right_gdf, npartitions=2)

    kw = dict(
        how="inner",
        predicate="intersects",
        lsuffix=lsuffix,
        rsuffix=rsuffix,
    )

    # dask / geopandas
    result = dask_geopandas.sjoin(ddf_left, right_gdf, **kw)
    got = result.compute().sort_index()
    assert list(got.columns) == list(expected.columns)
    assert_geodataframe_equal(expected, got)

    # geopandas / dask
    result = dask_geopandas.sjoin(left_gdf, ddf_right, **kw)
    got = result.compute().sort_index()
    assert list(got.columns) == list(expected.columns)
    assert_geodataframe_equal(expected, got)

    # dask / dask
    result = dask_geopandas.sjoin(ddf_left, ddf_right, **kw)
    got = result.compute().sort_index()
    assert list(got.columns) == list(expected.columns)
    assert_geodataframe_equal(expected, got)

    assert_geodataframe_equal(
        expected,
        ddf_left.sjoin(ddf_right, **kw).compute().sort_index(),
    )


def test_no_value_error():
    # https://github.com/geopandas/dask-geopandas/issues/303
    shape = shapely.geometry.box(-74.5, -74.0, 4.5, 5.0)
    df = dask_geopandas.from_geopandas(
        geopandas.GeoDataFrame(geometry=[shape]), npartitions=1
    ).spatial_shuffle()
    # no TypeError
    df.sjoin(df).compute()
