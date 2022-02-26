import random

import geopandas
from shapely.geometry import Polygon
import dask.dataframe as dd
import dask_geopandas
import pandas as pd

import pytest
from pandas.testing import assert_series_equal, assert_frame_equal
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


pytest.importorskip("pyogrio")


def test_read_file():
    path = geopandas.datasets.get_path("naturalearth_lowres")
    df = geopandas.read_file(path)
    result = dask_geopandas.read_file(path, npartitions=4)
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 4
    assert result.crs == df.crs
    assert_geodataframe_equal(result.compute(), df)

    result = dask_geopandas.read_file(path, chunksize=100)
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 2
    assert result.crs == df.crs
    assert_geodataframe_equal(result.compute(), df)

    msg = "Exactly one of npartitions and chunksize must be specified"
    with pytest.raises(ValueError, match=msg):
        dask_geopandas.read_file(path)
    with pytest.raises(ValueError, match=msg):
        dask_geopandas.read_file(path, npartitions=4, chunksize=100)


def test_read_file_divisions():
    path = geopandas.datasets.get_path("naturalearth_lowres")
    result = dask_geopandas.read_file(path, npartitions=4)
    assert result.known_divisions
    assert result.index.divisions == (0, 45, 90, 135, 176)
    assert result.divisions == (0, 45, 90, 135, 176)


def test_read_file_index():
    path = geopandas.datasets.get_path("naturalearth_lowres")
    df = geopandas.read_file(path)
    result = dask_geopandas.read_file(path, npartitions=4)
    assert (result.index.compute() == pd.RangeIndex(0, len(df))).all()


def test_read_file_columns():
    path = geopandas.datasets.get_path("naturalearth_lowres")
    df = geopandas.read_file(path)

    # explicit column selection
    result = dask_geopandas.read_file(
        path, npartitions=4, columns=["pop_est", "geometry"]
    )
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 4
    assert result.crs == df.crs
    assert len(result.columns) == 2
    assert_geodataframe_equal(
        result.compute(), df[["pop_est", "geometry"]]
    )
    # only selecting non-geometry column
    result = dask_geopandas.read_file(path, npartitions=4, columns=["pop_est"])
    assert type(result) == dd.DataFrame
    assert len(result.columns) == 1
    assert result.npartitions == 4
    assert_frame_equal(result.compute(), df[["pop_est"]])

    # column selection through getitem
    ddf = dask_geopandas.read_file(path, npartitions=4)
    result = ddf[["pop_est", "geometry"]]
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 4
    assert result.crs == df.crs
    assert_geodataframe_equal(result.compute(), df[["pop_est", "geometry"]])

    # only select non-geometry column
    result = ddf["pop_est"]
    assert isinstance(result, dd.Series)
    assert_series_equal(result.compute(), df["pop_est"])

    # only select geometry column
    result = ddf["geometry"]
    assert isinstance(result, dask_geopandas.GeoSeries)
    assert_geoseries_equal(result.compute(), df["geometry"])


def test_read_file_layer(tmp_path):
    df_points = geopandas.GeoDataFrame(
        {
            "col": [1, 2, 3, 4],
            "geometry": geopandas.points_from_xy([1, 2, 3, 4], [2, 3, 4, 1]),
        },
        crs=4326,
    )
    df_polygons = geopandas.GeoDataFrame(
        {
            "col": [5, 6, 7, 8],
            "geometry": [
                Polygon([(random.random(), random.random()) for i in range(3)])
                for _ in range(4)
            ],
        },
        crs=4326,
    )

    path = tmp_path / "test_layers.gpkg"
    df_points.to_file(path, layer="points")
    df_polygons.to_file(path, layer="polygons")

    ddf_points = dask_geopandas.read_file(path, npartitions=2, layer="points")
    assert_geodataframe_equal(ddf_points.compute(), df_points)
    ddf_polygons = dask_geopandas.read_file(path, npartitions=2, layer="polygons")
    assert_geodataframe_equal(ddf_polygons.compute(), df_polygons)
