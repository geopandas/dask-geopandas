import geopandas
import dask.dataframe as dd
import dask_geopandas

import pytest
from pandas.testing import assert_series_equal
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


pytest.importorskip("pyogrio")


def test_read_file():
    path = geopandas.datasets.get_path("naturalearth_lowres")
    df = geopandas.read_file(path)
    result = dask_geopandas.read_file(path, npartitions=4)
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 4
    assert result.crs == df.crs
    assert_geodataframe_equal(result.compute().reset_index(drop=True), df)

    result = dask_geopandas.read_file(path, chunksize=100)
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 2
    assert result.crs == df.crs
    assert_geodataframe_equal(result.compute().reset_index(drop=True), df)

    msg = "Exactly one of npartitions and chunksize must be specified"
    with pytest.raises(ValueError, match=msg):
        dask_geopandas.read_file(path)
    with pytest.raises(ValueError, match=msg):
        dask_geopandas.read_file(path, npartitions=4, chunksize=100)


def test_read_file_columns():
    path = geopandas.datasets.get_path("naturalearth_lowres")
    df = geopandas.read_file(path)

    # explicit column selection
    result = dask_geopandas.read_file(path, npartitions=4, columns=["pop_est"])
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 4
    assert result.crs == df.crs
    assert_geodataframe_equal(
        result.compute().reset_index(drop=True), df[["pop_est", "geometry"]]
    )

    # column selection through getitem
    ddf = dask_geopandas.read_file(path, npartitions=4)
    result = ddf[["pop_est", "geometry"]]
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 4
    assert result.crs == df.crs
    assert_geodataframe_equal(
        result.compute().reset_index(drop=True), df[["pop_est", "geometry"]]
    )

    # only select non-geometry column
    result = ddf["pop_est"]
    assert isinstance(result, dd.Series)
    assert_series_equal(result.compute().reset_index(drop=True), df["pop_est"])

    # only select geometry column
    result = ddf["geometry"]
    assert isinstance(result, dask_geopandas.GeoSeries)
    assert_geoseries_equal(result.compute().reset_index(drop=True), df["geometry"])
