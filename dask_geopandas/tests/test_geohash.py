import pytest
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_index_equal
from dask_geopandas.geohash import _calculate_mid_points
from dask_geopandas import from_geopandas
import geopandas
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads


@pytest.fixture
def geoseries_points():
    p1 = Point(1, 2)
    p2 = Point(2, 3)
    p3 = Point(3, 4)
    p4 = Point(4, 1)
    return geopandas.GeoSeries([p1, p2, p3, p4])


@pytest.fixture
def geoseries_lines():
    l1 = LineString([(0, 0), (0, 1), (1, 1)])
    l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
    return geopandas.GeoSeries([l1, l2] * 2)


@pytest.fixture()
def geoseries_polygons():
    t1 = Polygon([(0, 3.5), (7, 2.4), (1, 0.1)])
    t2 = Polygon([(0, 0), (1, 1), (0, 1)])
    sq1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    sq2 = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
    return geopandas.GeoSeries([t1, t2, sq1, sq2])


def geohash_dask(geoseries):
    pygeohash = pytest.importorskip("pygeohash")

    p = 12
    as_string = True
    bounds = geoseries.bounds.to_numpy()
    x_mids, y_mids = _calculate_mid_points(bounds)

    geohash_vec = np.vectorize(pygeohash.encode)
    # Encode mid points of geometries using geohash
    expected = geohash_vec(y_mids, x_mids, p)

    ddf = from_geopandas(geoseries, npartitions=1)
    result = ddf.geohash(precision=p, as_string=as_string).compute()

    assert_array_equal(np.array(result), expected)
    assert isinstance(result, pd.Series)
    assert_index_equal(ddf.index.compute(), result.index)


def test_geohash_points(geoseries_points):
    geohash_dask(geoseries_points)


def test_geohash_lines(geoseries_lines):
    geohash_dask(geoseries_lines)


def test_geohash_polygons(geoseries_polygons):
    geohash_dask(geoseries_polygons)


def test_geohash_range(geoseries_points):

    ddf = from_geopandas(geoseries_points, npartitions=1)

    with pytest.raises(ValueError):
        ddf.geohash(precision=0, as_string=False)
        ddf.geohash(precision=12, as_string=False)


def test_world():
    # world without Fiji
    geohash_dask(
        geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres")).iloc[1:]
    )


@pytest.mark.parametrize(
    "empty",
    [
        None,
        loads("POLYGON EMPTY"),
    ],
)
def test_empty(geoseries_polygons, empty):
    s = geoseries_polygons
    s.iloc[-1] = empty
    dask_obj = from_geopandas(s, npartitions=2)
    with pytest.raises(
        ValueError, match="cannot be computed on a GeoSeries with empty"
    ):
        dask_obj.geohash().compute()
