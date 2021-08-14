import pytest
import pandas as pd
from pandas.testing import assert_index_equal
from hilbertcurve.hilbertcurve import HilbertCurve
from dask_geopandas.hilbert_distance import _continuous_to_discrete_coords
from dask_geopandas import from_geopandas
import geopandas
from shapely.geometry import Point, LineString, Polygon


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


def hilbert_distance_dask(geoseries):

    bounds = geoseries.bounds.to_numpy()
    total_bounds = geoseries.total_bounds
    coords = _continuous_to_discrete_coords(total_bounds, bounds, p=15)

    hilbert_curve = HilbertCurve(p=15, n=2)
    expected = hilbert_curve.distances_from_points(coords)

    ddf = from_geopandas(geoseries, npartitions=1)
    result = ddf.hilbert_distance().compute()

    assert list(result) == expected
    assert isinstance(result, pd.Series)
    assert_index_equal(ddf.index.compute(), result.index)


def test_hilbert_distance_points(geoseries_points):
    hilbert_distance_dask(geoseries_points)


def test_hilbert_distance_lines(geoseries_lines):
    hilbert_distance_dask(geoseries_lines)


def test_hilbert_distance_polygons(geoseries_polygons):
    hilbert_distance_dask(geoseries_polygons)
