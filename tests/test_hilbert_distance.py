import pytest
import pandas as pd
import numpy as np

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


def hilbert_distance_dask(geoseries, level=15):

    bounds = geoseries.bounds.to_numpy()
    total_bounds = geoseries.total_bounds
    x, y = _continuous_to_discrete_coords(bounds, p=level, total_bounds=total_bounds)
    coords = np.stack((x, y), axis=1)

    hilbert_curve = HilbertCurve(p=level, n=2)
    expected = hilbert_curve.distances_from_points(coords)

    ddf = from_geopandas(geoseries, npartitions=1)
    result = ddf.hilbert_distance(p=level).compute()

    assert list(result) == expected
    assert isinstance(result, pd.Series)
    assert_index_equal(ddf.index.compute(), result.index)


@pytest.mark.parametrize("level", [2, 10, 15, 16])
def test_hilbert_distance_points(geoseries_points, level):
    hilbert_distance_dask(geoseries_points, level)


@pytest.mark.parametrize("level", [2, 10, 15, 16])
def test_hilbert_distance_lines(geoseries_lines, level):
    hilbert_distance_dask(geoseries_lines, level)


@pytest.mark.parametrize("level", [2, 10, 15, 16])
def test_hilbert_distance_polygons(geoseries_polygons, level):
    hilbert_distance_dask(geoseries_polygons, level)


def test_hilbert_distance_level(geoseries_points):
    ddf = from_geopandas(geoseries_points, npartitions=1)
    with pytest.raises(ValueError):
        ddf.hilbert_distance(p=20).compute()
