import pytest
import pandas as pd
import numpy as np

from pandas.testing import assert_index_equal, assert_series_equal
from dask_geopandas.hilbert_distance import (
    _hilbert_distance,
    _continuous_to_discrete_coords,
)
from dask_geopandas import from_geopandas
import geopandas
from shapely.geometry import Point, LineString, Polygon
from shapely.wkt import loads


def test_hilbert_distance():
    # test the actual Hilbert Code algorithm against some hardcoded values
    geoms = geopandas.GeoSeries.from_wkt(
        [
            "POINT (0 0)",
            "POINT (1 1)",
            "POINT (1 0)",
            "POLYGON ((0 0, 0 1, 1 1, 1 0, 0 0))",
        ]
    )
    result = _hilbert_distance(geoms, total_bounds=(0, 0, 1, 1), level=2)
    assert result.tolist() == [0, 10, 15, 2]

    result = _hilbert_distance(geoms, total_bounds=(0, 0, 1, 1), level=3)
    assert result.tolist() == [0, 42, 63, 10]

    result = _hilbert_distance(geoms, total_bounds=(0, 0, 1, 1), level=16)
    assert result.tolist() == [0, 2863311530, 4294967295, 715827882]


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


def hilbert_distance_dask(geoseries, level=16):
    pytest.importorskip("hilbertcurve")
    from hilbertcurve.hilbertcurve import HilbertCurve

    bounds = geoseries.bounds.to_numpy()
    total_bounds = geoseries.total_bounds
    x, y = _continuous_to_discrete_coords(
        bounds, level=level, total_bounds=total_bounds
    )
    coords = np.stack((x, y), axis=1)

    hilbert_curve = HilbertCurve(p=level, n=2)
    expected = hilbert_curve.distances_from_points(coords)

    ddf = from_geopandas(geoseries, npartitions=1)
    result = ddf.hilbert_distance(level=level).compute()

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
        ddf.hilbert_distance(level=20).compute()


def test_specified_total_bounds(geoseries_polygons):
    ddf = from_geopandas(geoseries_polygons, npartitions=2)

    result = ddf.hilbert_distance(total_bounds=geoseries_polygons.total_bounds)
    expected = ddf.hilbert_distance()
    assert_series_equal(result.compute(), expected.compute())


def test_total_bounds_from_partitions(geoseries_polygons):
    ddf = from_geopandas(geoseries_polygons, npartitions=2)
    expected = ddf.hilbert_distance().compute()

    ddf.calculate_spatial_partitions()
    result = ddf.hilbert_distance().compute()
    assert_series_equal(result, expected)


def test_world(naturalearth_lowres):
    # world without Fiji
    hilbert_distance_dask(geopandas.read_file(naturalearth_lowres).iloc[1:])


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
        dask_obj.hilbert_distance().compute()
