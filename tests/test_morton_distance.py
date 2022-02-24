import pytest
import pandas as pd
from pandas.testing import assert_index_equal, assert_series_equal
from pymorton import interleave  # https://github.com/trevorprater/pymorton
from dask_geopandas.hilbert_distance import _continuous_to_discrete_coords
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


def morton_distance_dask(geoseries):

    bounds = geoseries.bounds.to_numpy()
    total_bounds = geoseries.total_bounds
    x_coords, y_coords = _continuous_to_discrete_coords(
        bounds, level=16, total_bounds=total_bounds
    )

    ddf = from_geopandas(geoseries, npartitions=1)
    result = ddf.morton_distance().compute()

    expected = []

    for i in range(len(x_coords)):
        x = int(x_coords[i])
        y = int(y_coords[i])
        expected.append(interleave(x, y))

    assert list(result) == expected
    assert isinstance(result, pd.Series)
    assert_index_equal(ddf.index.compute(), result.index)


def test_morton_distance_points(geoseries_points):
    morton_distance_dask(geoseries_points)


def test_morton_distance_lines(geoseries_lines):
    morton_distance_dask(geoseries_lines)


def test_morton_distance_polygons(geoseries_polygons):
    morton_distance_dask(geoseries_polygons)


def test_specified_total_bounds(geoseries_polygons):
    ddf = from_geopandas(geoseries_polygons, npartitions=2)

    result = ddf.morton_distance(total_bounds=geoseries_polygons.total_bounds)
    expected = ddf.morton_distance()
    assert_series_equal(result.compute(), expected.compute())


def test_total_bounds_from_partitions(geoseries_polygons):
    ddf = from_geopandas(geoseries_polygons, npartitions=2)
    expected = ddf.morton_distance().compute()

    ddf.calculate_spatial_partitions()
    result = ddf.morton_distance().compute()
    assert_series_equal(result, expected)


def test_world():
    # world without Fiji
    morton_distance_dask(
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
        dask_obj.morton_distance().compute()
