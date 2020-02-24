import pytest
import pandas as pd
import geopandas
from shapely.geometry import Polygon
import dask.dataframe as dd
from dask.dataframe.core import Scalar
import dask_geopandas


@pytest.fixture
def geoseries():
    t1 = Polygon([(0, 3.5), (7, 2.4), (1, 0.1)])
    t2 = Polygon([(0, 0), (1, 1), (0, 1)])
    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    return geopandas.GeoSeries([t1, t2, sq] * 10)


@pytest.mark.parametrize(
    "attr",
    [
        "area",
        "geom_type",
        "type",
        "length",
        "is_valid",
        "is_empty",
        "is_simple",
        "is_ring",
        "has_z",
        "boundary",
        "centroid",
        "convex_hull",
        "envelope",
        "exterior",
        "interiors",
        "bounds",
        "total_bounds",
        # "sindex",
    ],
)
def test_geoseries_properties(geoseries, attr):
    original = getattr(geoseries, attr)

    dask_obj = dask_geopandas.from_geopandas(geoseries, npartitions=2)
    assert len(dask_obj.partitions[0]) < len(geoseries)
    assert isinstance(dask_obj, dask_geopandas.GeoSeries)

    daskified = getattr(dask_obj, attr)
    assert all(original == daskified.compute())


def test_geoseries_unary_union(geoseries):
    original = getattr(geoseries, "unary_union")

    dask_obj = dask_geopandas.from_geopandas(geoseries, npartitions=2)
    daskified = dask_obj.unary_union
    assert isinstance(daskified, Scalar)
    assert original.equals(daskified.compute())


def test_points_from_xy():
    x = [1, 2, 3]
    y = [4, 5, 6]
    expected = geopandas.points_from_xy(x, y)
    df = dd.from_pandas(pd.DataFrame({"x": x, "y": y}), npartitions=2)
    actual = dask_geopandas.points_from_xy(df)
    assert isinstance(actual, dask_geopandas.GeoSeries)
    list(actual) == list(expected)
