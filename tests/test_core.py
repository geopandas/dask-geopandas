import pytest
import pandas as pd
import numpy as np
import geopandas
from shapely.geometry import Polygon, Point, LineString, MultiPoint
import dask.dataframe as dd
from dask.dataframe.core import Scalar
import dask_geopandas


@pytest.fixture
def geoseries_polygons():
    t1 = Polygon([(0, 3.5), (7, 2.4), (1, 0.1)])
    t2 = Polygon([(0, 0), (1, 1), (0, 1)])
    sq1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    sq2 = Polygon([(0, 0), (1, 0), (1, 2), (0, 2)])
    return geopandas.GeoSeries([t1, t2, sq1, sq2])


@pytest.fixture
def geoseries_points():
    p1 = Point(1, 2)
    p2 = Point(2, 3)
    p3 = Point(3, 4)
    p4 = Point(4, 1)
    return geopandas.GeoSeries([p1, p2, p3, p4])


@pytest.fixture
def geodf_points_crs():
    x = np.arange(-1683723, -1683723 + 10, 1)
    y = np.arange(6689139, 6689139 + 10, 1)
    crs = "epsg:26918"
    return geopandas.GeoDataFrame(
        {"geometry": geopandas.points_from_xy(x, y), "value1": x + y, "value2": x * y},
        crs=crs,
    )


@pytest.fixture
def geoseries_lines():
    l1 = LineString([(0, 0), (0, 1), (1, 1)])
    l2 = LineString([(0, 0), (1, 0), (1, 1), (0, 1)])
    return geopandas.GeoSeries([l1, l2] * 2)


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
        "geometry",
    ],
)
def test_geoseries_properties(geoseries_polygons, attr):
    original = getattr(geoseries_polygons, attr)

    dask_obj = dask_geopandas.from_geopandas(geoseries_polygons, npartitions=2)
    assert len(dask_obj.partitions[0]) < len(geoseries_polygons)
    assert isinstance(dask_obj, dask_geopandas.GeoSeries)

    daskified = getattr(dask_obj, attr)
    assert all(original == daskified.compute())


def test_geoseries_unary_union(geoseries_points):
    original = getattr(geoseries_points, "unary_union")

    dask_obj = dask_geopandas.from_geopandas(geoseries_points, npartitions=2)
    daskified = dask_obj.unary_union
    assert isinstance(daskified, Scalar)
    assert original.equals(daskified.compute())


@pytest.mark.parametrize(
    "attr", ["x", "y"],
)
def test_geoseries_points_attrs(geoseries_points, attr):
    original = getattr(geoseries_points, attr)

    dask_obj = dask_geopandas.from_geopandas(geoseries_points, npartitions=2)
    assert len(dask_obj.partitions[0]) < len(geoseries_points)
    assert isinstance(dask_obj, dask_geopandas.GeoSeries)

    daskified = getattr(dask_obj, attr)
    assert all(original == daskified.compute())


def test_points_from_xy():
    x = [1, 2, 3]
    y = [4, 5, 6]
    expected = geopandas.points_from_xy(x, y)
    df = dd.from_pandas(pd.DataFrame({"x": x, "y": y}), npartitions=2)
    actual = dask_geopandas.points_from_xy(df)
    assert isinstance(actual, dask_geopandas.GeoSeries)
    list(actual) == list(expected)


def test_geodataframe_crs(geodf_points_crs):
    df = geodf_points_crs
    original = df.crs

    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)
    assert dask_obj.crs == original
    assert dask_obj.partitions[1].crs == original

    new_crs = "epsg:4316"
    new = dask_obj.set_crs("epsg:4316")
    assert new.crs == new_crs
    assert new.partitions[1].crs == new_crs
    assert dask_obj.crs == original

    dask_obj.crs = new_crs
    assert dask_obj.crs == new_crs
    assert dask_obj.partitions[1].crs == new_crs


def test_project(geoseries_lines):
    s = geoseries_lines
    pt = Point(1.0, 0.5)

    original = s.project(pt)

    dask_obj = dask_geopandas.from_geopandas(s, npartitions=2)
    daskified = dask_obj.project(pt)

    assert isinstance(daskified, dd.Series)
    assert original.equals(daskified.compute())

    original = s.project(pt, normalized=True)
    daskified = dask_obj.project(pt, normalized=True)

    assert isinstance(daskified, dd.Series)
    assert original.equals(daskified.compute())


@pytest.mark.parametrize(
    "meth",
    [
        "contains",
        "geom_equals",
        "geom_almost_equals",
        "crosses",
        "disjoint",
        "intersects",
        "overlaps",
        "touches",
        "within",
        "distance",
        "relate",
    ],
)
def test_elemwise_methods(geoseries_polygons, geoseries_points, meth):
    one = geoseries_polygons
    other = geoseries_points
    original = getattr(one, meth)(other)

    dask_one = dask_geopandas.from_geopandas(one, npartitions=2)
    dask_other = dask_geopandas.from_geopandas(other, npartitions=2)
    daskified = getattr(dask_one, meth)(dask_other)

    assert isinstance(daskified, dd.Series)
    assert original.equals(daskified.compute())


def test_geom_equals_exact(geoseries_polygons, geoseries_points):
    meth = "geom_equals_exact"
    one = geoseries_polygons
    other = geoseries_points
    original = getattr(one, meth)(other, tolerance=2)

    dask_one = dask_geopandas.from_geopandas(one, npartitions=2)
    dask_other = dask_geopandas.from_geopandas(other, npartitions=2)
    daskified = getattr(dask_one, meth)(dask_other, tolerance=2)

    assert isinstance(daskified, dd.Series)
    assert original.equals(daskified.compute())


@pytest.mark.parametrize(
    "meth", ["difference", "symmetric_difference", "union", "intersection"]
)
def test_operator_methods(geoseries_polygons, geoseries_points, meth):
    one = geoseries_polygons
    other = geoseries_points
    original = getattr(one, meth)(other)

    dask_one = dask_geopandas.from_geopandas(one, npartitions=2)
    dask_other = dask_geopandas.from_geopandas(other, npartitions=2)
    daskified = getattr(dask_one, meth)(dask_other)

    assert isinstance(daskified, dd.Series)
    assert all(original == daskified.compute())


""


@pytest.mark.parametrize(
    "meth,options",
    [
        ("representative_point", {}),
        ("buffer", dict(distance=5)),
        ("simplify", dict(tolerance=3)),
        ("interpolate", dict(distance=2)),
        ("affine_transform", dict(matrix=[1, 2, -3, 5, 20, 30])),
        ("translate", dict(xoff=3, yoff=2, zoff=4)),
        ("rotate", dict(angle=90)),
        ("scale", dict(xfact=3, yfact=20, zfact=0.1)),
        ("skew", dict(xs=45, ys=-90, origin="centroid")),
    ],
)
def test_meth_with_args_and_kwargs(geoseries_lines, meth, options):
    s = geoseries_lines
    original = getattr(s, meth)(**options)

    dask_s = dask_geopandas.from_geopandas(s, npartitions=2)
    daskified = getattr(dask_s, meth)(**options)

    assert isinstance(daskified, dask_geopandas.GeoSeries)
    assert all(original == daskified.compute())


def test_explode_geoseries():
    s = geopandas.GeoSeries(
        [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])]
    )
    original = s.explode()
    dask_s = dask_geopandas.from_geopandas(s, npartitions=2)
    daskified = dask_s.explode()
    assert isinstance(daskified, dask_geopandas.GeoSeries)
    assert all(original == daskified.compute())


def test_explode_geodf():
    s = geopandas.GeoSeries(
        [MultiPoint([(0, 0), (1, 1)]), MultiPoint([(2, 2), (3, 3), (4, 4)])]
    )
    df = geopandas.GeoDataFrame({"col": [1, 2], "geometry": s})
    original = df.explode()
    dask_s = dask_geopandas.from_geopandas(df, npartitions=2)
    daskified = dask_s.explode()
    assert isinstance(daskified, dask_geopandas.GeoDataFrame)
    assert all(original == daskified.compute())
