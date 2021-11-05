from distutils.version import LooseVersion
import pytest
import pandas as pd
import numpy as np
import geopandas
from shapely.geometry import Polygon, Point, LineString, MultiPoint
import dask.dataframe as dd
from dask.dataframe.core import Scalar
import dask_geopandas

from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


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
def geoseries_points_crs(geoseries_points):
    s = geoseries_points
    s.crs = "epsg:26918"
    s.name = "geoseries_with_crs"
    return s


@pytest.fixture
def geodf_points():
    x = np.arange(-1683723, -1683723 + 10, 1)
    y = np.arange(6689139, 6689139 + 10, 1)
    return geopandas.GeoDataFrame(
        {"geometry": geopandas.points_from_xy(x, y), "value1": x + y, "value2": x * y},
    )


@pytest.fixture
def geodf_points_crs(geodf_points):
    geo_df = geodf_points
    crs = "epsg:26918"
    geo_df.crs = crs
    return geo_df


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


@pytest.mark.parametrize("attr", ["x", "y"])
def test_geoseries_points_attrs(geoseries_points, attr):
    original = getattr(geoseries_points, attr)

    dask_obj = dask_geopandas.from_geopandas(geoseries_points, npartitions=2)
    assert len(dask_obj.partitions[0]) < len(geoseries_points)
    assert isinstance(dask_obj, dask_geopandas.GeoSeries)

    daskified = getattr(dask_obj, attr)
    assert all(original == daskified.compute())


def test_points_from_xy():
    x = [1, 2, 3, 4, 5]
    y = [4, 5, 6, 7, 8]
    expected = geopandas.points_from_xy(x, y)
    df = pd.DataFrame({"x": x, "y": y})
    ddf = dd.from_pandas(df, npartitions=2)
    actual = dask_geopandas.points_from_xy(ddf)
    assert isinstance(actual, dask_geopandas.GeoSeries)
    assert list(actual) == list(expected)

    # assign to geometry column and convert to GeoDataFrame
    df["geometry"] = expected
    expected = geopandas.GeoDataFrame(df)
    ddf["geometry"] = actual
    ddf = dask_geopandas.from_dask_dataframe(ddf)
    result = ddf.compute()
    assert_geodataframe_equal(result, expected)


def test_geodataframe_crs(geodf_points_crs):
    df = geodf_points_crs
    original = df.crs

    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)
    assert dask_obj.crs == original
    assert dask_obj.partitions[1].crs == original

    new_crs = "epsg:4316"
    with pytest.raises(
        ValueError, match=r".*already has a CRS which is not equal to the passed CRS.*"
    ):
        dask_obj.set_crs(new_crs)

    new = dask_obj.set_crs(new_crs, allow_override=True)
    assert new.crs == new_crs
    assert new.partitions[1].crs == new_crs
    assert dask_obj.crs == original

    dask_obj.crs = new_crs
    assert dask_obj.crs == new_crs
    assert dask_obj.partitions[1].crs == new_crs
    assert dask_obj.compute().crs == new_crs


def test_geoseries_crs(geoseries_points_crs):
    s = geoseries_points_crs
    original = s.crs
    name = s.name

    dask_obj = dask_geopandas.from_geopandas(s, npartitions=2)
    assert dask_obj.crs == original
    assert dask_obj.partitions[1].crs == original
    assert dask_obj.compute().crs == original

    new_crs = "epsg:4316"
    with pytest.raises(
        ValueError, match=r".*already has a CRS which is not equal to the passed CRS.*"
    ):
        dask_obj.set_crs(new_crs)

    new = dask_obj.set_crs(new_crs, allow_override=True)
    assert new.crs == new_crs
    assert new.name == name
    assert new.partitions[1].crs == new_crs
    assert dask_obj.crs == original

    dask_obj.crs = new_crs
    assert dask_obj.crs == new_crs
    assert dask_obj.partitions[1].crs == new_crs
    assert dask_obj.name == name
    assert dask_obj.compute().crs == new_crs


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


def test_get_geometry_property_on_geodf(geodf_points):
    df = geodf_points
    df = df.rename(columns={"geometry": "foo"}).set_geometry("foo")
    assert set(df.columns) == {"value1", "value2", "foo"}
    assert all(df.geometry == df.foo)

    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)
    assert all(dask_obj.geometry == dask_obj.foo)


def test_set_geometry_property_on_geodf(geodf_points):
    df = geodf_points
    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)

    df = dask_obj.rename(columns={"geometry": "foo"}).set_geometry("foo").compute()
    assert set(df.columns) == {"value1", "value2", "foo"}
    assert all(df.geometry == df.foo)


def test_set_geometry_with_dask_geoseries():
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [1, 2, 3, 4]})
    dask_obj = dd.from_pandas(df, npartitions=2)
    dask_obj = dask_geopandas.from_dask_dataframe(dask_obj)
    dask_obj = dask_obj.set_geometry(dask_geopandas.points_from_xy(dask_obj, "x", "y"))
    expected = df.set_geometry(geopandas.points_from_xy(df["x"], df["y"]))
    assert_geoseries_equal(dask_obj.geometry.compute(), expected.geometry)


def test_meta(geodf_points_crs):
    df = geodf_points_crs
    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)

    def check_meta(gdf, name):
        assert isinstance(gdf, geopandas.GeoDataFrame)
        assert isinstance(gdf.geometry, geopandas.GeoSeries)
        assert gdf.crs == df.crs
        assert gdf._geometry_column_name == name

    meta = dask_obj._meta
    check_meta(meta, "geometry")
    meta_non_empty = dask_obj._meta_nonempty
    check_meta(meta_non_empty, "geometry")

    # with non-default geometry name
    df = df.rename_geometry("foo")
    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)
    meta = dask_obj._meta
    check_meta(meta, "foo")
    meta_non_empty = dask_obj._meta_nonempty
    check_meta(meta_non_empty, "foo")


def test_to_crs_geodf(geodf_points_crs):
    df = geodf_points_crs
    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)

    new_crs = "epsg:4316"
    new = dask_obj.to_crs(new_crs)
    assert new.crs == new_crs
    assert all(new.compute() == df.to_crs(new_crs))


def test_to_crs_geoseries(geoseries_points_crs):
    s = geoseries_points_crs
    dask_obj = dask_geopandas.from_geopandas(s, npartitions=2)

    new_crs = "epsg:4316"
    new = dask_obj.to_crs(new_crs)
    assert new.crs == new_crs
    assert all(new.compute() == s.to_crs(new_crs))


def test_copy_spatial_partitions(geodf_points):
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    dask_obj.calculate_spatial_partitions()
    dask_obj_copy = dask_obj.copy()
    pd.testing.assert_series_equal(
        dask_obj.spatial_partitions, dask_obj_copy.spatial_partitions
    )


def test_set_crs_sets_spatial_partition_crs(geodf_points):
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)

    dask_obj.calculate_spatial_partitions()
    dask_obj = dask_obj.set_crs("epsg:4326")

    assert dask_obj.crs == dask_obj.spatial_partitions.crs


def test_propagate_on_set_crs(geodf_points):
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)

    dask_obj.calculate_spatial_partitions()
    result = dask_obj.set_crs("epsg:4326").spatial_partitions
    expected = dask_obj.spatial_partitions.set_crs("epsg:4326")

    assert_geoseries_equal(result, expected)


@pytest.mark.skipif(
    LooseVersion(geopandas.__version__) <= LooseVersion("0.8.1"),
    reason="geopandas 0.8 has bug in apply",
)
def test_geoseries_apply(geoseries_polygons):
    # https://github.com/jsignell/dask-geopandas/issues/18
    ds = dask_geopandas.from_geopandas(geoseries_polygons, npartitions=2)
    result = ds.apply(lambda geom: geom.area, meta="float").compute()
    expected = geoseries_polygons.area
    pd.testing.assert_series_equal(result, expected)


def test_repr(geodf_points):
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    assert "Dask GeoDataFrame" in dask_obj.__repr__()
    assert "Dask GeoSeries" in dask_obj.geometry.__repr__()
    assert "Dask-GeoPandas GeoDataFrame" in dask_obj._repr_html_()


def test_map_partitions_get_geometry(geodf_points):
    # https://github.com/geopandas/dask-geopandas/issues/100
    df = geodf_points.rename_geometry("foo")
    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)

    def get_geometry(partition):
        return partition.geometry

    result = dask_obj.map_partitions(get_geometry).compute()
    expected = dask_obj.geometry.compute()
    assert_geoseries_equal(result, expected)


def test_copy_none_spatial_partitions(geoseries_points):
    ddf = dask_geopandas.from_geopandas(geoseries_points, npartitions=2)
    ddf.spatial_partitions = None
    ddf_copy = ddf.copy()
    assert ddf_copy.spatial_partitions is None


class TestDissolve:
    def setup_method(self):
        self.world = geopandas.read_file(
            geopandas.datasets.get_path("naturalearth_lowres")
        )
        self.ddf = dask_geopandas.from_geopandas(self.world, npartitions=4)

    def test_default(self):
        gpd_default = self.world.dissolve("continent")
        dd_default = self.ddf.dissolve("continent").compute()
        assert_geodataframe_equal(gpd_default, dd_default, check_like=True)

    def test_sum(self):
        gpd_sum = self.world.dissolve("continent", aggfunc="sum")
        dd_sum = self.ddf.dissolve("continent", aggfunc="sum").compute()
        # drop due to https://github.com/geopandas/geopandas/issues/1999
        assert_geodataframe_equal(
            gpd_sum, dd_sum.drop(columns=["name", "iso_a3"]), check_like=True
        )

    def test_split_out(self):
        gpd_default = self.world.dissolve("continent")
        dd_split = self.ddf.dissolve("continent", split_out=4)
        assert dd_split.npartitions == 4
        assert_geodataframe_equal(gpd_default, dd_split.compute(), check_like=True)

    def test_dict(self):
        aggfunc = {
            "pop_est": "min",
            "name": "first",
            "iso_a3": "first",
            "gdp_md_est": "sum",
        }
        gpd_dict = self.world.dissolve("continent", aggfunc=aggfunc)

        dd_dict = self.ddf.dissolve("continent", aggfunc=aggfunc).compute()
        assert_geodataframe_equal(gpd_dict, dd_dict, check_like=True)

    def test_by_none(self):
        gpd_none = self.world.dissolve()
        dd_none = self.ddf.dissolve().compute()
        assert_geodataframe_equal(gpd_none, dd_none, check_like=True)
