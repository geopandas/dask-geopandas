import pickle
from packaging.version import Version
import pytest
import pandas as pd
import numpy as np
import geopandas
from shapely.geometry import Polygon, Point, LineString, MultiPoint
import dask
import dask.dataframe as dd
from dask.dataframe.core import Scalar
import dask_geopandas

from pandas.testing import assert_frame_equal, assert_series_equal
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal
from dask_geopandas.hilbert_distance import _hilbert_distance
from dask_geopandas.morton_distance import _morton_distance
from dask_geopandas.geohash import _geohash
from dask_geopandas.core import PANDAS_2_0_0


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


def test_points_from_xy_with_crs():
    latitude = [40.2, 66.3]
    longitude = [-105.1, -38.2]
    expected = geopandas.GeoSeries(
        geopandas.points_from_xy(longitude, latitude, crs="EPSG:4326")
    )
    df = pd.DataFrame({"longitude": longitude, "latitude": latitude})
    ddf = dd.from_pandas(df, npartitions=2)
    actual = dask_geopandas.points_from_xy(
        ddf, "longitude", "latitude", crs="EPSG:4326"
    )
    assert isinstance(actual, dask_geopandas.GeoSeries)
    assert_geoseries_equal(actual.compute(), expected)


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
        "covers",
        "covered_by",
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
    dask_obj = dask_obj.set_geometry(dask_geopandas.points_from_xy(dask_obj, "x", "y"))
    expected = df.set_geometry(geopandas.points_from_xy(df["x"], df["y"]))
    assert_geoseries_equal(dask_obj.geometry.compute(), expected.geometry)


def test_rename_geometry(geodf_points):
    df = geodf_points
    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)
    renamed = dask_obj.rename_geometry("points")
    assert renamed._meta.geometry.name == "points"

    for part in renamed.partitions:
        assert part.compute().geometry.name == "points"

    result = renamed.compute()
    assert_geodataframe_equal(result, df.rename_geometry("points"))


def test_rename_geometry_error(geodf_points):
    df = geodf_points
    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)
    with pytest.raises(ValueError, match="Column named value1 already exists"):
        dask_obj.rename_geometry("value1")


def test_from_dask_dataframe_with_dask_geoseries():
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [1, 2, 3, 4]})
    dask_obj = dd.from_pandas(df, npartitions=2)
    dask_obj = dask_geopandas.from_dask_dataframe(
        dask_obj, geometry=dask_geopandas.points_from_xy(dask_obj, "x", "y")
    )
    # Check that the geometry isn't concatenated and embedded a second time in
    # the high-level graph. cf. https://github.com/geopandas/dask-geopandas/issues/197
    k = next(k for k in dask_obj.dask.dependencies if k.startswith("GeoDataFrame"))
    deps = dask_obj.dask.dependencies[k]
    assert len(deps) == 1

    expected = df.set_geometry(geopandas.points_from_xy(df["x"], df["y"]))
    assert_geoseries_equal(dask_obj.geometry.compute(), expected.geometry)


def test_from_dask_dataframe_with_column_name():
    df = pd.DataFrame({"x": [0, 1, 2, 3], "y": [1, 2, 3, 4]})
    df["geoms"] = geopandas.points_from_xy(df["x"], df["y"])
    dask_obj = dd.from_pandas(df, npartitions=2)
    dask_obj = dask_geopandas.from_dask_dataframe(dask_obj, geometry="geoms")
    expected = geopandas.GeoDataFrame(df, geometry="geoms")
    assert_geodataframe_equal(dask_obj.compute(), expected)


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


def test_spatial_partitions_setter(geodf_points):
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)

    # needs to be a GeoSeries
    with pytest.raises(TypeError):
        dask_obj.spatial_partitions = geodf_points

    # wrong length
    with pytest.raises(ValueError):
        dask_obj.spatial_partitions = geodf_points.geometry


@pytest.mark.parametrize("calculate_spatial_partitions", [True, False])
def test_spatial_partitions_pickle(geodf_points, calculate_spatial_partitions):
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    if calculate_spatial_partitions:
        dask_obj.calculate_spatial_partitions()

    dask_obj2 = pickle.loads(pickle.dumps(dask_obj))
    assert hasattr(dask_obj2, "spatial_partitions")

    dask_series = pickle.loads(pickle.dumps(dask_obj.geometry))
    assert hasattr(dask_series, "spatial_partitions")


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


def test_persist_spatial_partitions(geodf_points):
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    dask_obj.calculate_spatial_partitions()
    dask_obj_persisted = dask_obj.persist()
    pd.testing.assert_series_equal(
        dask_obj.spatial_partitions, dask_obj_persisted.spatial_partitions
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
    Version(geopandas.__version__) <= Version("0.8.1"),
    reason="geopandas 0.8 has bug in apply",
)
def test_geoseries_apply(geoseries_polygons):
    # https://github.com/jsignell/dask-geopandas/issues/18
    ds = dask_geopandas.from_geopandas(geoseries_polygons, npartitions=2)
    result = ds.apply(lambda geom: geom.area, meta=pd.Series(dtype=float)).compute()
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


@pytest.mark.parametrize(
    "shuffle_method",
    [
        "disk",
        "tasks",
    ],
)
def test_set_index_preserves_class(geodf_points, shuffle_method):
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    dask_obj = dask_obj.set_index("value1", shuffle=shuffle_method)

    for partition in dask_obj.partitions:
        assert isinstance(partition.compute(), geopandas.GeoDataFrame)

    assert isinstance(dask_obj.compute(), geopandas.GeoDataFrame)


@pytest.mark.parametrize(
    "shuffle_method",
    [
        "disk",
        "tasks",
    ],
)
def test_set_index_preserves_class_and_name(geodf_points, shuffle_method):
    df = geodf_points.rename_geometry("geom")
    dask_obj = dask_geopandas.from_geopandas(df, npartitions=2)
    dask_obj = dask_obj.set_index("value1", shuffle=shuffle_method)

    for partition in dask_obj.partitions:
        part = partition.compute()
        assert isinstance(part, geopandas.GeoDataFrame)
        assert part.geometry.name == "geom"

    computed = dask_obj.compute()
    assert isinstance(computed, geopandas.GeoDataFrame)
    assert computed.geometry.name == "geom"


def test_copy_none_spatial_partitions(geoseries_points):
    ddf = dask_geopandas.from_geopandas(geoseries_points, npartitions=2)
    ddf.spatial_partitions = None
    ddf_copy = ddf.copy()
    assert ddf_copy.spatial_partitions is None


def test_sjoin():
    # test only the method, functionality tested in test_sjoin.py
    df_points = geopandas.read_file(geopandas.datasets.get_path("naturalearth_cities"))
    ddf_points = dask_geopandas.from_geopandas(df_points, npartitions=4)

    df_polygons = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres")
    )
    expected = df_points.sjoin(df_polygons, predicate="within", how="inner")
    expected = expected.sort_index()

    result = ddf_points.sjoin(df_polygons, predicate="within", how="inner")
    assert_geodataframe_equal(expected, result.compute().sort_index())


def test_clip(geodf_points):
    # test only the method, functionality tested in test_clip.py
    dask_obj = dask_geopandas.from_geopandas(geodf_points, npartitions=2)
    dask_obj.calculate_spatial_partitions()
    mask = geodf_points.iloc[:1]
    mask["geometry"] = mask["geometry"].buffer(2)
    expected = geodf_points.clip(mask)
    result = dask_obj.clip(mask).compute()
    assert_geodataframe_equal(expected, result)

    expected = geodf_points.geometry.clip(mask)
    result = dask_obj.geometry.clip(mask).compute()
    assert_geoseries_equal(expected, result)


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
        if not PANDAS_2_0_0:
            drop = ["name", "iso_a3"]
        else:
            drop = []
        assert_geodataframe_equal(
            gpd_sum, dd_sum.drop(columns=drop), check_like=True
        )

    @pytest.mark.skipif(
        Version(dask.__version__) == Version("2022.01.1"),
        reason="Regression in dask 2022.01.1 https://github.com/dask/dask/issues/8611",
    )
    def test_split_out(self):
        gpd_default = self.world.dissolve("continent")
        dd_split = self.ddf.dissolve("continent", split_out=4)
        assert dd_split.npartitions == 4
        assert_geodataframe_equal(gpd_default, dd_split.compute(), check_like=True)

    @pytest.mark.skipif(
        Version(dask.__version__) == Version("2022.01.1"),
        reason="Regression in dask 2022.01.1 https://github.com/dask/dask/issues/8611",
    )
    @pytest.mark.xfail
    def test_split_out_name(self):
        gpd_default = self.world.rename_geometry("geom").dissolve("continent")
        ddf = dask_geopandas.from_geopandas(
            self.world.rename_geometry("geom"), npartitions=4
        )
        dd_split = ddf.dissolve("continent", split_out=4)
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


class TestSpatialShuffle:
    def setup_method(self):
        self.world = geopandas.read_file(
            geopandas.datasets.get_path("naturalearth_lowres")
        )
        self.ddf = dask_geopandas.from_geopandas(self.world, npartitions=4)

    def test_default(self):
        expected = self.world.set_index(
            _hilbert_distance(self.world, self.world.total_bounds, level=16),
        ).sort_index()

        ddf = self.ddf.spatial_shuffle()
        assert ddf.npartitions == self.ddf.npartitions
        assert isinstance(ddf.spatial_partitions, geopandas.GeoSeries)

        assert_geodataframe_equal(ddf.compute(), expected)

    @pytest.mark.parametrize(
        "p,calculate_partitions,npartitions",
        [
            (10, True, 8),
            (None, False, None),
        ],
    )
    def test_hilbert(self, p, calculate_partitions, npartitions):
        exp_p = p if p else 16
        expected = self.world.set_index(
            _hilbert_distance(self.world, self.world.total_bounds, level=exp_p),
        ).sort_index()

        ddf = self.ddf.spatial_shuffle(
            level=p,
            calculate_partitions=calculate_partitions,
            npartitions=npartitions,
        )

        assert ddf.npartitions == npartitions if npartitions else self.ddf.partitions
        if calculate_partitions:
            assert isinstance(ddf.spatial_partitions, geopandas.GeoSeries)
        else:
            assert ddf.spatial_partitions is None

        assert_geodataframe_equal(ddf.compute(), expected)

    @pytest.mark.parametrize(
        "p,calculate_partitions,npartitions",
        [
            (10, True, 8),
            (None, False, None),
        ],
    )
    def test_morton(self, p, calculate_partitions, npartitions):
        exp_p = p if p else 16
        expected = self.world.set_index(
            _morton_distance(self.world, self.world.total_bounds, level=exp_p),
        ).sort_index()

        ddf = self.ddf.spatial_shuffle(
            "morton",
            level=p,
            calculate_partitions=calculate_partitions,
            npartitions=npartitions,
        )

        assert ddf.npartitions == npartitions if npartitions else self.ddf.partitions
        if calculate_partitions:
            assert isinstance(ddf.spatial_partitions, geopandas.GeoSeries)
        else:
            assert ddf.spatial_partitions is None

        assert_geodataframe_equal(ddf.compute(), expected)

    @pytest.mark.skipif(
        Version(dask.__version__) <= Version("2021.03.0"),
        reason="older Dask has a bug in sorting",
    )
    @pytest.mark.parametrize(
        "calculate_partitions,npartitions",
        [
            (True, 8),
            (False, None),
        ],
    )
    def test_geohash(self, calculate_partitions, npartitions):
        df = self.world.copy()
        # crossing meridian and resulting 0 causes inconsistencies among environments
        df = df[df.name != "Fiji"]
        expected = df.set_index(
            _geohash(df, as_string=False, precision=12),
        ).sort_index()

        ddf = dask_geopandas.from_geopandas(df, npartitions=4)

        ddf = ddf.spatial_shuffle(
            "geohash",
            calculate_partitions=calculate_partitions,
            npartitions=npartitions,
        )

        assert ddf.npartitions == npartitions if npartitions else ddf.partitions
        if calculate_partitions:
            assert isinstance(ddf.spatial_partitions, geopandas.GeoSeries)
        else:
            assert ddf.spatial_partitions is None

        assert_geodataframe_equal(ddf.compute(), expected)


def test_to_wkt(geodf_points_crs):
    df = geodf_points_crs
    df["polygons"] = df.buffer(1)
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    expected = df.to_wkt()
    result = ddf.to_wkt().compute()

    assert_frame_equal(expected, result)


def test_to_wkt_series(geoseries_points):
    s = geoseries_points
    dask_obj = dask_geopandas.from_geopandas(s, npartitions=4)
    expected = s.to_wkt()
    result = dask_obj.to_wkt().compute()

    assert_series_equal(expected, result)


@pytest.mark.parametrize("hex", [True, False])
def test_to_wkb(geodf_points_crs, hex):
    df = geodf_points_crs
    df["polygons"] = df.buffer(1)
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    expected = df.to_wkb(hex=hex)
    result = ddf.to_wkb(hex=hex).compute()

    assert_frame_equal(expected, result)


@pytest.mark.parametrize("hex", [True, False])
def test_to_wkb_series(geoseries_points, hex):
    s = geoseries_points
    dask_obj = dask_geopandas.from_geopandas(s, npartitions=4)
    expected = s.to_wkb(hex=hex)
    result = dask_obj.to_wkb(hex=hex).compute()

    assert_series_equal(expected, result)


@pytest.mark.parametrize("coord", ["x", "y", "z"])
def test_get_coord(coord):
    p1 = Point(1, 2, 3)
    p2 = Point(2, 3, 4)
    p3 = Point(3, 4, 5)
    p4 = Point(4, 1, 7)
    s = geopandas.GeoSeries([p1, p2, p3, p4])
    dask_obj = dask_geopandas.from_geopandas(s, npartitions=2)
    expected = getattr(s, coord)
    result = getattr(dask_obj, coord).compute()
    assert_series_equal(expected, result)


def test_to_dask_dataframe(geodf_points_crs):
    df = geodf_points_crs
    dask_gpd = dask_geopandas.from_geopandas(df, npartitions=2)
    dask_df = dask_gpd.to_dask_dataframe()

    assert isinstance(dask_df, dd.DataFrame) and not isinstance(
        dask_df, dask_geopandas.GeoDataFrame
    )
    expected = pd.DataFrame(df)
    result = dask_df.compute()
    assert_frame_equal(result, expected)
    assert isinstance(result, pd.DataFrame) and not isinstance(
        result, geopandas.GeoDataFrame
    )


def test_compute_empty_partitions():
    # https://github.com/geopandas/dask-geopandas/issues/190 - ensure to skip
    # empty partitions when concatting the computed results

    @dask.delayed
    def get_chunk(n):
        return geopandas.GeoDataFrame({"col": [1] * n, "geometry": [Point(1, 1)] * n})

    meta = geopandas.GeoDataFrame({"col": [1], "geometry": [Point(1, 1)]})

    ddf = dd.concat([dd.from_delayed(get_chunk(n), meta=meta) for n in [0, 2]])

    expected = geopandas.GeoDataFrame({"col": [1, 1], "geometry": [Point(1, 1)] * 2})
    assert_geodataframe_equal(ddf.compute(), expected)
