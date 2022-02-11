import geopandas
import dask_geopandas
import dask.dataframe as dd

import pytest
from geopandas.testing import assert_geodataframe_equal, assert_geoseries_equal


pa = pytest.importorskip("pyarrow")
ds = pytest.importorskip("pyarrow.dataset")


pytestmark = pytest.mark.filterwarnings(
    "ignore:this is an initial implementation:UserWarning"
)


def test_read(tmp_path):
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

    # writing a partitioned dataset with geopandas (to not rely on roundtrip)
    basedir = tmp_path / "dataset"
    basedir.mkdir()
    df.iloc[:100].to_feather(basedir / "data.0.feather")
    df.iloc[100:].to_feather(basedir / "data.1.feather")

    result = dask_geopandas.read_feather(basedir)
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.npartitions == 2
    assert result.crs == df.crs
    assert result.spatial_partitions is not None
    # TODO this reset_index should not be necessary
    result_gpd = result.compute().reset_index(drop=True)
    assert_geodataframe_equal(result_gpd, df)


def test_write(tmp_path):
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)

    basedir = tmp_path / "dataset"
    ddf.to_feather(basedir)

    # each partition (4) is written as a feather file
    paths = list(basedir.glob("*.feather"))
    assert len(paths) == 4

    # each individual file is a valid feather file
    result_part0 = geopandas.read_feather(basedir / "part.0.feather")
    result_part0.index.name = None
    assert_geodataframe_equal(result_part0, df.iloc[:45])

    # TODO geopandas doesn't actually support this for "feather" format
    # # the written dataset is also readable by plain geopandas
    # result_gpd = geopandas.read_feather(basedir)
    # # the dataset written by dask has "__null_dask_index__" index column name
    # result_gpd.index.name = None
    # assert_geodataframe_equal(result_gpd, df)


@pytest.mark.xfail  # https://github.com/dask/dask/issues/8022
def test_write_delayed(tmp_path):
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)

    basedir = tmp_path / "dataset"
    dataset = ddf.to_feather(basedir, compute=False)
    dataset.compute()
    result = dask_geopandas.read_feather(basedir)
    assert result.npartitions == 4
    # TODO this reset_index should not be necessary
    result_gpd = result.compute().reset_index(drop=True)
    assert_geodataframe_equal(result_gpd, df)


def test_roundtrip(tmp_path):
    # basic roundtrip
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)

    basedir = tmp_path / "dataset"
    ddf.to_feather(basedir)

    # reading back gives identical GeoDataFrame
    result = dask_geopandas.read_feather(basedir)
    assert result.npartitions == 4
    assert result.crs == df.crs
    # TODO this reset_index should not be necessary
    result_gpd = result.compute().reset_index(drop=True)
    assert_geodataframe_equal(result_gpd, df)
    # reading back also populates the spatial partitioning property
    ddf.calculate_spatial_partitions()
    assert_geoseries_equal(result.spatial_partitions, ddf.spatial_partitions.envelope)


def test_column_selection_push_down(tmp_path):
    # set up dataset
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    basedir = tmp_path / "dataset"
    # TODO awaiting a `to_feather` implementation
    # ddf.to_feather(basedir)
    basedir.mkdir()
    for i, part in enumerate(ddf.partitions):
        part.compute().to_feather(basedir / f"part.{i}.feather")

    ddf = dask_geopandas.read_feather(basedir)

    # selecting columns including geometry column still gives GeoDataFrame
    ddf_subset = ddf[["pop_est", "geometry"]]
    assert type(ddf_subset) is dask_geopandas.GeoDataFrame
    # and also preserves the spatial partitioning information
    assert ddf_subset.spatial_partitions is not None

    # selecting a single non-geometry column on the dataframe should work
    s = ddf["pop_est"]
    assert type(s) is dd.Series
    assert s.max().compute() == df["pop_est"].max()


def test_missing_metadata(tmp_path):
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    path = tmp_path / "test.feather"

    # convert to DataFrame with wkb -> writing to feather will have only pandas metadata
    df = df.to_wkb()
    df.to_feather(path)

    with pytest.raises(ValueError, match="Missing geo metadata"):
        dask_geopandas.read_feather(path)

    # remove metadata completely
    from pyarrow import feather

    table = feather.read_table(path)
    feather.write_feather(table.replace_schema_metadata(), path)

    with pytest.raises(ValueError, match="Missing geo metadata"):
        dask_geopandas.read_feather(path)


@pytest.mark.parametrize(
    "filter", [[("continent", "=", "Africa")], ds.field("continent") == "Africa"]
)
def test_filters(tmp_path, filter):
    # set up dataset
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    basedir = tmp_path / "dataset"
    ddf.to_feather(basedir)

    # specifying filters argument
    result = dask_geopandas.read_feather(basedir, filters=filter)
    assert result.npartitions == 4

    result_gpd = result.compute().reset_index(drop=True)
    expected = df[df["continent"] == "Africa"].reset_index(drop=True)
    assert_geodataframe_equal(result_gpd, expected)
