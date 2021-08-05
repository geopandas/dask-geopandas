import geopandas
import dask_geopandas
import dask.dataframe as dd

import pytest
from geopandas.testing import assert_geodataframe_equal


pa = pytest.importorskip("pyarrow")


pytestmark = pytest.mark.filterwarnings(
    "ignore:this is an initial implementation:UserWarning"
)


def test_roundtrip(tmp_path):
    # basic roundtrip
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)

    basedir = tmp_path / "dataset"
    # TODO awaiting a `to_feather` implementation
    # ddf.to_feather(basedir)
    basedir.mkdir()
    for i, part in enumerate(ddf.partitions):
        part.compute().to_feather(basedir / f"part.{i}.feather")

    # each partition (4) is written as a feather file
    paths = list(basedir.glob("*.feather"))
    assert len(paths) == 4

    # reading back gives identical GeoDataFrame
    result = dask_geopandas.read_feather(basedir)
    assert result.npartitions == 4
    # TODO this reset_index should not be necessary
    result_gpd = result.compute().reset_index(drop=True)
    assert_geodataframe_equal(result_gpd, df)
    # TODO
    # # reading back also populates the spatial partitioning property
    # assert result.spatial_partitions is not None

    # TODO geopandas doesn't actually support this for "feather" format
    # # the written dataset is also readable by plain geopandas
    # result_gpd = geopandas.read_feather(basedir)
    # # the dataset written by dask has "__null_dask_index__" index column name
    # result_gpd.index.name = None
    # assert_geodataframe_equal(result_gpd, df)

    result_part0 = geopandas.read_feather(basedir / "part.0.feather")
    result_part0.index.name = None
    assert_geodataframe_equal(result_part0, df.iloc[:45])


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
    # TODO
    # # and also preserves the spatial partitioning information
    # assert ddf_subset.spatial_partitions is not None

    # selecting a single non-geometry column on the dataframe should work
    s = ddf["pop_est"]
    assert type(s) is dd.Series
    assert s.max().compute() == df["pop_est"].max()
