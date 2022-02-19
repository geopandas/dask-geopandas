import geopandas
import dask_geopandas
import dask.dataframe as dd

import pytest
from geopandas.testing import assert_geodataframe_equal
from pandas.testing import assert_series_equal


pa = pytest.importorskip("pyarrow")


pytestmark = pytest.mark.filterwarnings(
    "ignore:this is an initial implementation:UserWarning"
)


def test_parquet_roundtrip(tmp_path):
    # basic roundtrip
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)

    basedir = tmp_path / "dataset"
    ddf.to_parquet(basedir)

    # each partition (4) is written as parquet file
    paths = list(basedir.glob("*.parquet"))
    assert len(paths) == 4

    # reading back gives identical GeoDataFrame
    result = dask_geopandas.read_parquet(basedir)
    assert result.npartitions == 4
    assert_geodataframe_equal(result.compute(), df)
    # reading back correctly sets the CRS in meta
    assert result.crs == df.crs
    # reading back also populates the spatial partitioning property
    assert result.spatial_partitions is not None
    assert result.spatial_partitions.crs == df.crs

    # the written dataset is also readable by plain geopandas
    result_gpd = geopandas.read_parquet(basedir)
    # the dataset written by dask has "__null_dask_index__" index column name
    result_gpd.index.name = None
    assert_geodataframe_equal(result_gpd, df)

    result_part0 = geopandas.read_parquet(basedir / "part.0.parquet")
    result_part0.index.name = None
    assert_geodataframe_equal(result_part0, df.iloc[:45])


def test_roundtrip_geometry_column_name(tmp_path):
    # basic roundtrip with different geometry column name
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    df = df.rename_geometry("geom")

    # geopandas -> dask-geopandas roundtrip
    path = tmp_path / "data.parquet"
    df.to_parquet(path)
    result = dask_geopandas.read_parquet(path)
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.geometry.name == "geom"
    assert result.crs == df.crs
    assert result.spatial_partitions is not None
    assert_geodataframe_equal(result.compute(), df)

    # dask-geopandas -> dask-geopandas roundtrip
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    assert ddf.geometry.name == "geom"
    basedir = tmp_path / "dataset"
    ddf.to_parquet(basedir)

    result = dask_geopandas.read_parquet(basedir)
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.geometry.name == "geom"
    assert result.crs == df.crs
    assert result.spatial_partitions is not None
    assert_geodataframe_equal(result.compute(), df)


def test_roundtrip_multiple_geometry_columns(tmp_path):
    # basic roundtrip with different geometry column name
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    df["geometry2"] = df.geometry.representative_point().to_crs("EPSG:3857")
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)

    basedir = tmp_path / "dataset"
    ddf.to_parquet(basedir)

    result = dask_geopandas.read_parquet(basedir)
    assert isinstance(result, dask_geopandas.GeoDataFrame)
    assert result.crs == df.crs
    assert result.spatial_partitions is not None
    assert_geodataframe_equal(result.compute(), df)

    # ensure the geometry2 column is also considered as geometry in meta
    assert_series_equal(result.dtypes, df.dtypes)
    assert isinstance(result["geometry2"], dask_geopandas.GeoSeries)
    assert result["geometry"].crs == "EPSG:4326"
    assert result["geometry2"].crs == "EPSG:3857"


def test_column_selection_push_down(tmp_path):
    # set up dataset
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    basedir = tmp_path / "dataset"
    ddf.to_parquet(basedir)

    ddf = dask_geopandas.read_parquet(basedir)

    # selecting columns including geometry column still gives GeoDataFrame
    ddf_subset = ddf[["pop_est", "geometry"]]
    assert type(ddf_subset) is dask_geopandas.GeoDataFrame
    # and also preserves the spatial partitioning information
    assert ddf_subset.spatial_partitions is not None

    # selecting a single non-geometry column on the dataframe should work
    s = ddf["pop_est"]
    assert type(s) is dd.Series
    assert s.max().compute() == df["pop_est"].max()


def test_parquet_roundtrip_s3(s3_resource, s3_storage_options):
    fs, endpoint_url = s3_resource

    # basic roundtrip
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)

    uri = "s3://geopandas-test/dataset.parquet"
    ddf.to_parquet(uri, storage_options=s3_storage_options)

    # reading back gives identical GeoDataFrame
    result = dask_geopandas.read_parquet(uri, storage_options=s3_storage_options)
    assert result.npartitions == 4
    assert_geodataframe_equal(result.compute(), df)
    # reading back correctly sets the CRS in meta
    assert result.crs == df.crs
    # reading back also populates the spatial partitioning property
    assert result.spatial_partitions is not None


def test_parquet_empty_partitions(tmp_path):
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    # Creating filtered dask dataframe with at least one empty partition
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    ddf_filtered = ddf[ddf["pop_est"] > 1_000_000_000]
    assert (ddf_filtered.map_partitions(len).compute() == 0).any()

    basedir = tmp_path / "dataset"
    # TODO don't write metadata file as that fails with empty partitions on
    # inferring the schema
    ddf_filtered.to_parquet(basedir, write_metadata_file=False)

    result = dask_geopandas.read_parquet(basedir)
    assert_geodataframe_equal(result.compute(), df[df["pop_est"] > 1_000_000_000])
    # once one partition has no spatial extent, we don't restore the spatial partitions
    assert result.spatial_partitions is None


def test_parquet_partition_on(tmp_path):
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)

    # Writing a partitioned dataset based on one of the attribute columns
    basedir = tmp_path / "naturalearth_lowres_by_continent.parquet"
    ddf.to_parquet(basedir, partition_on="continent")

    # Check for one of the partitions that the file is present and is correct
    assert len(list(basedir.iterdir())) == 10  # 8 continents + 2 metadata files
    assert (basedir / "continent=Africa").exists()
    result_africa = geopandas.read_parquet(basedir / "continent=Africa")
    expected = df[df["continent"] == "Africa"].drop(columns=["continent"])
    result_africa.index.name = None
    assert_geodataframe_equal(result_africa, expected)

    # Check roundtrip
    result = dask_geopandas.read_parquet(basedir)
    assert result.npartitions >= 8
    assert result.spatial_partitions is not None
    expected = df.copy()
    expected["continent"] = expected["continent"].astype("category")
    assert_geodataframe_equal(result.compute(), expected, check_like=True)
