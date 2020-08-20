import geopandas
import dask_geopandas

import pytest
from geopandas.testing import assert_geodataframe_equal


pa = pytest.importorskip("pyarrow")


def test_parquet_roundtrip(tmp_path):
    # basic roundtrip 
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    
    basedir = tmp_path / "dataset"
    dask_geopandas.to_parquet(ddf, basedir)

    # each partition (4) is written as parquet file
    paths = list(basedir.glob("*.parquet"))
    assert len(paths) == 4

    # reading back gives identical GeoDataFrame 
    result = dask_geopandas.read_parquet(basedir)
    assert ddf.npartitions == 4
    assert_geodataframe_equal(result.compute(), df)

    # the written dataset is also readable by plain geopandas
    result_gpd = geopandas.read_parquet(basedir)
    # the dataset written by dask has "__null_dask_index__" index column name
    result_gpd.index.name = None
    assert_geodataframe_equal(result_gpd, df)

    result_part0 = geopandas.read_parquet(basedir / "part.0.parquet")
    result_part0.index.name = None
    assert_geodataframe_equal(result_part0, df.iloc[:45])
