import geopandas
import dask_geopandas

import pytest
from geopandas.testing import assert_geodataframe_equal


pytest.importorskip("pyogrio")


def test_read_file():
    path = geopandas.datasets.get_path("naturalearth_lowres")
    df = geopandas.read_file(path)
    result = dask_geopandas.read_file(path, npartitions=4)
    assert result.npartitions == 4
    assert result.crs == df.crs
    assert_geodataframe_equal(result.compute().reset_index(drop=True), df)

    result = dask_geopandas.read_file(path, chunksize=100)
    assert result.npartitions == 2
    assert result.crs == df.crs
    assert_geodataframe_equal(result.compute().reset_index(drop=True), df)

    msg = "Exactly one of npartitions and chunksize must be specified"
    with pytest.raises(ValueError, match=msg):
        dask_geopandas.read_file(path)
    with pytest.raises(ValueError, match=msg):
        dask_geopandas.read_file(path, npartitions=4, chunksize=100)
