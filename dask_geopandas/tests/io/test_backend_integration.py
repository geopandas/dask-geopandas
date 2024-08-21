import geopandas

import dask_geopandas

import pytest
from geopandas.testing import assert_geodataframe_equal

try:
    import pyogrio  # noqa: F401

    PYOGRIO = True
except ImportError:
    PYOGRIO = False

BACKENDS = ["arrow", "file", "parquet"]


@pytest.fixture(params=BACKENDS)
def backend(request):
    param = request.param
    if not PYOGRIO and param == "file":
        pytest.skip("Unable to import pyogrio for file backend")
    return param


def from_arrow_backend(path, tmp_path, npartitions):
    df = geopandas.read_file(path)
    basedir = tmp_path / "dataset"
    basedir.mkdir()
    ddf = dask_geopandas.from_geopandas(df, npartitions=npartitions)
    for i, part in enumerate(ddf.partitions):
        part.compute().to_feather(basedir / f"data.{i}.feather")
    return dask_geopandas.read_feather(basedir)


def from_file_backend(path, tmp_path, npartitions):
    return dask_geopandas.read_file(path, npartitions=npartitions)


def from_parquet_backend(path, tmp_path, npartitions):
    ddf = dask_geopandas.from_geopandas(
        geopandas.read_file(path), npartitions=npartitions
    )
    basedir = tmp_path / "dataset"
    ddf.to_parquet(basedir)
    return dask_geopandas.read_parquet(basedir)


def get_from_backend(backend, data_path, tmp_path, npartitions=4):
    if backend == "arrow":
        ddf = from_arrow_backend(data_path, tmp_path, npartitions)
    elif backend == "file":
        ddf = from_file_backend(data_path, tmp_path, npartitions)
    elif backend == "parquet":
        ddf = from_parquet_backend(data_path, tmp_path, npartitions)
    else:
        raise ValueError()
    return ddf


def test_spatial_shuffle_integration(backend, naturalearth_lowres, tmp_path):
    ddf = get_from_backend(backend, naturalearth_lowres, tmp_path)
    new_idx = ddf.hilbert_distance()
    expected = ddf.compute().set_index(new_idx.compute())

    result = ddf.spatial_shuffle()
    # Sort because the index is shuffled
    assert_geodataframe_equal(result.compute().sort_index(), expected.sort_index())
