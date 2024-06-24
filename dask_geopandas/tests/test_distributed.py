from packaging.version import Version

import geopandas

import dask_geopandas

import pytest
from geopandas.testing import assert_geodataframe_equal

distributed = pytest.importorskip("distributed")


from distributed import Client, LocalCluster


@pytest.mark.skipif(
    Version(distributed.__version__) < Version("2024.6.0"),
    reason="distributed < 2024.6 has a wrong assertion",
    # https://github.com/dask/distributed/pull/8667
)
@pytest.mark.skipif(
    Version(distributed.__version__) < Version("0.13"),
    reason="geopandas < 0.13 does not implement sorting geometries",
)
def test_spatial_shuffle(naturalearth_cities):
    df_points = geopandas.read_file(naturalearth_cities)

    with LocalCluster(n_workers=1) as cluster:
        with Client(cluster):
            ddf_points = dask_geopandas.from_geopandas(df_points, npartitions=4)

            ddf_result = ddf_points.spatial_shuffle(
                by="hilbert", calculate_partitions=False
            )
            result = ddf_result.compute()

    expected = df_points.sort_values("geometry").reset_index(drop=True)
    assert_geodataframe_equal(result.reset_index(drop=True), expected)
