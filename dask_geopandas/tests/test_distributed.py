import pytest

import geopandas
import dask_geopandas

from geopandas.testing import assert_geodataframe_equal

distributed = pytest.importorskip("distributed")


from distributed import LocalCluster, Client

# from distributed.utils_test import gen_cluster


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


# @gen_cluster(client=True)
# async def test_spatial_shuffle(c, s, a, b, naturalearth_cities):
#     df_points = geopandas.read_file(naturalearth_cities)
#     ddf_points = dask_geopandas.from_geopandas(df_points, npartitions=4)

#     ddf_result = ddf_points.spatial_shuffle(by="hilbert", calculate_partitions=False)
#     result = (await c.compute(ddf_result)).val

#     expected = df_points.sort_values("geometry").reset_index(drop=True)
#     assert_geodataframe_equal(result.reset_index(drop=True), expected)
