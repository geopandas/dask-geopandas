import geopandas
from geopandas.testing import assert_geodataframe_equal

import dask_geopandas


def test_cx():
    # test cx using spatial partitions
    df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    ddf = dask_geopandas.from_geopandas(df, npartitions=4)
    ddf.calculate_spatial_partitions()

    subset = ddf.cx[-180:-70, 0:-80]
    assert len(subset) == 8
    expected = df.cx[-180:-70, 0:-80]
    assert_geodataframe_equal(subset.compute(), expected)

    # empty slice
    subset = ddf.cx[-200:-190, 300:400]
    assert len(subset) == 0
    expected = df.cx[-200:-190, 300:400]
    assert_geodataframe_equal(subset.compute(), expected)
