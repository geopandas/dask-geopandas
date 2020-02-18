import pytest
from shapely.geometry import (
    Polygon,
)
import geopandas
import dask_geopandas


@pytest.fixture
def geoseries():
    t1 = Polygon([(0, 3.5), (7, 2.4), (1, 0.1)])
    t2 = Polygon([(0, 0), (1, 1), (0, 1)])
    sq = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    return geopandas.GeoSeries([t1, t2, sq] * 10)


@pytest.mark.parametrize("attr", [
    "area",
    "geom_type",
    "is_valid",
    "bounds",
    "total_bounds"
])
def test_geoseries_properties(geoseries, attr):
    original = getattr(geoseries, attr)

    dask_obj = dask_geopandas.from_geopandas(geoseries, npartitions=2)
    assert len(dask_obj.partitions[0]) < len(geoseries)
    assert isinstance(dask_obj, dask_geopandas.GeoSeries)

    daskified = getattr(dask_obj, attr)
    assert all(original == daskified.compute())
