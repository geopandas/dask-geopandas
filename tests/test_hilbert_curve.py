import pytest
import pandas as pd
from hilbertcurve.hilbertcurve import HilbertCurve
from dask_geopandas.hilbert_distance import _continuous_to_discrete_coords
from dask_geopandas import from_geopandas
from .test_core import geoseries_points, geoseries_lines, geoseries_polygons


@pytest.fixture(params=[geoseries_points, geoseries_lines, geoseries_polygons])
def geom_types(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def test_hilbert_distance_dask(geom_types):

    bounds = geom_types.bounds.to_numpy()
    total_bounds = geom_types.total_bounds
    coords = _continuous_to_discrete_coords(total_bounds, bounds, p=15)

    hilbert_curve = HilbertCurve(p=15, n=2)
    expected = hilbert_curve.distances_from_points(coords)

    ddf = from_geopandas(geom_types, npartitions=1)
    result = ddf.hilbert_distance().compute()

    assert list(result) == expected
    assert isinstance(result, pd.Series)
