import pytest
import pandas as pd
from hilbertcurve.hilbertcurve import HilbertCurve
from dask_geopandas.hilbert_distance import _continuous_to_discrete_coords
from dask_geopandas import from_geopandas, hilbert_distance
from .test_core import geoseries_points, geoseries_lines, geoseries_polygons

@pytest.fixture
def geoseries_points(geoseries_points):
	return geoseries_points


@pytest.fixture
def geoseries_lines(geoseries_lines):
	return geoseries_lines


@pytest.fixture
def geoseries1(geoseries_polygons):
	return geoseries_polygons


@pytest.fixture(params=[geoseries_points, geoseries_lines, geoseries_polygons])
def geom_types(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def test_hilbert_distance_dask(geom_types):
    
    bounds = geoseries.bounds.to_numpy()
    total_bounds = geoseries.total_bounds
    coords = _continuous_to_discrete_coords(total_bounds, bounds, p=15)
    
    hilbert_curve = HilbertCurve(p=15, n=2)
    expected = hilbert_curve.distances_from_points(coords)

    ddf = from_geopandas(geoseries, npartitions=1)
    result = ddf.hilbert_distance().compute()
    
    assert list(result) == expected
    assert isinstance(result, pd.Series)
