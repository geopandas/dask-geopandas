import os.path
from packaging.version import Version

import dask

import geopandas

import pytest

# TODO update version once geopandas has a proper tag for 1.0
GEOPANDAS_GE_10 = (Version(geopandas.__version__) >= Version("0.14.0+70")) and (
    Version(geopandas.__version__) < Version("0.14.1")
)


# TODO Disable usage of pyarrow strings until the expected results in the tests
# are updated to use those as well
dask.config.set({"dataframe.convert-string": False})


# Datasets used in our tests

_HERE = os.path.abspath(os.path.dirname(__file__))
_TEST_DATA_DIR = os.path.join(_HERE, "data")
_NATURALEARTH_CITIES = os.path.join(
    _TEST_DATA_DIR, "naturalearth_cities", "naturalearth_cities.shp"
)
_NATURALEARTH_LOWRES = os.path.join(
    _TEST_DATA_DIR, "naturalearth_lowres", "naturalearth_lowres.shp"
)


@pytest.fixture(scope="session")
def naturalearth_lowres() -> str:
    # skip if data missing, unless on github actions
    if os.path.isfile(_NATURALEARTH_LOWRES) or os.getenv("GITHUB_ACTIONS"):
        return _NATURALEARTH_LOWRES
    else:
        pytest.skip("Naturalearth lowres dataset not found")


@pytest.fixture(scope="session")
def naturalearth_cities() -> str:
    # skip if data missing, unless on github actions
    if os.path.isfile(_NATURALEARTH_CITIES) or os.getenv("GITHUB_ACTIONS"):
        return _NATURALEARTH_CITIES
    else:
        pytest.skip("Naturalearth cities dataset not found")
