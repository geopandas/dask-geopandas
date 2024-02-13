import os.path
from packaging.version import Version
import warnings

import pytest

import geopandas
import dask


# TODO update version once geopandas has a proper tag for 1.0
GEOPANDAS_GE_10 = (Version(geopandas.__version__) >= Version("0.14.0+70")) and (
    Version(geopandas.__version__) < Version("0.14.1")
)


# TODO Disable usage of pyarrow strings until the expected results in the tests
# are updated to use those as well
dask.config.set({"dataframe.convert-string": False})


# Datasets used in our tests


if GEOPANDAS_GE_10:
    package_dir = os.path.abspath(geopandas.__path__[0])
    test_data_dir = os.path.join(package_dir, "tests", "data")
    _NATURALEARTH_CITIES = os.path.join(
        test_data_dir, "naturalearth_cities", "naturalearth_cities.shp"
    )
    _NATURALEARTH_LOWRES = os.path.join(
        test_data_dir, "naturalearth_lowres", "naturalearth_lowres.shp"
    )
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _NATURALEARTH_CITIES = geopandas.datasets.get_path("naturalearth_cities")
        _NATURALEARTH_LOWRES = geopandas.datasets.get_path("naturalearth_lowres")


@pytest.fixture(scope="session")
def naturalearth_lowres() -> str:
    # skip if data missing, unless on github actions
    return _NATURALEARTH_LOWRES


@pytest.fixture(scope="session")
def naturalearth_cities() -> str:
    return _NATURALEARTH_CITIES
