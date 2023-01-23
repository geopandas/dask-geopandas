# Installation

This package depends on GeoPandas and Dask. In addition, it also requires
either Shapely >= 2.0, or the PyGEOS package.

GeoPandas is written in pure Python, but has several dependencies written in C (GEOS, GDAL, PROJ). Those base C libraries can sometimes be a challenge to install. Therefore, we advise you to closely follow the [recommendations](https://geopandas.org/en/stable/getting_started/install.html) to avoid installation problems.

## Easy way

The best way to install Dask-GeoPandas is using `conda` or `mamba` and `conda-forge` channel:

```sh
conda install -c conda-forge dask-geopandas
```

## pip

You can install Dask-GeoPandas with `pip` from PyPI but make sure that your environment contains
properly installed GeoPandas (note that Dask-GeoPandas does not use `fiona` which therefore doesn't
have to be installed). See the [GeoPandas installation instructions](https://geopandas.org/en/stable/getting_started/install.html#installing-with-pip) for details.

```sh
pip install dask-geopandas
```

## Fresh environment

One way to install all required dependencies is to use the `conda` package manager to
create a new environment:

```shell
conda create -n geo_env
conda activate geo_env
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install dask-geopandas
```
