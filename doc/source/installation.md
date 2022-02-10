# Installation

This package depends on GeoPandas, PyGEOS, and Dask.

One way to install all required dependencies is to use the ``conda`` package manager to
create a new environment:

```shell
conda create -n geo_env
conda activate geo_env
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install dask-geopandas
```
