# Installation

This package depends on GeoPandas, PyGEOS, and Dask.

One way is to use the ``conda`` package manager to create a new environment:

```shell
conda create -n geo_env
conda activate geo_env
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict
conda install python=3 geopandas dask pygeos
pip install git+git://github.com/geopandas/dask-geopandas.git
```