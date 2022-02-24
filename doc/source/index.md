# dask-geopandas documentation

Parallel GeoPandas with Dask

Dask-GeoPandas is a project merging the geospatial capabilities of [GeoPandas](https://geopandas.org)
and scalability of [Dask](https://dask.org). GeoPandas is an open source project designed to make working with geospatial data in Python easier. GeoPandas extends the datatypes used by pandas to allow spatial operations on geometric types.
Dask provides advanced parallelism and distributed out-of-core computation with a dask.dataframe module designed to scale
pandas. Since GeoPandas is an extension to the pandas DataFrame, the same way Dask scales pandas can also be applied to GeoPandas.

This project is a bridge between Dask and GeoPandas and offers geospatial capabilities of GeoPandas backed by Dask.

## Install

Dask-GeoPandas depends on Dask, GeoPandas and PyGEOS. We recommend installing via `conda` or `mamba` from
the `conda-forge` channel but you can also install it from PyPI.

```sh
conda install dask-geopandas -c conda-forge
```

```sh
pip install dask-geopandas
```

## Example

As with `dask.dataframe` and `pandas`, the API of `dask_geopandas` mirrors the one of `geopandas`.

```py
import geopandas
import dask_geopandas

df = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
dask_df = dask_geopandas.from_geopandas(df, npartitions=4)

ddf.geometry.area.compute()
```

## When should I use Dask-GeoPandas?

Dask-GeoPandas is useful when dealing with large GeoDataFrames that either do not comfortably fit in memory or require expensive computation that can be easily parallelised. Note that using Dask-GeoPandas is not always faster than using GeoPandas as there is an unavoidable overhead in task scheduling and transfer of data between threads and processes, but in other cases, your performance gains can be almost linear with more threads.

## Useful links

[Source Repository (GitHub)](https://github.com/geopandas/dask-geopandas) | [Issues & Ideas](https://github.com/geopandas/dask-geopandas/issues) | [Gitter (chat)](https://gitter.im/geopandas/dask-geopandas)

```{toctree}
---
maxdepth: 2
caption: Documentation
hidden: true
---
installation
getting_started
guide
api
GitHub <https://github.com/geopandas/dask-geopandas>
```
