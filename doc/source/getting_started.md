# Getting started

The relationship between Dask-GeoPandas and GeoPandas is the same as the relationship
between `dask.dataframe` and `pandas`.  We recommend checking the
[Dask documentation](https://docs.dask.org/en/latest/dataframe.html) to better understand how
DataFrames are scaled before diving into Dask-GeoPandas.

## Dask-GeoPandas basics

Given a GeoPandas dataframe

```py
import geopandas
df = geopandas.read_file('...')
```

We can repartition it into a Dask-GeoPandas dataframe:

```py
import dask_geopandas
ddf = dask_geopandas.from_geopandas(df, npartitions=4)
```

By default, this repartitions the data naively by rows. However, you can
also provide spatial partitioning to take advantage of the spatial structure of
the GeoDataFrame.

```py
ddf = ddf.spatial_shuffle()
```

The familiar spatial attributes and methods of GeoPandas are also available
and will be computed in parallel:

```py
ddf.geometry.area.compute()
ddf.within(polygon)
```

Additionally, if you have a distributed dask.dataframe you can pass columns of
x-y points to the ``set_geometry`` method.

```py
import dask.dataframe as dd
import dask_geopandas

ddf = dd.read_csv('...')

ddf = ddf.set_geometry(
    dask_geopandas.points_from_xy(ddf, 'longitude', 'latitude')
)
```

Writing files (and reading back) is currently supported for the Parquet and Feather file
formats.

```py
ddf.to_parquet("path/to/dir/")
ddf = dask_geopandas.read_parquet("path/to/dir/")
```

Traditional GIS file formats can be read into partitioned GeoDataFrame
(requires `pyogrio`) but not written.

```py
ddf = dask_geopandas.read_file("file.gpkg", npartitions=4)
```
