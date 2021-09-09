# Getting started

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

Currently, this repartitions the data naively by rows. In the future, this will
also provide spatial partitioning to take advantage of the spatial structure of
the GeoDataFrame (but the current version still provides basic multi-core
parallelism).

The familiar spatial attributes and methods of GeoPandas are also available
and will be computed in parallel:

```py
ddf.geometry.area.compute()
ddf.within(polygon)
```

Additionally, if you have a distributed dask.dataframe you can pass columns of
x-y points to the ``set_geometry`` method. Currently, this only supports point
data.

```py
import dask.dataframe as dd
import dask_geopandas

ddf = dd.read_csv('...')

ddf = dask_geopandas.from_dask_dataframe(ddf)
ddf = ddf.set_geometry(
    dask_geopandas.points_from_xy(ddf, 'latitude', 'longitude')
)
```

Writing files (and reading back) is currently supported for the Parquet file
format:

```py
ddf.to_parquet("path/to/dir/")
ddf = dask_geopandas.read_parquet("path/to/dir/")
```