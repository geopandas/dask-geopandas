# Reading and Writing Apache Parquet

Similar to dask-dataframe, dask-geopandas supports reading and writing Apache Parquet files.

See the [Dask DataFrame](https://docs.dask.org/en/stable/dataframe-parquet.html#dataframe-parquet) 
and [Geopandas](https://geopandas.org/en/stable/docs/user_guide/io.html#apache-parquet-and-feather-file-formats) documentation
for more on Apache Parquet.

## Partitioning

As outlined in [guide/spatial-partitioning](guide/spatial-partitioning.md), dask-geopandas can spatially partition datasets. These partitions are
persisted in the parquet files.

By default, reading these spatial partitions requires opening every file and checking its spatial extent. This can be a
bit slow if the parquet dataset is made up of many individual partitions. To disable loading the spatial partitions,
specify ``gather_spatial_partitions=False`` when reading the file:


```py
ddf = dask_geopandas.read_parquet("...", gather_spatial_partitions=False)
ddf.spatial_partitions  # None
```
