dask-geopandas
==============

Parallel GeoPandas with Dask

Status
------

**EXPERIMENTAL** This project is in an early state. The basic element-wise
spatial methods are implemented, but also not yet much more than that.

If you would like to see this project in a more stable state, then you might
consider pitching in with developer time (contributions are very welcome!)
or with financial support from you or your company.

This is a new project that builds off the exploration done in
https://github.com/mrocklin/dask-geopandas

Example
-------

Given a GeoPandas dataframe

.. code-block:: python

   import geopandas
   df = geopandas.read_file('...')

We can repartition it into a Dask-GeoPandas dataframe:

.. code-block:: python

   import dask_geopandas
   ddf = dask_geopandas.from_geopandas(df, npartitions=4)

Currently, this repartitions the data naively by rows. In the future, this will
also provide spatial partitioning to take advantage of the spatial structure of
the GeoDataFrame (but the current version still provides basic multi-core
parallelism).

The familiar spatial attributes and methods of GeoPandas are also available
and will be computed in parallel:

.. code-block:: python

   ddf.geometry.area.compute()
   ddf.within(polygon)


Additionally, if you have a distributed dask.dataframe you can pass columns of
x-y points to the ``set_geometry`` method. Currently, this only supports point
data.

.. code-block:: python

   import dask.dataframe as dd
   import dask_geopandas

   ddf = dd.read_csv('...')

   ddf = dask_geopandas.from_dask_dataframe(ddf)
   ddf = ddf.set_geometry(
       dask_geopandas.points_from_xy(ddf, 'latitude', 'longitude')
   )

Writing files (and reading back) is currently supported for the Parquet file
format:

.. code-block:: python

   ddf.to_parquet("path/to/dir/")
   ddf = dask_geopandas.read_parquet("path/to/dir/")


Installation
------------

This package depends on GeoPandas and Dask. In addition, it is recommended to
install PyGEOS, to have faster spatial operations and enable multithreading. See
https://geopandas.readthedocs.io/en/latest/install.html#using-the-optional-pygeos-dependency
for details.

One way is to use the ``conda`` package manager to create a new environment:

::

    conda create -n geo_env
    conda activate geo_env
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    conda install python=3 geopandas dask pygeos
    pip install git+git://github.com/geopandas/dask-geopandas.git
