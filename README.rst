dask-geopandas
==============

Parallel GeoPandas with Dask

Status
------

**EXPERIMENTAL** This project is in an early state.

If you would like to see this project in a more stable state, then you might
consider pitching in with developer time (contributions are very welcome!)
or with financial support from you or your company.

This is a new project that builds off the exploration done in
https://github.com/mrocklin/dask-geopandas

Documentation
-------------

See the documentation on https://dask-geopandas.readthedocs.io/en/latest/

Installation
------------

This package depends on GeoPandas, Dask and PyGEOS.

One way to install all required dependencies is to use the ``conda`` package manager to
create a new environment:

::

    conda create -n geo_env
    conda activate geo_env
    conda config --env --add channels conda-forge
    conda config --env --set channel_priority strict
    conda install dask-geopandas



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

The familiar spatial attributes and methods of GeoPandas are also available
and will be computed in parallel:

.. code-block:: python

   ddf.geometry.area.compute()
   ddf.within(polygon)


