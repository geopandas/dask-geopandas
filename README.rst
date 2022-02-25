dask-geopandas |conda| |pypi| |docs| |gitter|
=============================================

Parallel GeoPandas with Dask

Dask-GeoPandas is a project merging the geospatial capabilities of GeoPandas
and scalability of Dask. GeoPandas is an open source project designed to make working with geospatial data in Python easier. GeoPandas extends the datatypes used by pandas to allow spatial operations on geometric types.
Dask provides advanced parallelism and distributed out-of-core computation with a dask.dataframe module designed to scale
pandas. Since GeoPandas is an extension to the pandas DataFrame, the same way Dask scales pandas can also be applied to GeoPandas.

This project is a bridge between Dask and GeoPandas and offers geospatial capabilities of GeoPandas backed by Dask.

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


.. |pypi| image:: https://img.shields.io/pypi/v/dask-geopandas.svg
   :target: https://pypi.python.org/pypi/dask-geopandas/

.. |conda| image:: https://img.shields.io/conda/vn/conda-forge/dask-geopandas.svg
   :target: https://anaconda.org/conda-forge/dask-geopandas
   :alt: Conda Version

.. |docs| image:: https://readthedocs.org/projects/dask-geopandas/badge/?version=latest
   :target: https://dask-geopandas.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |gitter| image:: https://badges.gitter.im/geopandas/geopandas.svg
   :target: https://gitter.im/geopandas/geopandas
   :alt: Gitter
