============
GeoDataFrame
============
.. currentmodule:: dask_geopandas

A ``GeoDataFrame`` is a tabular data structure that contains a column
which contains a ``GeoSeries`` storing geometry.

Constructor
-----------
.. autosummary::
   :toctree: api/

   GeoDataFrame

Serialization / IO / conversion
-------------------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.to_parquet

Projection handling
-------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.crs
   GeoDataFrame.set_crs
   GeoDataFrame.to_crs

Active geometry handling
------------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.set_geometry

Aggregating and exploding
-------------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.explode

Indexing
--------

.. autosummary::
   :toctree: api/

   GeoDataFrame.cx


All dask ``DataFrame`` methods are also available, although they may
not operate in a meaningful way on the ``geometry`` column. All methods
listed in `GeoSeries <geoseries>`__ work directly on an active geometry column of GeoDataFrame.

