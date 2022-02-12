============
GeoDataFrame
============
.. currentmodule:: dask_geopandas

A ``GeoDataFrame`` is a tabular data structure that contains a column
which stores geometries (a ``GeoSeries``).

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
   GeoDataFrame.to_wkb
   GeoDataFrame.to_wkt

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
   GeoDataFrame.rename_geometry

Aggregating and exploding
-------------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.explode
   GeoDataFrame.dissolve

Spatial joins
-------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.sjoin

Overlay operations
------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.clip

Indexing
--------

.. autosummary::
   :toctree: api/

   GeoDataFrame.cx

Spatial partitioning
--------------------

.. autosummary::
   :toctree: api/

   GeoDataFrame.spatial_shuffle


All dask ``DataFrame`` methods are also available, although they may
not operate in a meaningful way on the ``geometry`` column. All methods
listed in `GeoSeries <geoseries>`__ work directly on an active geometry column of GeoDataFrame.

