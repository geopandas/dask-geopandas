=========
GeoSeries
=========
.. currentmodule:: dask_geopandas

Constructor
-----------
.. autosummary::
   :toctree: api/

   GeoSeries

General methods and attributes
------------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.area
   GeoSeries.boundary
   GeoSeries.bounds
   GeoSeries.total_bounds
   GeoSeries.length
   GeoSeries.geom_type
   GeoSeries.distance
   GeoSeries.representative_point
   GeoSeries.exterior
   GeoSeries.interiors
   GeoSeries.x
   GeoSeries.y
   GeoSeries.z

Unary predicates
----------------

.. autosummary::
   :toctree: api/

   GeoSeries.is_empty
   GeoSeries.is_ring
   GeoSeries.is_simple
   GeoSeries.is_valid
   GeoSeries.has_z


Binary Predicates
-----------------

.. autosummary::
   :toctree: api/

   GeoSeries.contains
   GeoSeries.crosses
   GeoSeries.disjoint
   GeoSeries.geom_equals
   GeoSeries.geom_equals_exact
   GeoSeries.intersects
   GeoSeries.overlaps
   GeoSeries.touches
   GeoSeries.within
   GeoSeries.covers
   GeoSeries.covered_by


Set-theoretic Methods
---------------------

.. autosummary::
   :toctree: api/

   GeoSeries.difference
   GeoSeries.intersection
   GeoSeries.symmetric_difference
   GeoSeries.union

Constructive Methods and Attributes
-----------------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.buffer
   GeoSeries.boundary
   GeoSeries.centroid
   GeoSeries.convex_hull
   GeoSeries.envelope
   GeoSeries.simplify

Affine transformations
----------------------

.. autosummary::
   :toctree: api/

   GeoSeries.affine_transform
   GeoSeries.rotate
   GeoSeries.scale
   GeoSeries.skew
   GeoSeries.translate

Aggregating and exploding
-------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.unary_union
   GeoSeries.explode

Serialization / IO / conversion
-------------------------------

.. autosummary::
   :toctree: api/

   GeoSeries.to_wkb
   GeoSeries.to_wkt

Projection handling
-------------------

.. autosummary::
   :toctree: api/

   GeoSeries.crs
   GeoSeries.set_crs
   GeoSeries.to_crs

Missing values
--------------

.. autosummary::
   :toctree: api/

   GeoSeries.fillna
   GeoSeries.isna

Overlay operations
------------------

.. autosummary::
   :toctree: api/

   GeoSeries.clip

Indexing
--------

.. autosummary::
   :toctree: api/

   GeoSeries.cx

Spatial partitioning
--------------------

.. autosummary::
   :toctree: api/

   GeoSeries.calculate_spatial_partitions
   GeoSeries.hilbert_distance
   GeoSeries.morton_distance
   GeoSeries.geohash



Methods of dask ``Series`` objects are also available, although not
all are applicable to geometric objects and some may return a
``Series`` rather than a ``GeoSeries`` result when appropriate. The methods
``isna()`` and ``fillna()`` have been
implemented specifically for ``GeoSeries`` and are expected to work
correctly.
