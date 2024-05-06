Changelog
=========

Version 0.4.0 (March ??, 2024)
------------------------------

- Removed support for PyGEOS, now requiring Shapely >= 2

Updated minimum supported versions of dependencies, now requiring Python 3.9,
GeoPandas 0.12, numpy 1.23 and dask/distributed 2022.06.0.

Version 0.3.1 (April 28, 2023)
------------------------------

Bug fixes:

- Compatibility with dask >= 2023.4 and changes regarding ``use_nullable_dtypes``
  keyword (#242).
- Ensure ``spatial_partitions`` are preserved when serialized deserialized
  with pickle (#237).

Version 0.3.0 (January 23, 2023)
--------------------------------

Enhancements:

- Dask-GeoPandas is now compatible with Shapely 2.0 (and if this version is
  installed, no longer requires PyGEOS)

Bug fixes:

- Compatibility with dask >= 2022.12 for ``read_parquet()`` (#230) and for
  ``dissolve()`` (#229)
- Fix the ``spatial_partitions`` of the result of ``sjoin()`` (#216)

Version 0.2.0 (July 1, 2022)
----------------------------

Enhancements:

- Optionally skip spatial bounds in ``read_parquet`` (#203)

Bug fixes:

- Don't put ``GeoSeries`` in ``map_partitions`` kwarg (#205)

Version 0.1.3 (June 21, 2021)
-----------------------------

Compatibility:

- MAINT: use ``predicate`` instead of ``op`` in ``sjoin`` (#204)

Version 0.1.2 (June 20, 2021)
-----------------------------

Bug fixes:

- Update ``to_parquet`` to handle custom schema (to fix writing partitions with all missing data) (#201)

Version 0.1.1 (June 19, 2021)
-----------------------------

Bug fixes:

- Compat with dask 2022.06.0: fix schema inference in ``to_parquet`` (#199)
- Remove custom ``__dask_postcompute__`` (#191)
- BUG: persist ``spatial_partitions`` information in ``persist()`` (#192)
