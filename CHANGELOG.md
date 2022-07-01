Changelog
=========

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
