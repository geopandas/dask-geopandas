Changelog
=========

Version 0.5.0 (upcoming)
------------------------

Deprecations and compatibility notes:

- The deprecated `geom_almost_equals` method has been removed. Use `geom_equals_exact` instead.


Version 0.4.3 (January, 2025)
-----------------------------

Packaging:

- `dask>=2025.1.0` is now required.
- `python>=3.10` is now required.

Bug fixes:

- Fixed `GeoDataFrame.drop` returning a `GeoDataFrame`
  instead of a `DataFrame`, when dropping the geometry
  column (#321).

Version 0.4.2 (September 24, 2024)
----------------------------------

Bug fixes:

- Ensure `read_file()` produces a correct empty meta object, avoiding later
  errors in `spatial_shuffle()` (#302).
- Fix in `sjoin()` to work with GeoDataFrames after a `spatial_shuffle()` (#303).

Packaging:

- `distributed` was dropped as a required dependency, only depending on
  `dask[dataframe]` (#258).


Version 0.4.1 (June 25, 2024)
-----------------------------

Bug fixes:

- Allow to run dask-geopandas with recent dask versions without using query
  planning (without dask-expr being installed).

Packaging:

- The `dask` dependency was updated to `dask[dataframe]` in pyproject.toml (when
  installing from source or binary wheels from PyPI). This ensures dask-expr
  gets installed automatically for recent versions of dask.

Version 0.4.0 (June 24, 2024)
-----------------------------

Enhancements:

- Added preliminary support for dask's new query planning (dask >= 2024.3.0) (#285).
- Added support for using dask-geopandas with distributed's P2P shuffle (this
  requires the latest distributed>=2024.6.0 to work) (#295).
- Added new `from_wkb()` and `from_wkt()` functions to convert a dask Series of
  WKB or WKT values into a dask-geopandas GeoSeries (#293).

Notes on dependencies:

- Removed support for PyGEOS, now requiring Shapely >= 2 (#280).
- Updated minimum supported versions of dependencies, now requiring Python 3.9,
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
