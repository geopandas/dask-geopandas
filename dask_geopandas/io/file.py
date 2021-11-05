from math import ceil

from dask.delayed import delayed
import dask.dataframe as dd


def read_file(path, npartitions=None, chunksize=None):
    """
    Read a GIS file into a Dask GeoDataFrame.

    This function requires `pyogrio <https://github.com/geopandas/pyogrio/>`__.

    Parameters
    ----------
    filename : str
        The absolute or relative path to the file or URL to
        be opened.
    npartitions : int, optional
        The number of partitions to create.
    chunksize : int, optional
        The number of rows per partition to use.

    """
    import pyogrio

    # TODO smart inference for a good default partition size ?
    if (npartitions is None) == (chunksize is None):
        raise ValueError("Exactly one of npartitions and chunksize must be specified.")

    total_size = pyogrio.read_info(path)["features"]

    if chunksize is None:
        chunksize = int(ceil(total_size / npartitions))

    row_offset = 0
    dfs = []

    while row_offset < total_size:
        batch_size = min(chunksize, total_size - row_offset)
        df = delayed(pyogrio.read_dataframe)(
            path, skip_features=row_offset, max_features=batch_size
        )
        dfs.append(df)
        row_offset += batch_size

    # TODO this could be inferred from read_info ?
    meta = pyogrio.read_dataframe(path, max_features=5)

    return dd.from_delayed(dfs, meta, prefix="read_file")
