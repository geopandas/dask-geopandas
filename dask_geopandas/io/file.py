from math import ceil

from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.layers import DataFrameIOLayer
from dask.dataframe.core import new_dd_object


class FileFunctionWrapper:
    """
    GDAL File reader Function-Wrapper Class

    Reads data from disk to produce a partition (given row subset to read).
    """

    def __init__(self, columns):
        self.columns = columns
        self.read_geometry = True
        if columns is not None and "geometry" not in columns:
            self.read_geometry = False

    def project_columns(self, columns):
        """Return a new FileFunctionWrapper object with
        a sub-column projection.
        """
        if columns == self.columns:
            return self
        return FileFunctionWrapper(columns)

    def __call__(self, part):
        path, row_offset, batch_size = part

        import pyogrio

        df = pyogrio.read_dataframe(
            path,
            columns=self.columns,
            read_geometry=self.read_geometry,
            skip_features=row_offset,
            max_features=batch_size,
        )
        return df


def read_file(path, npartitions=None, chunksize=None, columns=None):
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

    # TODO this could be inferred from read_info ?
    meta = pyogrio.read_dataframe(path, columns=columns, max_features=5)

    # Define parts
    parts = []
    row_offset = 0

    while row_offset < total_size:
        batch_size = min(chunksize, total_size - row_offset)
        parts.append((path, row_offset, batch_size))
        row_offset += batch_size

    # Create Blockwise layer
    label = "read-file-"
    output_name = label + tokenize(path, chunksize, columns)
    layer = DataFrameIOLayer(
        output_name,
        columns,
        parts,
        FileFunctionWrapper(columns),
        label=label,
    )
    graph = HighLevelGraph({output_name: layer}, {output_name: set()})
    return new_dd_object(graph, output_name, meta, [None] * (len(parts) + 1))
