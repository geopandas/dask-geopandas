from math import ceil

from pandas import RangeIndex

from dask.base import tokenize
from dask.dataframe.core import new_dd_object
from dask.highlevelgraph import HighLevelGraph

from .. import backends


class FileFunctionWrapper:
    """
    GDAL File reader Function-Wrapper Class

    Reads data from disk to produce a partition (given row subset to read).
    """

    def __init__(self, layer, columns):
        self.layer = layer
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
        return FileFunctionWrapper(self.layer, columns)

    def __call__(self, part):
        path, row_offset, batch_size = part

        import pyogrio

        df = pyogrio.read_dataframe(
            path,
            layer=self.layer,
            columns=self.columns,
            read_geometry=self.read_geometry,
            skip_features=row_offset,
            max_features=batch_size,
        )
        df.index = RangeIndex(row_offset, row_offset + batch_size)
        return df


def read_file(
    path, npartitions=None, chunksize=None, layer=None, columns=None, **kwargs
):
    """
    Read a GIS file into a Dask GeoDataFrame.

    This function requires `pyogrio <https://github.com/geopandas/pyogrio/>`__.

    Parameters
    ----------
    path : str
        The absolute or relative path to the file or URL to
        be opened.
    npartitions : int, optional
        The number of partitions to create. Either this or `chunksize` should
        be specified.
    chunksize : int, optional
        The number of rows per partition to use. Either this or `npartitions`
        should be specified.
    layer : int or str, optional (default: first layer)
        If an integer is provided, it corresponds to the index of the layer
        with the data source.  If a string is provided, it must match the name
        of the layer in the data source.  Defaults to first layer in data source.
    columns : list-like, optional (default: all columns)
        List of column names to import from the data source.  Column names must
        exactly match the names in the data source, and will be returned in
        the order they occur in the data source.  To avoid reading any columns,
        pass an empty list-like.

    """
    try:
        import pyogrio
    except ImportError as err:
        raise ImportError(
            "The 'read_file' function requires the 'pyogrio' package, but it is "
            "not installed or does not import correctly."
            f"\nImporting pyogrio resulted in: {err}"
        )

    from dask.layers import DataFrameIOLayer

    # TODO smart inference for a good default partition size ?
    if (npartitions is None) == (chunksize is None):
        raise ValueError("Exactly one of npartitions and chunksize must be specified.")

    if "skip_features" in kwargs or "max_features" in kwargs:
        # TODO we currently use those keywords already for reading in each
        # partition (we would need to take those into account for determining
        # the part start/ends)
        raise ValueError(
            "The 'skip_features'/'max_feature' keywords are not yet supported"
        )
    if kwargs:
        raise ValueError("Additional pyogrio keywords are not yet supported")

    total_size = pyogrio.read_info(path, layer=layer)["features"]

    if chunksize is None:
        chunksize = int(ceil(total_size / npartitions))

    # TODO this could be inferred from read_info ?
    read_geometry = True
    if columns is not None and "geometry" not in columns:
        read_geometry = False
    meta = pyogrio.read_dataframe(
        path, layer=layer, columns=columns, read_geometry=read_geometry, max_features=5
    )

    # Define parts
    parts = []
    row_offset = 0
    divs = [row_offset]

    while row_offset < total_size:
        batch_size = min(chunksize, total_size - row_offset)
        parts.append((path, row_offset, batch_size))
        row_offset += batch_size
        divs.append(row_offset)
    # Set the last division value to be the largest index value in the last partition
    divs[-1] = divs[-1] - 1

    # Create Blockwise layer
    label = "read-file-"
    output_name = label + tokenize(path, chunksize, layer, columns)
    layer = DataFrameIOLayer(
        output_name,
        columns,
        parts,
        FileFunctionWrapper(layer, columns),
        label=label,
    )
    graph = HighLevelGraph({output_name: layer}, {output_name: set()})

    if backends.QUERY_PLANNING_ON:
        from dask_expr import from_graph

        result = from_graph(
            graph,
            meta,
            divs,
            [(output_name, i) for i in range(len(divs) - 1)],
            "read_file",
        )
        return result
    else:
        return new_dd_object(graph, output_name, meta, divs)
