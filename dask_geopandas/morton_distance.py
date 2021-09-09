import pandas as pd
from dask_geopandas.hilbert_distance import _continuous_to_discrete_coords


def _morton_distance(gdf, total_bounds, p):
    """
    Calculate distance of geometries along Morton curve

    The Morton curve is also known as Z-order https://en.wikipedia.org/wiki/Z-order_curve

    Parameters
    ----------
    gdf : GeoDataFrame
    total_bounds : array_like
        array containing xmin, ymin, xmax, ymax
    p : int
        precision of the Morton curve

    Returns
    -------
    type : pandas.Series
        Series containing distances from Morton curve

    """

    # Calculate bounds as numpy array
    bounds = gdf.bounds.to_numpy()
    # Calculate discrete coords based on total bounds and bounds
    coords = _continuous_to_discrete_coords(total_bounds, bounds, p)
    # Calculate distance from morton curve
    distances = _distances_from_coordinates(coords)

    return pd.Series(distances, index=gdf.index, name="morton_distance")


def _distances_from_coordinates(coords):
    """
    Calculate distances from geometry mid-points along Morton curve

    Parameters
    ----------
    coords : array_like
        x, y coordinate pairs based on mid-points of geoms

    Returns
    -------
    type : int
        Integer distances from Morton curve
    """

    return _part1by1(coords[:, 0]) | (_part1by1(coords[:, 1]) << 1)


def _part1by1(n):
    """
    Interleave bits by ninary magic numbers

    Based on #http://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN

    Parameters
    ----------
    n : np.array
        X or Y coordinates

    Returns
    -------
    n : int
        Interleaved bits
    """
    n &= 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555

    return n
