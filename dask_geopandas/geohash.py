import numpy as np
import pandas as pd
from pygeohash import encode


def _geohash(gdf, precision):

    """
    Calculate geohash for the mid points of each geometry
    int coordinates

    Parameters
    ----------
    gdf : GeoDataFrame
    precision : int
        precision of the Geohash

    Returns
    ---------
    Pandas Series containing geohash
    """

    # Calculate bounds
    bounds = gdf.bounds.to_numpy()
    # Calculate mid points based on bounds
    x_mids, y_mids = _calculate_mid_points(bounds)
    # Vectorize geohash for fast speed up
    geohash_vec = np.vectorize(encode)
    # Encode mid points of geometries using geohash
    geohash = geohash_vec(y_mids, x_mids, precision)

    return pd.Series(geohash, index=gdf.index, name="geohash")


def _calculate_mid_points(bounds):

    """
    Calculate geohash for the mid points of each geometry
    int coordinates

    Parameters
    ----------
    gdf : GeoDataFrame
    precision : int
        precision of the Geohash

    Returns
    ---------
    x_mids : mid points of x values
    y_mids : mid points of y values
    """

    # Calculate mid points for x and y bound coords
    x_mids = (bounds[:, 0] + bounds[:, 2]) / 2.0
    y_mids = (bounds[:, 1] + bounds[:, 3]) / 2.0

    return x_mids, y_mids
