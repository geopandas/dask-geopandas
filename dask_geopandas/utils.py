def _calculate_mid_points(bounds):
    """
    Calculate geohash based on the mid points of each geometry

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
