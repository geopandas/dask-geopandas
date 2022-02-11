"""
Geohash implementation

The code is originally based on the neathgeohash package,
Copyright (c) 2020 Marek Dwulit, MIT License
(https://pypi.org/project/neathgeohash/#description).
The vectorized implementation for quantization and bit interleaving is in turn based on,
"Geohash in Golang Assembly" blog (https://mmcloughlin.com/posts/geohash-assembly).

"""
import warnings

import numpy as np
import pandas as pd


def _geohash(gdf, as_string, precision):
    """
    Calculate geohash based on the middle points of the geometry bounds
    for a given precision

    Parameters
    ----------
    gdf : GeoDataFrame
    as_string : bool
        to return string or int Geohash
    precision : int
        precision of the string Geohash


    Returns
    ---------
    type : pandas.Series
        Series containing geohash
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "GeoSeries.isna() previously returned True", UserWarning
        )
        if gdf.is_empty.any() | gdf.geometry.isna().any():
            raise ValueError(
                "Geohash cannot be computed on a GeoSeries with empty or "
                "missing geometries.",
            )

    # Calculate bounds
    bounds = gdf.bounds.to_numpy()
    # Calculate mid points based on bounds
    x_mids, y_mids = _calculate_mid_points(bounds)
    # Create pairs of x and y midpoints
    coords = np.array([y_mids, x_mids]).T
    # Encode coords with Geohash
    geohash = encode_geohash(coords, as_string, precision)

    return pd.Series(geohash, index=gdf.index, name="geohash")


def _calculate_mid_points(bounds):
    """
    Calculate middle points based on the geometry bounds

    Parameters
    ----------
    bounds : array_like
        array containing xmin, ymin, xmax, ymax

    Returns
    ---------
    x_mids : mid points of x values
    y_mids : mid points of y values
    """

    # Calculate mid points for x and y bound coords
    x_mids = (bounds[:, 0] + bounds[:, 2]) / 2.0
    y_mids = (bounds[:, 1] + bounds[:, 3]) / 2.0

    return x_mids, y_mids


def encode_geohash(coords, as_string, precision):
    """
    Calculate geohash based on coordinates for a
    given precision

    Parameters
    ----------
    coords : array_like of shape (n, 2)
        array of [x, y] pairs
    as_string : bool
        to return string or int Geohash
    precision : int
        precision of the string Geohash
    Returns
    ---------
    geohash: array containing either int or string
        geohashes for each mid point
    """

    quantized_coords = _quantize_points(coords)
    int_geohash = _encode_into_uint64(quantized_coords)

    if not as_string:
        return int_geohash

    gs_uint8_mat = _encode_base32(int_geohash)
    str_geohash = _encode_unicode(gs_uint8_mat, precision)

    return str_geohash


def _quantize_points(coords):
    """
    Quantize coordinates by mapping onto
    unit intervals [0, 1] and multiplying by 2^32.

    Parameters
    ----------
    coords : array_like of shape (n, 2)
        array of [x, y] pairs
        coordinate pairs

    Returns
    ---------
    array_like of shape (n, 2)
    """

    _q = np.array([(2.0 ** 32 / 180, 0), (0, 2.0 ** 32 / (180 * 2))], dtype="float64")

    quantized_coords = coords + np.array([90, 180])
    quantized_coords = np.dot(quantized_coords, _q)
    quantized_coords = np.floor(quantized_coords)

    return quantized_coords


def _encode_into_uint64(quantized_coords):
    """

    Encode quantized coordinates into uint64
    using both spreading and interleaving bits

    Implementation based on "Geohash in Golang Assembly"
    blog (https://mmcloughlin.com/posts/geohash-assembly)

    Parameters
    ----------
    quantized_coords : array_like of shape (n, 2)
        array of quantized coordinate pairs

    Returns
    ---------
    array_like of shape (n, 2)
        coordinate pairs encoded to uint64 values
        quantized coordinate pairs
    """

    # spread out 32 bits of x into 64 bits, where the bits occupy even bit positions.
    x = quantized_coords.astype(np.uint64)
    x = x.reshape(-1, 2)
    x = (x | (x << 16)) & 0x0000FFFF0000FFFF
    x = (x | (x << 8)) & 0x00FF00FF00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F
    x = (x | (x << 2)) & 0x3333333333333333
    x = (x | (x << 1)) & 0x5555555555555555

    # Dot
    __s1 = np.array([(1, 0), (0, 2)], dtype=np.uint64)
    x = x @ __s1
    # Interleave x and y bits so that x and y occupy even and odd bit levels
    x = x[:, 0] | x[:, 1]
    x = x >> 4

    return x


def _encode_base32(encoded_uint64):
    """
    Encode quantized coordinates into base32 pairs.
    Encoding starts at the highest bit, consuming 5 bits for each character precision.
    This means encoding happens 12 times for the 12 character precision or 60 bits.

    Implementation is based on "Geohash in Golang Assembly"
    blog (https://mmcloughlin.com/posts/geohash-assembly)

    Parameters
    ----------
    g_uint64 : array_like
        coordinate pairs encoded to uint64 values

    Returns
    ---------
    array_like of shape (n, 12)
        with base 32 values as 8-bit unasigned integer
    """
    # Define 32 bit mask
    mask = np.uint64(0x1F).flatten()  # equivalent to 32-1
    # Return array for each character
    c11 = (encoded_uint64 >> 0) & mask
    c10 = (encoded_uint64 >> 5) & mask
    c9 = (encoded_uint64 >> 10) & mask
    c8 = (encoded_uint64 >> 15) & mask
    c7 = (encoded_uint64 >> 20) & mask
    c6 = (encoded_uint64 >> 25) & mask
    c5 = (encoded_uint64 >> 30) & mask
    c4 = (encoded_uint64 >> 35) & mask
    c3 = (encoded_uint64 >> 40) & mask
    c2 = (encoded_uint64 >> 45) & mask
    c1 = (encoded_uint64 >> 50) & mask
    c0 = (encoded_uint64 >> 55) & mask

    # Stack each array vertically
    return np.column_stack((c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)).astype(
        "uint8"
    )


def _encode_unicode(encoded_base32, precision):
    """
    Encode base32 pairs into geohash bytes with an option to return
    the geohash in unicode format

    Parameters
    ----------
    encoded_base32 : array_like
        coordinate pairs
    p : int
        precision of the Geohash

    Returns
    ---------
    array_like of shape (n, precision)
        containing geohash for a given precision
    """

    # Define replacement values
    replacement = np.array(
        [
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            98,
            99,
            100,
            101,
            102,
            103,
            104,
            106,
            107,
            109,
            110,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
        ],
        dtype="uint8",
    )

    encoded_base32 = replacement[encoded_base32]

    encoded_base32 = encoded_base32.view(np.dtype("|S12"))
    return encoded_base32.flatten().astype(f"U{precision}")
