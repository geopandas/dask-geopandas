import numpy as np
import pandas as pd
from .utils import _calculate_mid_points


# Implementation based on deprecated https://pypi.org/project/neathgeohash/#description


def _geohash(gdf, precision):
    """
    Calculate geohash based on the middle points of the geometry bounds
    for a given precision

    Parameters
    ----------
    gdf : GeoDataFrame
    precision : int
        precision of the Geohash

    Returns
    ---------
    type : pandas.Series
        Series containing geohash
    """

    # Calculate bounds
    bounds = gdf.bounds.to_numpy()
    # Calculate mid points based on bounds
    x_mids, y_mids = _calculate_mid_points(bounds)
    # Create pairs of x and y midpoints
    coords = np.array([y_mids, x_mids]).T
    # Encode coords with Geohash
    geohash = encode_geohash(coords, precision)

    return pd.Series(geohash, index=gdf.index, name="geohash")


def encode_geohash(coords, precision):
    """
    Calculate geohash based on coordinates for a
    given precision

    Parameters
    ----------
    coords : array_like of shape (n, 2)
        array of [x, y] pairs
    precision : int
        precision of the Geohash

    Returns
    ---------
    geohash: array containing geohashes for each mid point
    """

    quantized_coords = _encode_quantize_points(coords)
    g_uint64 = _encode_into_uint64(quantized_coords)
    gs_uint8_mat = _encode_base32(g_uint64)
    geohash = _encode_unicode(gs_uint8_mat, precision)

    return geohash


def _encode_quantize_points(coords):
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
    quantized_coords : array_like
        quantized coordinate pairs
    """

    _q = np.array([(2.0 ** 32 / 180, 0), (0, 2.0 ** 32 / (180 * 2))], dtype="float64")

    quantized_coords = coords + np.array((90, 180))
    quantized_coords = np.dot(quantized_coords, _q)
    quantized_coords = np.floor(quantized_coords)

    return quantized_coords


def _encode_into_uint64(quantized_coords):
    """
    Encode quantized coordinates into uint64

    Implementation based on "Geohash in Golang Assembly"
    blog (https://mmcloughlin.com/posts/geohash-assembly)

    Parameters
    ----------
    quantized_coords : array_like
        quantized coordinate pairs

    Returns
    ---------
    array_like of shape (n, 2)
        coordinate pairs encoded to uint64 values
        quantized coordinate pairs
    """

    __s1 = np.array([(1, 0), (0, 2)], dtype="uint64")

    g_uint64 = np.uint64(quantized_coords)
    g_uint64 = g_uint64.reshape(-1, 2)
    g_uint64 = np.bitwise_and(
        np.bitwise_or(g_uint64, np.left_shift(g_uint64, 16)), 0x0000FFFF0000FFFF
    )
    g_uint64 = np.bitwise_and(
        np.bitwise_or(g_uint64, np.left_shift(g_uint64, 8)), 0x00FF00FF00FF00FF
    )
    g_uint64 = np.bitwise_and(
        np.bitwise_or(g_uint64, np.left_shift(g_uint64, 4)), 0x0F0F0F0F0F0F0F0F
    )
    g_uint64 = np.bitwise_and(
        np.bitwise_or(g_uint64, np.left_shift(g_uint64, 2)), 0x3333333333333333
    )
    g_uint64 = np.bitwise_and(
        np.bitwise_or(g_uint64, np.left_shift(g_uint64, 1)), 0x5555555555555555
    )
    g_uint64 = np.dot(g_uint64, __s1)
    g_uint64 = np.bitwise_or(g_uint64[:, 0], g_uint64[:, 1])
    g_uint64 = np.right_shift(g_uint64, 4)

    return g_uint64


def _encode_base32(g_uint64):
    """
    Encode quantized coordinates into uint64

    Implementation based on "Geohash in Golang Assembly"
    blog (https://mmcloughlin.com/posts/geohash-assembly)
    
    Parameters
    ----------
    g_uint64 : array_like
        coordinate pairs encoded to uint64 values

    Returns
    ---------
    array_like of shape (n, 12)
        with encoded base32 values
    """

    mask = np.uint64(0x1F)  # equivelant to 32-1

    c11 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 0), mask)).flatten()
    c10 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 5), mask)).flatten()
    c9 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 10), mask)).flatten()
    c8 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 15), mask)).flatten()
    c7 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 20), mask)).flatten()
    c6 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 25), mask)).flatten()
    c5 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 30), mask)).flatten()
    c4 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 35), mask)).flatten()
    c3 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 40), mask)).flatten()
    c2 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 45), mask)).flatten()
    c1 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 50), mask)).flatten()
    c0 = np.uint8(np.bitwise_and(np.right_shift([g_uint64], 55), mask)).flatten()

    return np.column_stack((c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11))


def _encode_unicode(gs_uint8_mat, precision):
    """
    Encode base32 pairs into unicode string 
    
    Parameters
    ----------
    gs_uint8_mat : array_like
        coordinate pairs
    precision : int
        precision of the Geohash

    Returns
    ---------
    array_like of shape (n, precision)
        containing geohash for a given precision
    """

    # Replacement values
    gs_uint8 = np.where(gs_uint8_mat == 0, 48, gs_uint8_mat)  # 0
    gs_uint8 = np.where(gs_uint8 == 1, 49, gs_uint8)  # 1
    gs_uint8 = np.where(gs_uint8 == 2, 50, gs_uint8)  # 2
    gs_uint8 = np.where(gs_uint8 == 3, 51, gs_uint8)  # 3
    gs_uint8 = np.where(gs_uint8 == 4, 52, gs_uint8)  # 4
    gs_uint8 = np.where(gs_uint8 == 5, 53, gs_uint8)  # 5
    gs_uint8 = np.where(gs_uint8 == 6, 54, gs_uint8)  # 6
    gs_uint8 = np.where(gs_uint8 == 7, 55, gs_uint8)  # 7
    gs_uint8 = np.where(gs_uint8 == 8, 56, gs_uint8)  # 8
    gs_uint8 = np.where(gs_uint8 == 9, 57, gs_uint8)  # 9
    gs_uint8 = np.where(gs_uint8 == 10, 98, gs_uint8)  # b
    gs_uint8 = np.where(gs_uint8 == 11, 99, gs_uint8)  # c
    gs_uint8 = np.where(gs_uint8 == 12, 100, gs_uint8)  # d
    gs_uint8 = np.where(gs_uint8 == 13, 101, gs_uint8)  # e
    gs_uint8 = np.where(gs_uint8 == 14, 102, gs_uint8)  # f
    gs_uint8 = np.where(gs_uint8 == 15, 103, gs_uint8)  # g
    gs_uint8 = np.where(gs_uint8 == 16, 104, gs_uint8)  # h
    gs_uint8 = np.where(gs_uint8 == 17, 106, gs_uint8)  # j
    gs_uint8 = np.where(gs_uint8 == 18, 107, gs_uint8)  # k
    gs_uint8 = np.where(gs_uint8 == 19, 109, gs_uint8)  # m
    gs_uint8 = np.where(gs_uint8 == 20, 110, gs_uint8)  # n
    gs_uint8 = np.where(gs_uint8 == 21, 112, gs_uint8)  # p
    gs_uint8 = np.where(gs_uint8 == 22, 113, gs_uint8)  # q
    gs_uint8 = np.where(gs_uint8 == 23, 114, gs_uint8)  # r
    gs_uint8 = np.where(gs_uint8 == 24, 115, gs_uint8)  # s
    gs_uint8 = np.where(gs_uint8 == 25, 116, gs_uint8)  # t
    gs_uint8 = np.where(gs_uint8 == 26, 117, gs_uint8)  # u
    gs_uint8 = np.where(gs_uint8 == 27, 118, gs_uint8)  # v
    gs_uint8 = np.where(gs_uint8 == 28, 119, gs_uint8)  # w
    gs_uint8 = np.where(gs_uint8 == 29, 120, gs_uint8)  # x
    gs_uint8 = np.where(gs_uint8 == 30, 121, gs_uint8)  # y
    gs_uint8 = np.where(gs_uint8 == 31, 122, gs_uint8)  # z

    gs_uint8.dtype = np.dtype("|S12")

    return gs_uint8.flatten().astype("U%s" % precision)
