from typing import Literal

import dask_expr as dx

import geopandas


def _drop(df: geopandas.GeoDataFrame, columns, errors):
    return df.drop(columns=columns, errors=errors)


def _validate_axis(axis=0, none_is_zero: bool = True) -> None | Literal[0, 1]:
    if axis not in (0, 1, "index", "columns", None):
        raise ValueError(f"No axis named {axis}")
    # convert to numeric axis
    numeric_axis: dict[str | None, Literal[0, 1]] = {"index": 0, "columns": 1}
    if none_is_zero:
        numeric_axis[None] = 0

    return numeric_axis.get(axis, axis)


class Drop(dx.expr.Drop):
    operation = staticmethod(_drop)
