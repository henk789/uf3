from argparse import ArgumentError
from typing import List, Tuple, Union
import jax.numpy as jnp
import numpy as onp
from jax.lax import dynamic_slice, scan, map, dynamic_update_slice
from jax import vmap, jit

from functools import partial

import warnings

from enum import Enum

Array = jnp.ndarray

from uf3.util import jax_utils


class BSplineBackend(Enum):
    Symbolic = 0
    DeBoor = 1


def ndSpline(
    coefficients: Array,
    knots: Union[Array, List[Array]],
    degrees: Union[int, Tuple[int]],
    backend=BSplineBackend.Symbolic,
):
    """
    Generates a spline function to evaluate the spline on given x.
    For more flexibility see: _ndSpline_unsafe

    The generated N-dimensional spline function takes N arguments of arrays
    with shape (n,) for n inputs.

    Args:
        coefficients: A jax.ndarray of shape (len(knots[i]) - degrees[i] - 1, ...)
        knots: A list of knots with knots for each dimension or
            a single array if the spline is one dimensional
        degrees: A tuple with the spline degrees for each dimension.
            Degree 3 for cubic splines is most optimized for.
            Or a single integer is the spline is one dimensional.
        backend: Choose how the B-splines are evaluated.
            BSplineBackend.Symbolic is faster, but only available for degree 3.
            BSplineBackend.DeBoor is more flexible.
            TODO benchmark backend properly
    """
    coefficients, _knots, degrees = jax_utils.check_inputs(
        knots, degrees, coefficients, padding=True
    )
    dimensions = len(degrees)
    spline = _ndSpline_unsafe(
        coefficients, _knots, degrees, backend=backend, featurization=False
    )

    def fn(*xs):
        # check number of arguments
        if len(xs) != dimensions:
            raise ArgumentError(
                f"This spline has dimension {dimensions}, but {len(xs)} arguments were given."
            )

        # return 0 for x outside range and check if for each dimension the same number of inputs is given
        mask = jnp.ones_like(xs[0], dtype=bool)
        n_x = len(xs[0])
        for i in range(dimensions):
            if n_x != xs[i]:
                raise ArgumentError(
                    "Each dimension of the input has to be the same length."
                )
            mask = mask & (knots[i][0] <= xs[i]) & (knots[i][-1] > xs[i])
        return spline(*xs) * mask


def _ndSpline_unsafe(
    coefficients: Array,
    knots: List[Array],
    degrees: Tuple[int],
    backend=BSplineBackend.Symbolic,
    featurization=False,
):
    """
    Generates a spline function to evaluate the spline on given x.
    The generated function can take a coefficients keyword argument,
    all other parameters are fixed.

    The generated N-dimensional spline function takes N arguments of arrays
    with shape (n,) for n inputs.
    Or use with vmap and inputs being of shape (N,1) for each axis.

    If featurization is True vmap has to be used and inputs thus shaped (N,1)

    Args:
        coefficients: A jax.ndarray of shape (len(knots[i]) - degrees[i] - 1, ...)
        knots: A list of knots with knots for each dimension
        degrees: A tuple with the spline degrees for each dimension.
            Degree 3 for cubic splines is most optimized for.
        backend: Choose how the B-splines are evaluated.
            BSplineBackend.Symbolic is faster, but only available for degree 3.
            BSplineBackend.DeBoor is more flexible.
            TODO benchmark backend properly
        featurisation: Choose whether the returned function shoud evaluate the spline
            or return an array of coefficients.shape with the contribution of each B-spline.
            Note that if set to True, vmap has to be used.
    """
    if k != 3:
        backend = BSplineBackend.DeBoor
        warnings.warn(
            "The symbolic backend is only available for k=3. Changed to DeBoor."
        )

    min = []
    max = []
    s = []
    for t, k in zip(knots, degrees):
        min.append(t[0])
        max.append(t[-1])
        s.append(vmap(partial(bspline_factors, t, k=k, basis=backend)))

    min = jnp.asarray(min)
    max = jnp.asarray(max)
    x_dim = len(knots)

    indices = list(range(1, x_dim + 1))
    selector = [[0, i] for i in indices]

    if featurization:
        out = list(range(1, x_dim + 1))
    else:
        out = [0]

    @jit
    def spline_fn(*xs, coefficients=coefficients):
        """
        Use as is with inputs being arrays for each axis

        Use with vmap and inputs being of shape (N,1) for each axis.

        If featurization is True vmap has to be used and inputs thus shaped (N,1)
        """
        data = []
        in_cutoff = True
        for i in range(x_dim):
            data.append(s[i](xs[i]))
            in_cutoff = (xs[i] >= min[i]) & (xs[i] < max[i]) & in_cutoff

        # jnp.einsum(coefficients, *results of basis splines and axis, coefficient axis for featurization)
        # jnp.einsum(coefficients, [0,1,2], A, [0], B, [1], C, [2])
        arg = [item for sublist in zip(data, selector) for item in sublist]
        return jnp.where(in_cutoff, jnp.einsum(coefficients, indices, *arg, out), 0)

    return spline_fn


@partial(jit, static_argnames=["k", "basis", "safe"])
def bspline_factors(knots: Array, x, k=3, basis=BSplineBackend.Symbolic, safe=False):
    """
    safe = False -> Will silently return wrong values if knots[k] > x or knots[-k-1] <= x

    It is generally faster to ensure the safety property in advance as safe=True results
    in inefficient padding. You should apply padding in advance if necessary.
    See: uf3.utils.jax_utils.add_padding
    """
    i = jnp.searchsorted(knots, x, side="right")
    if safe:
        t = dynamic_slice(jnp.pad(knots, (k, k), "edge"), (i,), (2 * k,))
    else:
        t = dynamic_slice(knots, (i - k,), (2 * k,))
    max = len(knots) - k - 1
    res = jnp.zeros(max)

    if basis is BSplineBackend.Symbolic:
        if k == 3:
            r = symbolic_basis(t, x)
        else:
            raise NotImplementedError("Symbolic backend only for k=3")
    if basis is BSplineBackend.DeBoor:
        r = deBoor_basis(k, t, x)

    return dynamic_update_slice(res, r, (i - k - 1,))
    


def deBoor_basis(k: int, t: Array, x):
    """
    Requires len(t) = 2*k
    And t[k-1] <= x < t[k]

    This implementation is similar to SciPys _deBoor_D in __fitpack.h
    Which itself is an adaptation of deBoors algorithm.
    """

    f = lambda c, a: (None, dynamic_slice(t, (a,), (k,)))

    _, Order = scan(f, None, jnp.arange(1, k + 1))

    # Division by 0 should result in 0
    # TODO find relevant section in the deBoor book
    base = Order - t[:k]
    pred = base != 0

    A = ((x - t[:k]) / base) * pred

    B = ((Order - x) / base) * pred

    def do(c, a):
        # c = c.at[k + 1].set(0.0) #necessary?
        # A[:,iteration] = a[0]
        c = c.at[k + 2 : 2 * k + 2].set(c[1 : k + 1] * a[0])

        # B[:,iteration] = a[1]
        c = c.at[k + 1 : 2 * k + 1].add(c[1 : k + 1] * a[1])

        c = c.at[: k + 1].set(c[k + 1 : 2 * k + 2])

        return (c, None)

    init = jnp.zeros(2 * (k + 1), dtype=x.dtype)
    init = init.at[k].set(1.0)

    out, _ = scan(do, init, jnp.stack([A, B], axis=1))

    return out[: k + 1]


def symbolic_basis(t: Array, x):
    """
    Requires len(t) = 2*k
    And t[k-1] <= x < t[k]

    Basis evaluation for cubic B-splines based on the symbolic evaluation of the recursive definition.
    """
    k = 3

    out = jnp.zeros(k + 1, x.dtype)

    t32 = t[3] - t[2]
    B11 = jnp.where(t32 == 0.0, 0.0, (t[3] - x) / t32)
    B21 = jnp.where(t32 == 0.0, 0.0, (x - t[2]) / t32)

    t31 = t[3] - t[1]
    t42 = t[4] - t[2]
    B02 = jnp.where(t31 == 0.0, 0.0, ((t[3] - x) / t31) * B11)
    B12a = jnp.where(t31 == 0.0, 0.0, ((x - t[1]) / t31) * B11)
    B12b = jnp.where(t42 == 0.0, 0.0, ((t[4] - x) / t42) * B21)
    B22 = jnp.where(t42 == 0.0, 0.0, ((x - t[2]) / t42) * B21)
    B12 = B12a + B12b

    t30 = t[3] - t[0]
    t41 = t[4] - t[1]
    t52 = t[5] - t[2]
    Bn13 = jnp.where(t30 == 0.0, 0.0, ((t[3] - x) / t30) * B02)
    B03a = jnp.where(t30 == 0.0, 0.0, ((x - t[0]) / t30) * B02)
    B03b = jnp.where(t41 == 0.0, 0.0, ((t[4] - x) / t41) * B12)
    B13a = jnp.where(t41 == 0.0, 0.0, ((x - t[1]) / t41) * B12)
    B13b = jnp.where(t52 == 0.0, 0.0, ((t[5] - x) / t52) * B22)
    B23 = jnp.where(t52 == 0.0, 0.0, ((x - t[2]) / t52) * B22)
    B03 = B03a + B03b
    B13 = B13a + B13b

    out = out.at[0].set(Bn13)
    out = out.at[1].set(B03)
    out = out.at[2].set(B13)
    out = out.at[3].set(B23)

    return out

    # def deBoor_basis_reference(knots: onp.ndarray, k: int, x):
    """
    B-splines evaluation with the recursive form.
    Left here only for reference.
    """


#     def f(x, i, k):
#         if k == 0:
#             if knots[i] <= x and x < knots[i + 1]:
#                 return 1
#             else:
#                 return 0
#         a = (x - knots[i]) / (knots[k + i] - knots[i])
#         b = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1])
#         return a * f(x, i, k - 1) + b * f(x, i + 1, k - 1)

#     return f(x, 0, k)
