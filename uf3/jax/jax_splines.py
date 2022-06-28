from argparse import ArgumentError
from typing import List, Tuple, Union
import jax.numpy as jnp
import numpy as onp
from jax.lax import dynamic_slice, scan, map, dynamic_update_slice
from jax import vmap, jit

from functools import partial

Array = jnp.ndarray

from uf3.util import jax_utils


def ndSpline(
    coefficients: Array,
    knots: Union[Array, List[Array]],
    degrees: Union[int, Tuple[int]],
):
    coefficients, knots, degrees = jax_utils.check_inputs(knots, degrees, coefficients, padding=True)

    return _ndSpline_unsafe(coefficients, knots, degrees)


def _ndSpline_unsafe(coefficients: Array, knots: List[Array], degrees: Tuple[int]):

    if len(degrees) == 1:
        return spline_1d(coefficients, knots, degrees)
    if len(degrees) == 3:
        return spline_3d(coefficients, knots, degrees)

    # raise NotImplementedError("Arbitrary ND Splines are not supported at the moment.")
    min = []
    max = []
    min = jnp.asarray(min)
    max = jnp.asarray(max)
    s = []
    for t, k in zip(knots, degrees):
        min.append(t[0])
        max.append(t[-1])
        s.append(
            jit(vmap(partial(deBoor_basis_unsafe, k, t)))
        )  # TODO benchmark with and without this jit

    x_dim = len(degrees)

    def spline_fn(x: Array):

        mask = jnp.logical_or(jnp.any(x < min, 1), jnp.any(x >= max, 1))

        x = jnp.where(mask[:, jnp.newaxis], x, min)

        # jnp.einsum(coefficients, *results of basis splines and axis, coefficient axis for featurization)
        # jnp.einsum(coefficients, [0,1,2], A, [0], B, [1], C, [2])

        # Test (from ndsplines line 221):
        #  coefficient_selector = tuple(self.coefficient_selector[:num_points, ...].swapaxes(0,-1)) + (slice(None),)
        result = jnp.einsum()  # TODO

        return jnp.where(mask, result, 0)

    return jit(spline_fn)


def spline_1d(coefficients: Array, knots: List[Array], degrees: Tuple[int]):
    """
    Optimized implementation specifically for 2-body potentials
    """
    pass


def spline_3d(coefficients: Array, knots: List[Array], degrees: Tuple[int]):
    """
    Optimized implementation specifically for 3-body potentials
    """
    pass


def deBoor_basis(k: int, knots: Array, x):
    """
    knots of shape (2*k+1,) with 2*k+1 knots for the basis spline of degree k
    (degree k represents k-th order polynomials, you will see k+1 used as the spline degree in some literature)
    The values of the knots after knots[k+1] do not matter as long as they are larger.
    They are simply needed since JAX requires static array sizes.

    x has to be smaller than the last knot, but larger or equal to the first knot
    Condition: knots[0] <= x < knots[-1]
    """
    # this time all knots - returns the solution to all 4 non zero basis functions

    i = jnp.argmax(knots > x)
    t = dynamic_slice(jnp.pad(knots, (k, k), "edge"), (i,), (2 * k,))

    f = lambda c, a: (None, dynamic_slice(t, (a,), (k,)))

    _, Order = scan(f, None, jnp.arange(1, k + 1))

    # Division by 0 should result in 0
    # TODO find relevant section in the deBoor book
    A = jnp.nan_to_num((x - t[:k]) / (Order - t[:k]), False, 0.0, 0.0, 0.0)

    B = jnp.nan_to_num((Order - x) / (Order - t[:k]), False, 0.0, 0.0, 0.0)

    def do(c, a):
        c = c.at[k + 1].set(0.0)
        # A = a[0]
        c = c.at[k + 2 : 2 * k + 2].set(c[1 : k + 1] * a[0])

        # B = a[1]
        c = c.at[k + 1 : 2 * k + 1].add(c[1 : k + 1] * a[1])

        c = c.at[: k + 1].set(c[k + 1 : 2 * k + 2])

        return (c, None)

    init = jnp.zeros(2 * (k + 1), dtype=knots.dtype)
    init = init.at[k].set(1.0)

    out, _ = scan(do, init, jnp.stack([A, B], axis=1))

    return (i - k - 1, out[: k + 1])


def deBoor_getBasis(k: int):
    return partial(deBoor_basis, k)


def deBoor_basis_unsafe(k: int, knots: Array, x):
    """
    Will silently return wrong values if knots[k] > x or knots[-k-1] <= x
    """

    i = jnp.searchsorted(knots, x, side='right')
    t = dynamic_slice(knots, (i - k,), (2 * k,))

    f = lambda c, a: (None, dynamic_slice(t, (a,), (k,)))

    _, Order = scan(f, None, jnp.arange(1, k + 1))

    # Division by 0 should result in 0
    # TODO find relevant section in the deBoor book
    base = Order - t[:k]
    pred = base == 0
    base = jnp.where(pred, 1.0, base)

    # This version should be more memory efficient? But results in nan gradients for duplicate knots...
    # A = jnp.nan_to_num((x - t[:k]) / (Order - t[:k]), False, 0.0, 0.0, 0.0)
    # B = jnp.nan_to_num((Order - x) / (Order - t[:k]), False, 0.0, 0.0, 0.0)

    A = jnp.where(pred, 0.0, (x - t[:k]) / base)

    B = jnp.where(pred, 0.0, (Order - x) / base)

    def do(c, a):
        c = c.at[k + 1].set(0.0)
        # A = a[0]
        c = c.at[k + 2 : 2 * k + 2].set(c[1 : k + 1] * a[0])

        # B = a[1]
        c = c.at[k + 1 : 2 * k + 1].add(c[1 : k + 1] * a[1])

        c = c.at[: k + 1].set(c[k + 1 : 2 * k + 2])

        return (c, None)

    init = jnp.zeros(2 * (k + 1), dtype=knots.dtype)
    init = init.at[k].set(1.0)

    out, _ = scan(do, init, jnp.stack([A, B], axis=1))

    return (i - k - 1, out[: k + 1])


@partial(jit, static_argnames=['k'])
def deBoor_factor_unsafe(k: int, knots: Array, x):
    i, r = deBoor_basis_unsafe(k, knots, x)
    max = len(knots) - k - 1
    res = jnp.zeros(max)
    res = dynamic_update_slice(res, r, (i,))
    return res


def deBoor_basis_reference(knots: onp.ndarray, deg: int, x):
    # c = 1.0 * ((knots <= x)[:-1] & (knots > x)[1:])[: deg + 1]

    def f(x, i, k):
        if k == 0:
            if knots[i] <= x and x < knots[i + 1]:
                return 1
            else:
                return 0
        a = (x - knots[i]) / (knots[k + i] - knots[i])
        b = (knots[i + k + 1] - x) / (knots[i + k + 1] - knots[i + 1])
        return a * f(x, i, k - 1) + b * f(x, i + 1, k - 1)

    return f(x, 0, deg)

def symbolic_basis_unsafe(knots: Array, x):
    """
    Will silently return wrong values if knots[3] > x or knots[-4] <= x
    """
    k = 3

    i = jnp.searchsorted(knots, x, side='right')
    t = dynamic_slice(knots, (i - k,), (2 * k,))
    out = jnp.zeros(k+1)

    t32 = t[3] - t[2]
    B11 = (t[3] - x) / t32
    B21 = (x - t[2]) / t32

    t31 = t[3] - t[1]
    t42 = t[4] - t[2]
    B02 = ((t[3] - x) / t31) * B11
    B12 = ((x - t[1]) / t31) * B11 + ((t[4] - x) / t42) * B21
    B22 = ((x - t[2]) / (t[4] - t[2])) * B21

    t30 = t[3] - t[0]
    t41 = t[4] - t[1]
    t52 = t[5] - t[2]
    out = out.at[0].set(((t[3] - x) / t30) * B02)
    out = out.at[1].set(((x - t[0]) / t30) * B02 + ((t[4] - x) / t41) * B12)
    out = out.at[2].set(((x - t[1]) / t41) * B12 + ((t[5] - x) / t52) * B22)
    out = out.at[3].set(((x - t[2]) / t52) * B22)

    return (i - k - 1, out)


@jit
def symbolic_factor_unsafe(knots: Array, x):
    k = 3
    i, r = symbolic_basis_unsafe(knots, x)
    max = len(knots) - k - 1
    res = jnp.zeros(max)
    res = dynamic_update_slice(res, r, (i,))
    return res