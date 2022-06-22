import pytest

from uf3.jax.jax_splines import *
import jax.numpy as jnp
import numpy as np
import ndsplines
from numpy.testing import assert_allclose
from utils import make_random_spline

from jax.config import config
config.update("jax_enable_x64", True)

# def test_basic():
#     t = np.arange(15.0)
#     c = np.ones(11)
#     k = 3
#     x = np.asarray([3.2, 5.7,11.8])
#     f = deBoor_getBasis(k)

#     spline = ndsplines.NDSpline([t], c, k)
#     jspline = vmap(f, (None, 0))(jnp.asarray(t), jnp.asarray(x))

#     assert_allclose(spline(x), jspline)

# deBoor feature has to sum up to 1.0

def test_deBoor_factor():
    rng = onp.random.default_rng()
    seed = rng.integers(0,999)
    print(f"Seed for energy test: {seed}")
    rng = onp.random.default_rng(seed)

    n_xs = 100
    k=3

    knots = rng.uniform(0.0, 11, 50)

    knots = onp.sort(knots)
    knots[0] = 0
    knots = onp.pad(knots, (3,3), 'edge')

    knots = jnp.asarray(knots)

    xs = rng.uniform(0.0, knots[-1], n_xs)

    res = vmap(deBoor_factor_unsafe, (None,None,0))(3, knots, xs)

    assert jnp.allclose(jnp.sum(res, 1), jnp.ones(n_xs))

    assert len(res[res != 0.0]) == n_xs * 4

    a = deBoor_factor_unsafe(3, knots, knots[k])

    assert a[0] == 1.0

    b = deBoor_factor_unsafe(3, knots, knots[-k-1]-0.000000001)

    assert jnp.allclose(jnp.sum(b[-k-1:]), 1.0)

    c = vmap(deBoor_factor_unsafe, (None,None,0))(3, knots, knots[k+1:-k-1])

    assert len(c[c != 0.0]) == len(knots[k+1:-k-1]) * 3


