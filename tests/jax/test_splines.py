from bz2 import compress
import pytest

from uf3.jax.jax_splines import *
from uf3.util.random import random_spline
import jax.numpy as jnp
import numpy as onp

import ndsplines

from jax import vmap, jacrev

from jax.config import config

config.update("jax_enable_x64", True)


def test_ndSpline_unsafe():
    rng = onp.random.default_rng()
    seed = rng.integers(0, 999)
    print(f"Seed for energy test: {seed}")

    # 1-D
    c, k, x = random_spline((15,), sample=100, seed=seed)
    s = ndSpline_unsafe(k, (3,), c)
    sp = ndsplines.NDSpline(k, c, (3,))

    assert jnp.allclose(vmap(s)(*x), sp(jnp.stack(x, 1)))

    # 2-D
    c, k, x = random_spline((15, 15), sample=100, seed=seed)
    s = ndSpline_unsafe(k, (3, 3), c)
    sp = ndsplines.NDSpline(k, c, (3, 3))

    assert jnp.allclose(vmap(s)(*x), sp(jnp.stack(x, 1)))

    # 3-D
    c, k, x = random_spline((15, 15, 15), sample=100, seed=seed)
    s = ndSpline_unsafe(k, (3, 3, 3), c)
    sp = ndsplines.NDSpline(k, c, (3, 3, 3))

    assert jnp.allclose(vmap(s)(*x), sp(jnp.stack(x, 1)))

    # odd stuff
    c, k, x = random_spline((15, 13), degrees=(3, 5), sample=100, seed=seed)
    s = ndSpline_unsafe(k, (3, 5), c)
    sp = ndsplines.NDSpline(k, c, (3, 5))

    assert jnp.allclose(vmap(s)(*x), sp(jnp.stack(x, 1)))


def test_deBoor_backend():
    rng = onp.random.default_rng()
    seed = rng.integers(0, 999)
    print(f"Seed for energy test: {seed}")
    rng = onp.random.default_rng(seed)

    n_xs = 100
    k = 3

    deBoor_factor_unsafe = partial(
        bspline_factors, k=k, basis=BSplineBackend.DeBoor, safe=False
    )

    knots = rng.uniform(0.0, 11, 50)

    knots = onp.sort(knots)
    knots[0] = 0
    knots = onp.pad(knots, (3, 3), "edge")

    knots = jnp.asarray(knots)

    xs = rng.uniform(0.0, knots[-1], n_xs)

    res = vmap(deBoor_factor_unsafe, (None, 0))(knots, xs)

    assert jnp.allclose(jnp.sum(res, 1), jnp.ones(n_xs))

    assert len(res[res != 0.0]) == n_xs * (k + 1)

    a = deBoor_factor_unsafe(knots, knots[k])

    assert a[0] == 1.0

    b = deBoor_factor_unsafe(knots, knots[-k - 1] - 0.000000001)

    assert jnp.allclose(jnp.sum(b[-k - 1 :]), 1.0)

    c = vmap(deBoor_factor_unsafe, (None, 0))(knots, knots[k + 1 : -k - 1])

    assert len(c[c != 0.0]) == len(knots[k + 1 : -k - 1]) * k

    # no division-by-zero with maximal duplicate knots
    knots = onp.ones_like(knots)
    knots[20:] = 2.0
    out = deBoor_factor_unsafe(knots, 1.5)
    assert not jnp.isnan(out).any()
    assert not jnp.isinf(out).any()


def test_symbolic_backend():
    rng = onp.random.default_rng()
    seed = rng.integers(0, 999)
    print(f"Seed for energy test: {seed}")
    rng = onp.random.default_rng(seed)

    n_xs = 100
    k = 3

    symbolic_factor_unsafe = partial(
        bspline_factors, k=k, basis=BSplineBackend.Symbolic, safe=False
    )

    knots = rng.uniform(0.0, 11, 50)

    knots = onp.sort(knots)
    knots[0] = 0
    knots = onp.pad(knots, (3, 3), "edge")

    knots = jnp.asarray(knots)

    xs = rng.uniform(0.0, knots[-1], n_xs)

    res = vmap(symbolic_factor_unsafe, (None, 0))(knots, xs)

    assert jnp.allclose(jnp.sum(res, 1), jnp.ones(n_xs))

    assert len(res[res != 0.0]) == n_xs * 4

    a = symbolic_factor_unsafe(knots, knots[k])

    assert a[0] == 1.0

    b = symbolic_factor_unsafe(knots, knots[-k - 1] - 0.000000001)

    assert jnp.allclose(jnp.sum(b[-k - 1 :]), 1.0)

    c = vmap(symbolic_factor_unsafe, (None, 0))(knots, knots[k + 1 : -k - 1])

    assert len(c[c != 0.0]) == len(knots[k + 1 : -k - 1]) * 3

    # no division-by-zero with maximal duplicate knots
    knots = onp.ones_like(knots)
    knots[20:] = 2.0
    out = symbolic_factor_unsafe(knots, 1.5)
    assert not jnp.isnan(out).any()
    assert not jnp.isinf(out).any()


def test_backends():
    rng = onp.random.default_rng()
    seed = rng.integers(0, 999)
    print(f"Seed for energy test: {seed}")
    rng = onp.random.default_rng(seed)

    n_xs = 100
    k = 3

    symbolic_factor_unsafe = partial(
        bspline_factors, k=k, basis=BSplineBackend.Symbolic, safe=False, compress=True
    )

    deBoor_factor_unsafe = partial(
        bspline_factors, k=k, basis=BSplineBackend.DeBoor, safe=False, compress=True
    )

    knots = rng.uniform(0.0, 11, 50)

    knots = onp.sort(knots)
    knots[0] = 0
    knots = onp.pad(knots, (3, 3), "edge")

    knots = jnp.asarray(knots)

    xs = rng.uniform(0.0, knots[-1], n_xs)

    f1 = lambda x: symbolic_factor_unsafe(knots, x)
    f2 = lambda x: deBoor_factor_unsafe(knots, x)

    res1, _ = vmap(f1)(xs)
    res2, _ = vmap(f1)(xs)

    assert jnp.allclose(res1, res2)

    d1, _ = vmap(jacrev(f1, has_aux=True))(xs)
    d2, _ = vmap(jacrev(f2, has_aux=True))(xs)

    assert jnp.allclose(d1, d2)
