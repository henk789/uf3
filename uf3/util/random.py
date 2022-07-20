from argparse import ArgumentError
import numpy as onp
import jax.numpy as jnp

from ase import Atoms

"""Utility functions to create random splines, potentials and atomic systems"""


def random_spline(resolution, min=0, max=10, sample=None, degrees=None, seed=None):
    """
    Args:
        resolution: A tuple with the number of valid knot intervals for each dimension
        min: A scalar or tuple with the minimum valid input for all or per dimension
        max: A scalar or tuple with the maximum valid input for all or per dimension
        sample: Set to a scalar to generate N sample inputs for the spline in the valid range
        degrees: A tuple with the spline degrees for each dimension - defaults to cubic splines
        seed: A fixed seed for numpy.random.default_rng
    Output:
        (coefficients, knots, sample): A tuple with a coefficient matrix and a list of knots for each dimension.
            sample is none or an nd-array with sample inputs for the spline function.
            Suitable for use with uf3.jax.jax_splines._ndSpline_unsafe
    """
    dim = len(resolution)

    if seed is None:
        rng = onp.random.default_rng()
        seed = rng.integers(0, 999)
    rng = onp.random.default_rng(seed)

    if degrees is None:
        degrees = (3,) * dim

    if onp.isscalar(min):
        min = (min,) * dim
    if onp.isscalar(max):
        max = (max,) * dim

    if len(degrees) != dim:
        raise ArgumentError(
            "Degrees and resolution has to be specified for each dimension."
        )
    if dim != len(min) != len(max):
        raise ArgumentError(
            "min and max have to be scalar or have the same dimension as resolution."
        )

    knots = []
    coeff_dim = ()
    coeff_pad = ()
    for r, d, mi, ma in zip(resolution, degrees, min, max):
        k = rng.uniform(mi, ma, r + 1)
        k = onp.sort(k)
        k[0] = mi
        k[-1] = ma
        coeff_dim += (len(k) - d - 1,)
        coeff_pad += ((d, d),)
        k = onp.pad(k, (d, d), "edge")
        knots.append(jnp.asarray(k))

    coefficients = rng.standard_normal(coeff_dim) * 10.0
    coefficients = onp.pad(coefficients, coeff_pad)
    coefficients = jnp.asarray(coefficients)

    if sample is not None:
        maxi = onp.asarray(max) - 1e-10
        xs = rng.uniform(min, maxi, (sample, dim))
        sample = jnp.asarray(xs)

    return (coefficients, knots, sample)


def random_system(cell_size, size, n_trajectories, atom='W', seed=123):


    if seed is None:
        rng = onp.random.default_rng()
        seed = rng.integers(0, 999)
    rng = onp.random.default_rng(seed)

    positions = onp.zeros((n_trajectories, size, 3))
    atoms = []

    for i in range(n_trajectories):
        positions[i,:,:] = rng.uniform(0.0, cell_size, size=(size,3))
        atoms.append(Atoms(atom+str(size), positions=positions[i,:,:], cell=[cell_size]*3, pbc=True))
    
    return positions, atoms
