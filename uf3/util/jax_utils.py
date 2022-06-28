import ase
import numpy as np
import jax.numpy as jnp

from argparse import ArgumentError
from typing import List, Tuple, Union
Array = jnp.ndarray

from uf3.data import geometry


def scale_atoms(atoms, cutoff):
    r_cut = cutoff * 2
    fact = geometry.get_supercell_factors(atoms.cell, r_cut)
    pos = atoms.positions
    new_pos = atoms.get_positions(wrap=True)
    an = atoms.get_atomic_numbers()
    new_an = atoms.get_atomic_numbers()
    for vec, fac in zip(atoms.cell, fact):
        for i in range(1, int(fac)):
            tmp = pos + (vec * i)
            new_pos = np.concatenate((new_pos, tmp))
            new_an = np.concatenate((new_an, an))
        pos = new_pos
        an = new_an

    new_cell = fact[:,None] * atoms.cell
    return ase.Atoms(an, positions=pos, cell=new_cell, pbc=atoms.pbc)


def check_inputs(
    knots: Union[Array, List[Array]],
    degrees: Union[int, Tuple[int]],
    coefficients: Array = None,
    padding=True
):
    '''
    #TODO add padding if unsafe
    '''
    if not isinstance(knots, List):
        knots = [knots]

    if isinstance(degrees, int):
        if len(degrees) == 1:
            degrees = (degrees,) * len(knots)
        elif len(knots) != len(degrees):
            raise ArgumentError(
                "There has to be one degree per knot sequence or only one degree that will be applied to all dimensions."
            )

    for d, k in zip(degrees, knots):
        if len(k) < 2 * d:
            raise ArgumentError(
                "There have to be atleast 2 * degree knots in each dimension."
            )

    if coefficients is not None:
        shape = coefficients.shape
        if len(shape) != len(degrees):
            raise ArgumentError(
                "The coefficient array has to have the same dimensions as the spline."
            )
        correct_shape = []
        for d, k in zip(degrees, knots):
            correct_shape.append(len(k) + d + 1)
        correct_shape = tuple(correct_shape)
        if correct_shape != shape:
            raise ArgumentError(
                "There have to be num_of_knots + degree + 1 coefficients in each dimension. The shape has to be"
                + correct_shape
            )
    
    #TODO check and rework
    padding = []
    knot = []
    for t, k in zip(knots, degrees):
        t = jnp.pad(t, (k, k), "edge")
        knot.append(t)
        padding.append((k, k))

    padding = tuple(padding)
    c = jnp.pad(coefficients, padding)

    return (coefficients, knots, degrees)