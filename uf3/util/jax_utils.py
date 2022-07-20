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

    new_cell = fact[:, None] * atoms.cell
    return ase.Atoms(an, positions=pos, cell=new_cell, pbc=atoms.pbc)


def check_inputs(
    knots: Union[Array, List[Array]],
    degrees: Union[int, Tuple[int]],
    coefficients: Array = None,
    padding=True,
):

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

    if padding:
        coefficients, knots = add_padding(coefficients, knots, degrees)

    return (coefficients, knots, degrees)


def add_padding(coefficients, knots, degrees):
    """
    Adds padding where necessary for safe use with jax_splines on the whole range of the original knots.
    """
    # TODO only add padding if necessary instead of always
    padding = []
    knot = []
    for t, k in zip(knots, degrees):
        t = jnp.pad(t, (k, k), "edge")
        knot.append(t)
        padding.append((k, k))

    padding = tuple(padding)
    c = jnp.pad(coefficients, padding)

    return c, knot


def from_ase_calculator(calculator):
    """
    Returns a JAX UF potential equivalent to the provided calculator.
    """
    pairs = calculator.pair_potentials
    triples = {}
    if calculator.degree > 2:
        triples = calculator.trio_potentials

    elements = calculator.chemical_system.element_list
    
    pair_knots = {}
    triplet_knots = {}
    pair_coefficients = {}
    triplet_coefficients = {}

    for pair, spline in pairs.items():
        i = elements.index(pair[0])
        j = elements.index(pair[1])
        if i > j:
            i, j = j, i

        knots = jnp.asarray(spline.knots[0])
        coeff = jnp.asarray(spline.coefficients[:,0])

        pair_knots[(i,j)] = knots
        pair_coefficients[(i,j)] = coeff

    for triple, spline in triples.items():
        i = elements.index(triple[0])
        j = elements.index(triple[1])
        k = elements.index(triple[2])
        if k > j:
            j,k = k,j

        knots = [jnp.asarray(i) for i in spline.knots]
        coeff = jnp.asarray(spline.coefficients[:,:,:,0])

        triplet_knots[(i,j,k)] = knots
        triplet_coefficients[(i,j,k)] = coeff

    return ([pair_knots, triplet_knots], [pair_coefficients, triplet_coefficients])

    
