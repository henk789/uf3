import pytest

from jax import random
import numpy as onp
import jax.numpy as jnp

from jax.config import config

config.update("jax_enable_x64", True)

from jax_md import space
from uf3.jax.potentials import *
import ase
from uf3.data import composition
from uf3.representation import bspline
from uf3.regression import least_squares
from uf3.forcefield import calculator


def test_energy_2_body():
    N = 50
    dimension = 3
    box_size = 12.0
    # key = random.PRNGKey(123)

    # key, split = random.split(key)
    rng = onp.random.default_rng()
    seed = rng.integers(0, 999)
    print(f"Seed for energy test: {seed}")
    rng = onp.random.default_rng(seed)
    rng = onp.random.default_rng(848)

    R = rng.uniform(0.0, box_size, (N, dimension))
    R = jnp.asarray(R, dtype=jnp.float64)

    box = jnp.eye(dimension) * box_size
    displacement, shift = space.periodic_general(box, fractional_coordinates=False)

    species = onp.concatenate([onp.zeros(N // 2), onp.ones(N - (N // 2))])
    rng.shuffle(species)
    species = jnp.asarray(species, dtype=jnp.int16)

    knots = rng.uniform(0.0, 5.5, 17)

    knots = onp.sort(knots)
    knots[0] = 0
    knots = onp.pad(knots, (3, 3), "edge")

    coefficients = rng.standard_normal(len(knots) - 7) * 5
    coefficients = onp.pad(coefficients, (0, 3))

    coefficients = jnp.asarray(coefficients)
    knots = jnp.asarray(knots)

    pair = uf2_pair(
        displacement, coefficients=coefficients, knots=knots, cutoff=knots[-1]
    )
    nf, ef = uf2_neighbor(
        displacement, box_size, coefficients=coefficients, knots=knots, cutoff=knots[-1]
    )

    coeff_dict = {}
    coeff_dict[(0, 0)] = coefficients
    coeff_dict[(0, 1)] = coefficients
    coeff_dict[(1, 1)] = coefficients

    knot_dict = {}
    knot_dict[(0, 0)] = knots
    knot_dict[(0, 1)] = knots
    knot_dict[(1, 1)] = knots
    nfs, efs = uf2_neighbor(
        displacement,
        box_size,
        species=species,
        coefficients=coeff_dict,
        knots=knot_dict,
        cutoff=knots[-1],
    )

    energy_1 = pair(R)

    nbrs = nf.allocate(R)
    energy_2 = ef(R, neighbor=nbrs)

    nbrs = nfs.allocate(R)
    energy_3 = efs(R, neighbor=nbrs)

    assert jnp.allclose(energy_1, energy_2)
    assert jnp.allclose(energy_1, energy_3)

    pos = onp.asarray(R)
    cell = onp.asarray(box)
    pbc = onp.asarray([True,True,True])
    atoms = ase.Atoms('W'+str(len(R)), positions=pos, cell=cell, pbc=pbc)
    r_min_map = {('W', 'W'): float(knots[0]),
            }
    r_max_map = {('W', 'W'): float(knots[-1]),
            }
    chemical_system = composition.ChemicalSystem(element_list=['W'],
                                        degree=2)
    bspline_config = bspline.BSplineBasis(chemical_system,
                                        r_min_map=r_min_map,
                                        r_max_map=r_max_map)
    bspline_config.knots_map[('W','W')] = onp.asarray(knots)
    model = least_squares.WeightedLinearModel(bspline_config)
    model.coefficients = onp.insert(onp.asarray(coefficients), 0, 0)
    calc = calculator.UFCalculator(model)
    atoms.set_calculator(calc)

    energy_4 = atoms.get_potential_energy(force_consistent=True) / 2.0

    assert jnp.allclose(energy_1, energy_4)



def test_2_body_interaction_count():
    # test with all coefficients set to 1
    # -> energy should be the same number as number of all pairs
    N = 50
    dimension = 3
    box_size = 10.0
    # key = random.PRNGKey(123)

    # key, split = random.split(key)
    rng = onp.random.default_rng(216)
    R = rng.uniform(0.0, box_size, (N, dimension))
    R = jnp.asarray(R, dtype=jnp.float64)

    box = jnp.eye(dimension) * box_size
    displacement, shift = space.periodic_general(box, fractional_coordinates=False)

    species = onp.concatenate([onp.zeros(N // 2), onp.ones(N - (N // 2))])
    rng.shuffle(species)
    species = jnp.asarray(species, dtype=jnp.int16)

    knots = jnp.arange(0, box_size + 1, 0.25)
    knots = jnp.pad(knots, (3, 3), "edge")

    coefficients = jnp.ones(len(knots) - 4)

    pair = uf2_pair(displacement, coefficients=coefficients, knots=knots)
    nf, ef = uf2_neighbor(
        displacement, box_size, coefficients=coefficients, knots=knots, cutoff=box_size
    )

    n_pairs = N * (N - 1) / 2

    assert jnp.allclose(pair(R), n_pairs)

    nbrs = nf.allocate(R)
    assert jnp.allclose(ef(R, nbrs), n_pairs)

    knots3 = [knots, knots, knots]
    coefficients3 = jnp.zeros((len(coefficients),) * 3)
    pair3 = uf3_pair(
        displacement,
        coefficients=coefficients,
        knots=knots,
        coefficients3=coefficients3,
        knots3=knots3,
    )

    assert jnp.allclose(pair3(R), n_pairs)

    coeff_dict = {}
    coeff_dict[(0, 0)] = coefficients
    coeff_dict[(0, 1)] = coefficients
    coeff_dict[(1, 1)] = coefficients

    knot_dict = {}
    knot_dict[(0, 0)] = knots
    knot_dict[(0, 1)] = knots
    knot_dict[(1, 1)] = knots
    nfs, efs = uf2_neighbor(
        displacement,
        box_size,
        species=species,
        coefficients=coeff_dict,
        knots=knot_dict,
        cutoff=box_size,
    )

    nbrs = nfs.allocate(R)
    assert jnp.allclose(efs(R, nbrs), n_pairs)

    c3 = {}
    c3[(0, 0, 0)] = coefficients3
    c3[(0, 0, 1)] = coefficients3
    c3[(0, 1, 1)] = coefficients3
    c3[(1, 0, 0)] = coefficients3
    c3[(1, 0, 1)] = coefficients3
    c3[(1, 1, 1)] = coefficients3

    k3 = {}
    k3[(0, 0, 0)] = knots3
    k3[(0, 0, 1)] = knots3
    k3[(0, 1, 1)] = knots3
    k3[(1, 0, 0)] = knots3
    k3[(1, 0, 1)] = knots3
    k3[(1, 1, 1)] = knots3

    nf, ef = uf3_neighbor(
        displacement,
        box_size,
        species=species,
        knots2=knot_dict,
        coefficients2=coeff_dict,
        knots3=k3,
        coefficients3=c3,
        cutoff=box_size,
    )
    nbrs = nf.allocate(R)

    assert jnp.allclose(ef(R, nbrs), n_pairs)


def test_3_body_interaction_count():
    # test with all coefficients set to 1
    # -> energy should be the same number as number of all triangles
    N = 50
    dimension = 3
    box_size = 10.0
    # key = random.PRNGKey(123)

    # key, split = random.split(key)
    rng = onp.random.default_rng(216)
    R = rng.uniform(0.0, box_size, (N, dimension))
    R = jnp.asarray(R, dtype=jnp.float64)

    box = jnp.eye(dimension) * box_size
    displacement, shift = space.periodic_general(box, fractional_coordinates=False)

    species = onp.concatenate([onp.zeros(N // 2), onp.ones(N - (N // 2))])
    rng.shuffle(species)
    species = jnp.asarray(species, dtype=jnp.int16)

    knots = jnp.arange(
        0, box_size * 2, 0.25
    )  # for triangles one side might need to span the entire cube diagonally
    knots = jnp.pad(knots, (3, 3), "edge")

    coefficients = jnp.zeros(len(knots) - 4)

    n_triplets = (N * (N - 1) * (N - 2)) / 2

    knots3 = [knots, knots, knots]
    coefficients3 = jnp.ones((len(coefficients),) * 3)
    pair3 = uf3_pair(
        displacement,
        coefficients=coefficients,
        knots=knots,
        coefficients3=coefficients3,
        knots3=knots3,
    )

    assert jnp.allclose(pair3(R), n_triplets)

    nf, ef = uf3_neighbor(
        displacement,
        box_size,
        knots2=knots,
        coefficients2=coefficients,
        knots3=knots3,
        coefficients3=coefficients3,
        cutoff=box_size,
    )
    nbrs = nf.allocate(R)

    assert jnp.allclose(ef(R, nbrs), n_triplets)

    coeff_dict = {}
    coeff_dict[(0, 0)] = coefficients
    coeff_dict[(0, 1)] = coefficients
    coeff_dict[(1, 1)] = coefficients

    knot_dict = {}
    knot_dict[(0, 0)] = knots
    knot_dict[(0, 1)] = knots
    knot_dict[(1, 1)] = knots

    c3 = {}
    c3[(0, 0, 0)] = coefficients3
    c3[(0, 0, 1)] = coefficients3
    c3[(0, 1, 1)] = coefficients3
    c3[(1, 0, 0)] = coefficients3
    c3[(1, 0, 1)] = coefficients3
    c3[(1, 1, 1)] = coefficients3

    k3 = {}
    k3[(0, 0, 0)] = knots3
    k3[(0, 0, 1)] = knots3
    k3[(0, 1, 1)] = knots3
    k3[(1, 0, 0)] = knots3
    k3[(1, 0, 1)] = knots3
    k3[(1, 1, 1)] = knots3

    nfs, efs = uf3_neighbor(
        displacement,
        box_size,
        species=species,
        knots2=knot_dict,
        coefficients2=coeff_dict,
        knots3=k3,
        coefficients3=c3,
        cutoff=box_size,
    )
    nbrss = nf.allocate(R)

    assert jnp.allclose(efs(R, nbrss), n_triplets)

