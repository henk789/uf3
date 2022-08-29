from uf3.jax import jax_splines as jsp

from functools import partial

from typing import Any, Callable, List

import jax.numpy as jnp
from jax import vmap, grad, jit
from jax_md import space, partition, util

# Types

f32 = util.f32
f64 = util.f64
Array = util.Array

Box = space.Box
DisplacementFn = space.DisplacementFn
DisplacementOrMetricFn = space.DisplacementOrMetricFn

NeighborFn = partition.NeighborFn
NeighborList = partition.NeighborList
NeighborListFormat = partition.NeighborListFormat


def uf2_pair(
    displacement: DisplacementFn,
    #  species=None,
    **kwargs
) -> Callable[[Array], Array]:
    """
    2-body pair potential.
    coefficients, knots, and cutoff need to be supplied to uf2_pair or the returned compute function.
    More user friendly way is in the works.

    Species parameter not yet supported.

    For usage see the example notebooks here or in the JAX MD package

    Better docstrings comming!
    """

    def compute_fn(R, **dynamic_kwargs):
        _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)

        d = partial(displacement, **_kwargs)
        dR = space.map_product(d)(R, R)
        dr = space.distance(dR)

        two_body_fn = partial(uf2_mapped, **_kwargs)

        two_body_term = util.high_precision_sum(two_body_fn(dr)) / 2.0

        return two_body_term

    return compute_fn


def uf3_pair(
    displacement,
    # species = None,
    **kwargs
) -> Callable[[Array], Array]:
    """
    #TODO
    """

    def compute_fn(R, **dynamic_kwargs):
        _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)

        d = partial(displacement, **_kwargs)
        dR = space.map_product(d)(R, R)
        dr = space.distance(dR)

        two_body_fn = partial(uf2_mapped, **_kwargs)
        three_body_fn = partial(uf3_mapped, **_kwargs)

        two_body_term = util.high_precision_sum(two_body_fn(dr)) / 2.0
        three_body_term = util.high_precision_sum(three_body_fn(dR, dR)) / 2.0

        return two_body_term + three_body_term

    return compute_fn


def uf2_neighbor(
    displacement,
    box_size,
    species=None,
    coefficients=None,
    knots=None,
    cutoff=5.5,
    dr_threshold: float = 0.5,
    format: NeighborListFormat = partition.Dense,
    **kwargs
):
    """
    Supports only dense neighbor lists.

    2-body neighbor list potential.
    coefficients, knots need to be supplied to uf2_neighbor
    More user friendly way is in the works.

    For usage see the example notebooks here or in the JAX MD package

    if species are None then knots and coefficients shoud be Arrays

    if species is specified then knots and coefficients need to be dicts
    with the arrays for the interactions given as tuples (i,j) where i,j are ints for the species
    """

    r_cutoff = jnp.array(cutoff, jnp.float32)
    dr_threshold = jnp.float32(dr_threshold)

    neighbor_fn = partition.neighbor_list(
        displacement, box_size, r_cutoff, dr_threshold, format=format, **kwargs
    )

    if species is None:
        two_body_fn = partial(uf2_mapped, knots=knots)

        def energy_fn(R, neighbor, **dynamic_kwargs):

            _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)
            d = partial(displacement, **_kwargs)
            mask = partition.neighbor_list_mask(neighbor)

            if neighbor.format is partition.Dense:
                dR = space.map_neighbor(d)(R, R[neighbor.idx])
                dr = space.distance(dR)

                _coefficients = dynamic_kwargs.get("coefficients", coefficients)

                two_body_term = (
                    util.high_precision_sum(
                        two_body_fn(dr, coefficients=_coefficients) * mask
                    )
                    / 2.0
                )
            else:
                raise NotImplementedError(
                    "UF2 potential only implemented with Dense neighbor lists."
                )

            return two_body_term

    else:
        two_body_splines = {}
        for k, v in knots.items():
            two_body_splines[k] = partial(uf2_mapped, knots=v)

        max_species = len(species)
        species_enum = jnp.arange(max_species)

        if coefficients is None:
            coefficients = {}

        def energy_fn(R, neighbor, **dynamic_kwargs):
            _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)
            d = partial(displacement, **_kwargs)

            tmp = _kwargs.get("coefficients", {})
            _coefficients = util.merge_dicts(coefficients, tmp)

            if neighbor.format is partition.Dense:
                dR = space.map_neighbor(d)(R, R[neighbor.idx])
                dr = space.distance(dR)

                two_body_term = 0.0

                mask = neighbor.idx < max_species

                for k, s in two_body_splines.items():
                    i, j = k
                    normalization = 1.0
                    if i == j:
                        normalization = 2.0
                    idx = jnp.where(species == j, species_enum, max_species)
                    mask_j = jnp.isin(neighbor.idx, idx) * mask
                    mask_ij = (i == species)[:, None] * mask_j

                    fn = partial(s, coefficients=_coefficients[k])
                    two_body_term += (
                        util.high_precision_sum(fn(dr) * mask_ij) / normalization
                    )

            else:
                raise NotImplementedError(
                    "UF2 potential only implemented with Dense neighbor lists."
                )
            return two_body_term

    return neighbor_fn, energy_fn


def uf3_neighbor(
    displacement,
    box_size,
    species=None,
    coefficients=None,
    knots=None,
    cutoff=5.5,
    dr_threshold: float = 0.5,
    format: NeighborListFormat = partition.Dense,
    **kwargs
):
    """
    2-body neighbor list potential.
    coefficients, knots need to be supplied to uf2_neighbor or the returned compute function.
    More user friendly way is in the works.

    Species parameter not yet supported.

    For usage see the example notebooks here or in the JAX MD package

    Better docstrings comming!
    """

    r_cutoff = jnp.array(cutoff, jnp.float32)
    dr_threshold = jnp.float32(dr_threshold)

    neighbor_fn = partition.neighbor_list(
        displacement, box_size, r_cutoff, dr_threshold, format=format, **kwargs
    )

    if species is None:
        two_body_fn = partial(uf2_mapped, knots=knots[0])
        three_body_fn = partial(uf3_mapped, knots3=knots[1])

        def energy_fn(R, neighbor, **dynamic_kwargs):

            _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)

            _coefficients = _kwargs.get("coefficients", coefficients)

            d = partial(displacement, **_kwargs)
            mask = partition.neighbor_list_mask(neighbor)
            mask_ijk = mask[:, None, :] * mask[:, :, None]

            if neighbor.format is partition.Dense:
                dR = space.map_neighbor(d)(R, R[neighbor.idx])
                dr = space.distance(dR)

                two_body_term = (
                    util.high_precision_sum(
                        two_body_fn(dr, coefficients=_coefficients[0]) * mask
                    )
                    / 2.0
                )

                three_body_term = (
                    util.high_precision_sum(
                        three_body_fn(dR, dR, coefficients3=_coefficients[1]) * mask_ijk
                    )
                ) / 2.0
            else:
                raise NotImplementedError(
                    "UF3 potential only implemented with Dense neighbor lists."
                )
            # print(three_body_term)
            return two_body_term + three_body_term

    else:
        two_body_splines = {}
        for k, v in knots[0].items():
            two_body_splines[k] = partial(uf2_mapped, knots=v)
        three_body_splines = {}
        for k, v in knots[1].items():
            three_body_splines[k] = partial(uf3_mapped, knots3=v)

        max_species = len(species)
        species_enum = jnp.arange(max_species)

        if coefficients is None:
            coefficients = [{},{}]

        def energy_fn(R, neighbor, **dynamic_kwargs):
            _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)
            d = partial(displacement, **_kwargs)

            tmp = _kwargs.get("coefficients", [{},{}])
            # _coefficients = [0,0]
            _coefficients2 = util.merge_dicts(coefficients[0], tmp[0])
            _coefficients3 = util.merge_dicts(coefficients[1], tmp[1])

            if neighbor.format is partition.Dense:
                dR = space.map_neighbor(d)(R, R[neighbor.idx])
                dr = space.distance(dR)

                two_body_term = 0.0

                mask = neighbor.idx < len(neighbor.idx)

                for k, s in two_body_splines.items():
                    i, j = k
                    normalization = 1.0
                    if i == j:
                        normalization = 2.0
                    idx = jnp.where(species == j, species_enum, max_species)
                    mask_j = jnp.isin(neighbor.idx, idx) * mask
                    mask_ij = (i == species)[:, None] * mask_j

                    fn = partial(s, coefficients=_coefficients2[k])
                    two_body_term += (
                        util.high_precision_sum(fn(dr) * mask_ij) / normalization
                    )

                three_body_term = 0.0

                for key, s in three_body_splines.items():
                    i, j, k = key
                    normalization = 1.0
                    if j == k:
                        normalization = 2.0

                    imask = species == i
                    idxj = jnp.where(species == j, species_enum, max_species)
                    mask_j = jnp.isin(neighbor.idx, idxj) * mask
                    idxk = jnp.where(species == k, species_enum, max_species)
                    mask_k = jnp.isin(neighbor.idx, idxk) * mask
                    mask_ijk = (
                        imask[:, None, None] * mask_j[:, None, :] * mask_k[:, :, None]
                    )

                    fn = partial(s, coefficients3=_coefficients3[key])
                    three_body_term += (
                        util.high_precision_sum(fn(dR, dR) * mask_ijk) / normalization
                    )

            else:
                raise NotImplementedError(
                    "UF3 potential only implemented with Dense neighbor lists."
                )
            return two_body_term + three_body_term

    return neighbor_fn, energy_fn


@jit
def uf2_interaction(
    dr: Array,
    coefficients: jnp.ndarray = None,
    knots: jnp.ndarray = None,
) -> Array:
    k = 3
    mint = knots[k]
    maxt = knots[-k-1]
    within_cutoff = (dr > 0) & (dr >= mint) & (dr < maxt)
    dr = jnp.where(within_cutoff, dr, 0.0)
    spline = vmap(partial(jsp.bspline_factors, knots))
    return jnp.where(
        within_cutoff, jnp.sum(coefficients * spline(dr), 1), 0.0
    )  # TODO check performance vs einsum


@jit
def uf2_mapped(
    dr: Array,
    coefficients: Array = None,
    knots: Array = None,
    **kwargs
) -> Array:
    fn = partial(uf2_interaction, coefficients=coefficients, knots=knots)
    return vmap(fn)(dr)


@jit
def uf3_interaction(
    dR12: Array,
    dR13: Array,
    coefficients: jnp.ndarray = None,
    knots: List[jnp.ndarray] = None,
) -> Array:
    k = 3
    cutoff = knots[0][-k-1]
    # sane default value is now standard, due to featurization simplification in original uf3
    angular_cutoff = cutoff * 2

    min1 = knots[0][k]
    min2 = knots[1][k]
    min3 = knots[2][k]
    max1 = knots[0][-k-1]
    max2 = knots[1][-k-1]
    max3 = knots[2][-k-1]


    dR23 = dR13 - dR12
    dr12 = space.distance(dR12)
    dr13 = space.distance(dR13)
    dr23 = space.distance(dR23)
    dr12 = jnp.where(dr12 < cutoff, dr12, 0)
    dr13 = jnp.where(dr13 < cutoff, dr13, 0)
    dr23 = jnp.where(dr23 < angular_cutoff, dr23, 0)

    # 3-D Spline
    # k = 3
    spline1 = partial(jsp.bspline_factors, knots[0])
    spline2 = partial(jsp.bspline_factors, knots[1])
    spline3 = partial(jsp.bspline_factors, knots[2])

    within_cutoff = (
        (dr12 >= min1)
        & (dr13 >= min2)
        & (dr23 >= min3)
        & (dr12 < max1)
        & (dr13 < max2)
        & (dr23 < max3)
        & (dr12 > 0.0)
        & (dr13 > 0.0)
        & (dr23 > 0.0)
    )
    return jnp.where(
        within_cutoff,
        jnp.einsum(
            coefficients,
            [1, 2, 3],
            spline1(dr12),
            [1],
            spline2(dr13),
            [2],
            spline3(dr23),
            [3],
        ),
        0,
    )


@jit
def uf3_mapped(dR12, dR13, coefficients3=None, knots3=None, **kwargs):
    fn = partial(
        uf3_interaction, coefficients=coefficients3, knots=knots3
    )
    return vmap(vmap(vmap(fn, (0, None)), (None, 0)))(dR12, dR13)
