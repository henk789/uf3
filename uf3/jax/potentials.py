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


def uf23_pair(
    displacement,
    # species = None,
    **kwargs
) -> Callable[[Array], Array]:
    """
    #TODO test
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
    # species=None,
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

    _two_body_fn = partial(uf2_mapped, cutoff=cutoff)

    neighbor_fn = partition.neighbor_list(
        displacement, box_size, r_cutoff, dr_threshold, format=format, **kwargs
    )

    def energy_fn(R, neighbor, **dynamic_kwargs):

        _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)
        d = partial(displacement, **_kwargs)
        mask = partition.neighbor_list_mask(neighbor)

        if neighbor.format is partition.Dense:
            dR = space.map_neighbor(d)(R, R[neighbor.idx])
            dr = space.distance(dR)

            two_body_fn = partial(_two_body_fn, **_kwargs)

            first_term = util.high_precision_sum(two_body_fn(dr) * mask)
        else:
            raise NotImplementedError(
                "UF2 potential only implemented with Dense neighbor lists."
            )

        return first_term

    return neighbor_fn, energy_fn


def uf23_neighbor(
    displacement,
    box_size,
    # species=None,
    cutoff=5.5,
    dr_threshold: float = 0.5,
    format: NeighborListFormat = partition.Dense,
    **kwargs
):
    """
    #TODO test
    """

    r_cutoff = jnp.array(cutoff, jnp.float32)
    dr_threshold = jnp.float32(dr_threshold)

    _two_body_fn = partial(uf2_mapped, cutoff=cutoff)
    _three_body_fn = partial(uf3_mapped, cutoff3=cutoff)

    neighbor_fn = partition.neighbor_list(
        displacement, box_size, r_cutoff, dr_threshold, format=format, **kwargs
    )

    def energy_fn(R, neighbor, **dynamic_kwargs):

        _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)
        d = partial(displacement, **_kwargs)
        mask = partition.neighbor_list_mask(neighbor)

        if neighbor.format is partition.Dense:
            dR = space.map_neighbor(d)(R, R[neighbor.idx])
            dr = space.distance(dR)

            # maybe need to check if cutoff for 3-body is resupplied? -> need to change config parameter handling anyway
            two_body_fn = partial(_two_body_fn, **_kwargs)
            three_body_fn = partial(_three_body_fn, **_kwargs)

            two_body_term = util.high_precision_sum(two_body_fn(dr) * mask)
            mask_ijk = mask[:, None, :] * mask[:, :, None] #TODO how does the mask work? And does it with uf3?
            three_body_term = util.high_precision_sum(three_body_fn(dR, dR) * mask_ijk)
        else:
            raise NotImplementedError(
                "UF23 potential only implemented with Dense neighbor lists."
            )

        return two_body_term + three_body_term

    return neighbor_fn, energy_fn


def get_stress_fn(energy_fn, box):
    """
    Stress transformation adapted from
    https://github.com/sirmarcel/asax/blob/c4e2ca89880b154b026d5b645d58e9d6163429a1/asax/jax_utils.py#L17
    """

    transform_box_fn = lambda deformation: space.transform(
        jnp.eye(deformation.shape[0]) + (deformation + deformation.T) * 0.5, box
    )

    strained_energy_fn = lambda R, deformation, *args, **kwargs: energy_fn(
        R, *args, **kwargs, box=transform_box_fn(deformation)
    )

    total_strained_energy_fn = lambda R, deformation, *args, **kwargs: jnp.sum(
        strained_energy_fn(R, deformation, *args, **kwargs)
    )

    box_volume = jnp.linalg.det(box)
    stress_fn = (
        lambda R, deformation, *args, **kwargs: grad(
            total_strained_energy_fn, argnums=1
        )(R, deformation, *args, **kwargs)
        / box_volume
    )

    box_volume = jnp.linalg.det(box)
    deformation = jnp.zeros_like(box)

    return lambda R: stress_fn(R, deformation)


def uf2_interaction(
    dr: Array,
    coefficients: jnp.ndarray = None,
    knots: jnp.ndarray = None,
    cutoff: float = 5.5,
) -> Array:
    k = 3
    mint = knots[k]
    maxt = knots[-k]
    # TODO lower cut_off might have to be modified or knots and coefficients have to be corespondingly set
    within_cutoff = (dr > 0) & (dr < cutoff) & (dr >= mint) & (dr < maxt)
    dr = jnp.where(within_cutoff, dr, 0.0)
    spline = jit(vmap(partial(jsp.deBoor_factor_unsafe, k, knots)))
    return jnp.where(
        within_cutoff, jnp.sum(coefficients * spline(dr), 1), 0.0
    )  # TODO check performance vs einsum


def uf2_mapped(
    dr: Array,
    coefficients: Array = None,
    knots: Array = None,
    cutoff: float = 5.5,
    **kwargs
) -> Array:
    fn = partial(uf2_interaction, coefficients=coefficients, knots=knots, cutoff=5.5)
    return vmap(fn)(dr)


def uf3_interaction(
    dR12: Array,
    dR13: Array,
    coefficients: jnp.ndarray = None,
    knots: List[jnp.ndarray] = None,
    cutoff: float = 3.0,
) -> Array:
    # sane default value is now standard, due to featurization simplification in original uf3
    angular_cutoff = cutoff * 2

    k = 3
    min1 = knots[0][k]
    min2 = knots[1][k]
    min3 = knots[2][k]

    dR23 = dR13 - dR12
    dr12 = space.distance(dR12)
    dr13 = space.distance(dR13)
    dr23 = space.distance(dR23)
    dr12 = jnp.where(dr12 < cutoff, dr12, 0)
    dr13 = jnp.where(dr13 < cutoff, dr13, 0)
    dr23 = jnp.where(dr23 < angular_cutoff, dr23, 0)

    # 3-D Spline
    k = 3
    spline1 = jit(partial(jsp.deBoor_factor_unsafe, k, knots[0]))
    spline2 = jit(partial(jsp.deBoor_factor_unsafe, k, knots[1]))
    spline3 = jit(partial(jsp.deBoor_factor_unsafe, k, knots[2]))

    within_cutoff = (dr12 > min1) & (dr13 > min2) & (dr23 > min3)
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


def uf3_mapped(dR12, dR13, coefficients3=None, knots3=None, cutoff3=3.5, **kwargs):
    fn = partial(uf3_interaction, coefficients=coefficients3, knots=knots3, cutoff=cutoff3)
    return vmap(vmap(vmap(fn, (0, None)), (None, 0)))(dR12, dR13)
