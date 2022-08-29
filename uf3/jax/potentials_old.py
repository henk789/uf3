from uf3.jax import jax_splines as jsp

from functools import partial

from typing import Callable, List, Union, Dict, Tuple

import jax.numpy as jnp
from jax import vmap, jit
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
    knots,
    coefficients=None,
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
    two_body_fn = uf2_mapped(knots)

    def compute_fn(R, coefficients=coefficients, **dynamic_kwargs):
        _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)

        d = partial(displacement, **_kwargs)
        dR = space.map_product(d)(R, R)
        dr = space.distance(dR)

        two_body_term = util.high_precision_sum(two_body_fn(dr, coefficients)) / 2.0

        return two_body_term

    return compute_fn


def uf3_pair(
    displacement,
    knots,
    coefficients=None,
    # species = None,
    **kwargs
) -> Callable[[Array], Array]:
    """
    #TODO
    """
    two_body_fn = uf2_mapped(knots[0])
    three_body_fn = uf3_mapped(knots[1])

    def compute_fn(R, coefficients=coefficients, **dynamic_kwargs):
        _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)

        d = partial(displacement, **_kwargs)
        dR = space.map_product(d)(R, R)
        dr = space.distance(dR)

        two_body_term = util.high_precision_sum(two_body_fn(dr, coefficients[0])) / 2.0
        three_body_term = (
            util.high_precision_sum(three_body_fn(dR, dR, coefficients[1])) / 2.0
        )

        return two_body_term + three_body_term

    return compute_fn


def uf3_neighbor(
    displacement,
    box_size,
    knots: List[Union[Array, Dict[Tuple[int], Array]]],
    coefficients: List[Union[Array, Dict[Tuple[int], Array]]] = None,
    species=None,
    cutoff=5.5,
    dr_threshold: float = 0.5,
    format: NeighborListFormat = partition.Dense,
    featurization=False,
    force_features=False,
    **kwargs
):
    """
    UF3 potential with neighbor lists.
    Supports 1-, 2-, and 3-body terms.

    Args:
        displacement:
        box_size:
        coefficients: The coefficients for the UF3 potential.
            The list contains coefficients for [1-,2-,3-] body terms.
            If the list has the same length as the list for the knots, then no 1-body term is present.
            With species=None the elements of the list are the arrays with the coefficients.
            With species set, the list contains dictionaries with the coefficients per interaction,
            given by tuples:
                (i,j)   - for 2-body coefficients for interactions between species i and j
                (i,j,k) - for 3-bdoy coefficients for interactions between the central atom i and j, k
            Note: The optinal coefficients for 1-body terms are always an array.
        knots: The knots for the UF3 potential.
            The list contains knots for [2-,3-] body terms.
            If the list only contains one element it is assumed to be for 2-body terms only and
            the coefficent list will be assumed to be missing 3-body terms as well.
            The format for knots is identical to the coefficient list, either arrays or dicitonaries
            depending on if species are None.
        species: 
        cutoff: The cutoff for the neighbor list. No interactions beyond this cutoff are considered.
        dr_threshold: 
        format: The JAX-MD neighbor list format. Currently ONLY Dense is supported.
        featurization: Compute energy contributions per atom per coefficient instead of the total energy.
        force_features: Additionally compute force contributions.

        Returns:
            A JAX-MD neighbor list object  `neighbor_fn`.
            And a function `energy_fn` that takes in positions of atoms (shape (n,3)) and neighbors
            obtaint from `neighbor_fn.allocate`.
            `energy_fn` can take a keyword argument `coefficients` where the coefficients can be over-written.
            This argument has to have the same format as the coefficients used to generate this function.
            Dictionaries must contain all the same interactions.
            `energy_fn` can take a keyword argument `species` where the species can be set for new inputs.
    """

    return _uf3_neighbor(
        displacement,
        box_size,
        knots,
        coefficients,
        species,
        cutoff,
        dr_threshold,
        format,
        featurization,
        force_features,
        **kwargs
    )


def _uf3_neighbor(
    displacement,
    box_size,
    knots: List[Union[Array, Dict[Tuple[int], Array]]],
    coefficients: List[Union[Array, Dict[Tuple[int], Array]]] = None,
    species=None,
    cutoff=5.5,
    dr_threshold: float = 0.5,
    format: NeighborListFormat = partition.Dense,
    featurization=False,
    force_features=True,
    **kwargs
):
    r_cutoff = jnp.array(cutoff, jnp.float32)
    dr_threshold = jnp.float32(dr_threshold)

    if format is not partition.Dense:
        raise NotImplementedError(
            "UF 3-body potential only implemented with Dense neighbor lists."
        )

    neighbor_fn = partition.neighbor_list(
        displacement, box_size, r_cutoff, dr_threshold, format=format, **kwargs
    )

    if species is None:
        two_body_fn = uf2_mapped(
            knots[0], featurization=featurization, force_features=force_features
        )
        if len(knots) == 1:
            three_body_fn = None
        elif len(knots) == 2:
            three_body_fn = uf3_mapped(
                knots[1], featurization=featurization, force_features=force_features
            )
        else:
            raise ValueError(
                "Knots has to be a list of lists of knots for either, 2-body terms only, ",
                "or both  2 and 3 body terms.",
            )

        def energy_fn(R, neighbor, **dynamic_kwargs):

            _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)

            _coefficients = dynamic_kwargs.get("coefficients", coefficients)

            if len(_coefficients) == 1:
                one_body_term = 0.0
                coefficients_two_body = _coefficients[0]
            elif len(_coefficients) == 2 and three_body_fn is None:
                one_body_term = _coefficients[0] * len(R)
                coefficients_two_body = _coefficients[1]
            elif len(_coefficients) == 2:
                one_body_term = 0.0
                coefficients_two_body = _coefficients[0]
                coefficients_three_body = _coefficients[1]
            else:
                one_body_term = _coefficients[0] * len(R)
                coefficients_two_body = _coefficients[1]
                coefficients_three_body = _coefficients[2]

            d = partial(displacement, **_kwargs)
            mask = partition.neighbor_list_mask(neighbor)

            dR = space.map_neighbor(d)(R, R[neighbor.idx])
            dr = space.distance(dR)

            if not featurization:
                two_body_term = (
                    util.high_precision_sum(
                        two_body_fn(dr, coefficients=coefficients_two_body) * mask
                    )
                    / 2.0
                )

                if three_body_fn is not None:
                    mask_ijk = mask[:, None, :] * mask[:, :, None]
                    three_body_term = (
                        util.high_precision_sum(
                            three_body_fn(dR, dR, coefficients=coefficients_three_body)
                            * mask_ijk
                        )
                    ) / 2.0
                else:
                    three_body_term = 0.0

                return one_body_term + two_body_term + three_body_term
            else:
                two_body_term = util.high_precision_sum(
                    two_body_fn(dr, coefficients=coefficients_two_body)
                    * mask[:, :, None],
                    1,
                )

                if three_body_fn is not None:
                    mask_ijk = mask[:, None, :] * mask[:, :, None]
                    three_body_term = util.high_precision_sum(
                        three_body_fn(dR, dR, coefficients=coefficients_three_body)
                        * mask_ijk[:, :, :, None],
                        (1, 2),
                    )

                    return two_body_term / 2.0, three_body_term / 2.0

                return two_body_term / 2.0

    else:
        two_body_fns = {}
        for k, v in knots[0].items():
            two_body_fns[k] = uf2_mapped(
                v, featurization=featurization, force_features=force_features
            )
        if len(knots) == 1:
            three_body_fns = None
        elif len(knots) == 2:
            three_body_fns = {}
            for k, v in knots[1].items():
                three_body_fns[k] = uf3_mapped(
                    v, featurization=featurization, force_features=force_features
                )
        else:
            raise ValueError(
                "Knots has to be a list of dictionaries for either, 2-body terms only, ",
                "or both 2 and 3 body terms.",
            )

        max_species = len(species)
        species_enum = jnp.arange(max_species)

        def energy_fn(R, neighbor, **dynamic_kwargs):
            _kwargs = util.merge_dicts(kwargs, dynamic_kwargs)
            d = partial(displacement, **_kwargs)

            _coefficients = dynamic_kwargs.get("coefficients", coefficients)

            if len(_coefficients) == 1:
                one_body_term = 0.0
                coefficients_two_body = _coefficients[0]
            elif len(_coefficients) == 2 and three_body_fns is None:
                one_body_term = util.high_precision_sum(_coefficients[0][species])
                coefficients_two_body = _coefficients[1]
            elif len(_coefficients) == 2:
                one_body_term = 0.0
                coefficients_two_body = _coefficients[0]
                coefficients_three_body = _coefficients[1]
            else:
                one_body_term = util.high_precision_sum(_coefficients[0][species])
                coefficients_two_body = _coefficients[1]
                coefficients_three_body = _coefficients[2]

            dR = space.map_neighbor(d)(R, R[neighbor.idx])
            dr = space.distance(dR)

            two_body_term = 0.0
            three_body_term = 0.0
            if featurization:
                two_body_term = {}
                three_body_term = {}

            mask = neighbor.idx < len(neighbor.idx)

            for k, fn in two_body_fns.items():
                i, j = k
                normalization = 1.0
                if i == j:
                    normalization = 2.0
                idx = jnp.where(species == j, species_enum, max_species)
                mask_j = jnp.isin(neighbor.idx, idx) * mask
                mask_ij = (i == species)[:, None] * mask_j

                c = coefficients_two_body[k]

                if not featurization:
                    two_body_term += (
                        util.high_precision_sum(fn(dr, coefficients=c) * mask_ij)
                        / normalization
                    )
                else:
                    two_body_term[k] = util.high_precision_sum(
                        fn(dr, coefficients=c) * mask_ij[:, :, None], 1
                    )

            if three_body_fns is not None:
                for key, fn in three_body_fns.items():
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

                    c = coefficients_three_body[key]

                    if not featurization:
                        three_body_term += (
                            util.high_precision_sum(
                                fn(dR, dR, coefficients=c) * mask_ijk
                            )
                            / normalization
                        )
                    else:
                        three_body_term[k] = util.high_precision_sum(
                            fn(dR, dR, coefficients=c)
                            * mask_ijk[:, :, :, None, None, None],
                            (1, 2),
                        )

            if not featurization:
                return one_body_term + two_body_term + three_body_term
            else:
                return two_body_term, three_body_term

    return neighbor_fn, energy_fn


def uf2_mapped(knots, featurization=False, force_features=False):
    spline = jsp.ndSpline_unsafe(knots, (3,), featurization=featurization)

    def op(dr, coefficients=None):
        k = 3
        mint = knots[0][k]
        maxt = knots[0][-k - 1]
        within_cutoff = (dr >= mint) & (dr < maxt)
        dr = jnp.where(within_cutoff, dr, 0.0)
        s = partial(spline, coefficients=coefficients)
        if not featurization:
            return jnp.where(within_cutoff, s(dr), 0.0)
        else:
            if force_features:
                s = jsp.featurization_with_gradients(s, coefficients=coefficients)
                e, f = s(dr)
                e = jnp.where(within_cutoff, e, 0.0)
                f = jnp.where(within_cutoff, f, 0.0)
                return jnp.stack([e, f]).flatten()
            else:
                return jnp.where(within_cutoff, s(dr), 0.0)

    @jit
    def fn(dr, coefficients=None):
        f = partial(op, coefficients=coefficients)
        return vmap(vmap(f))(dr)

    return fn


def uf3_mapped(knots, featurization=False, force_features=False):
    three_body = jsp.ndSpline_unsafe(knots, (3, 3, 3), featurization=featurization)

    def op(dR12, dR13, coefficients=None):
        dR23 = dR13 - dR12
        dr12 = space.distance(dR12)
        dr13 = space.distance(dR13)
        dr23 = space.distance(dR23)

        k = 3
        min1 = knots[0][k]
        min2 = knots[1][k]
        min3 = knots[2][k]
        max1 = knots[0][-k - 1]
        max2 = knots[1][-k - 1]
        max3 = knots[2][-k - 1]

        within1 = (dr12 >= min1) & (dr12 < max1)
        within2 = (dr13 >= min2) & (dr13 < max2)
        within3 = (dr23 >= min3) & (dr23 < max3)
        within_cutoff = within1 & within2 & within3

        dr12 = jnp.where(within1, dr12, 0)
        dr13 = jnp.where(within2, dr13, 0)
        dr23 = jnp.where(within3, dr23, 0)

        s = partial(three_body, coefficients=coefficients)

        if not featurization:
            return jnp.where(within_cutoff, s(dr12, dr13, dr23), 0.0)
        else:
            if force_features:
                s = jsp.featurization_with_gradients(s, coefficients=coefficients)
                e, f = s(dr12, dr13, dr23)
                e = jnp.where(within_cutoff, e, 0.0)
                f = jnp.where(within_cutoff, f, 0.0)
                return jnp.stack([e, f]).flatten()
            else:
                return jnp.where(within_cutoff, s(dr12, dr13, dr23), 0.0).flatten()

    @jit
    def fn(dR12, dR13, coefficients=None):
        f = partial(op, coefficients=coefficients)
        return vmap(vmap(vmap(f, (0, None)), (None, 0)))(dR12, dR13)

    return fn
