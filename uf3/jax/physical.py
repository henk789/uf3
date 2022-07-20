import jax.numpy as jnp
from jax import grad

from jax_md import space


def stress_fn(energy_fn, box):
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


def stress_neighborlist_fn(energy_fn, box):
    """
    Stress transformation adapted from
    https://github.com/sirmarcel/asax/blob/c4e2ca89880b154b026d5b645d58e9d6163429a1/asax/jax_utils.py#L17
    """

    transform_box_fn = lambda deformation: space.transform(
        jnp.eye(deformation.shape[0]) + (deformation + deformation.T) * 0.5, box
    )

    strained_energy_fn = lambda R, nbrs, deformation, *args, **kwargs: energy_fn(
        R, nbrs, *args, **kwargs, box=transform_box_fn(deformation)
    )

    total_strained_energy_fn = lambda R, nbrs, deformation, *args, **kwargs: jnp.sum(
        strained_energy_fn(R, nbrs, deformation, *args, **kwargs)
    )

    box_volume = jnp.linalg.det(box)
    stress_fn = (
        lambda R, nbrs, deformation, *args, **kwargs: grad(
            total_strained_energy_fn, argnums=2
        )(R, nbrs, deformation, *args, **kwargs)
        / box_volume
    )

    box_volume = jnp.linalg.det(box)
    deformation = jnp.zeros_like(box)

    return lambda R, nbrs, *args, **kwargs: stress_fn(R, nbrs, deformation, *args, **kwargs)