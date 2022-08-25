import jax.numpy as jnp
from jax import grad, jacrev #,jvp

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

    deformation = jnp.zeros_like(box)

    return lambda R, nbrs, *args, **kwargs: stress_fn(
        R, nbrs, deformation, *args, **kwargs
    )


def stress_neighborlist_featurization_fn(energy_fn, box):
    """
    Returns the per atom per coefficient contributions as a list in Voigt order (xx, yy, zz, yz, xz, xy).
    energy_fn has to be a uf3_potential with featurization=True and force_features=False.

    Stress transformation adapted from
    https://github.com/sirmarcel/asax/blob/c4e2ca89880b154b026d5b645d58e9d6163429a1/asax/jax_utils.py#L17
    """

    transform_box_fn = lambda deformation: space.transform(
        jnp.eye(deformation.shape[0]) + (deformation + deformation.T) * 0.5, box
    )

    strained_energy_fn = lambda R, nbrs, deformation, *args, **kwargs: energy_fn(
        R, nbrs, *args, **kwargs, box=transform_box_fn(deformation)
    )

    box_volume = jnp.linalg.det(box)
    # tangentxx = jnp.asarray([[1.0, 0, 0], [0, 0, 0], [0, 0, 0]])
    # tangentyy = jnp.asarray([[0.0, 0, 0], [0, 1, 0], [0, 0, 0]])
    # tangentzz = jnp.asarray([[0.0, 0, 0], [0, 0, 0], [0, 0, 1]])
    # tangentyz = jnp.asarray([[0.0, 0, 0], [0, 0, 1], [0, 0, 0]])
    # tangentxz = jnp.asarray([[0.0, 0, 1], [0, 0, 0], [0, 0, 0]])
    # tangentxy = jnp.asarray([[0.0, 1, 0], [0, 0, 0], [0, 0, 0]])

    def stress_fn(R, nbrs, deformation, *args, **kwargs):
        fn = lambda x: strained_energy_fn(R, nbrs, x, *args, **kwargs)
        stress = jacrev(fn)(deformation)
        # _, stressxx = jvp(fn, (deformation,), (tangentxx,))
        # _, stressyy = jvp(fn, (deformation,), (tangentyy,))
        # _, stresszz = jvp(fn, (deformation,), (tangentzz,))
        # _, stressyz = jvp(fn, (deformation,), (tangentyz,))
        # _, stressxz = jvp(fn, (deformation,), (tangentxz,))
        # _, stressxy = jvp(fn, (deformation,), (tangentxy,))
        # stressxx = stressxx / box_volume
        # stressyy = stressyy / box_volume
        # stresszz = stresszz / box_volume
        # stressyz = stressyz / box_volume
        # stressxz = stressxz / box_volume
        # stressxy = stressxy / box_volume
        # stress = [stressxx, stressyy, stresszz, stressyz, stressxz, stressxy]
        return stress / box_volume

    deformation = jnp.zeros_like(box)

    return lambda R, nbrs, *args, **kwargs: stress_fn(
        R, nbrs, deformation, *args, **kwargs
    )

