import numpy as onp

import jax.numpy as jnp
from jax import vmap, grad, jit
from jax import lax

import optax

from functools import partial

from uf3.regression.regularize import get_regularizer_matrix, get_penalty_matrix_3D


def regularizer(coefficients, ridge=0.0, curvature=1.0):
    if len(coefficients.shape) == 1:
        return jnp.sum(
            jnp.einsum(
                "ij,j->i",
                get_regularizer_matrix(len(coefficients), ridge=ridge, curvature=curvature),
                coefficients,
            )
            ** 2
        )
    if len(coefficients.shape) == 3:
        return jnp.sum(
            jnp.einsum(
                "ij,j->i",
                get_penalty_matrix_3D(*coefficients.shape, ridge=ridge, curvature=curvature),
                coefficients.flatten(),
            )
            ** 2
        )

# def loss_uf2(coefficients, S, E, F, ufp, kappa=0.5, lam=1.0):
#     sigE = jnp.sum((E - E.mean())**2)
#     sigF = jnp.sum((F - F.mean())**2)
#     dE = 0.0
#     dF = 0.0
#     # for s, e, f in zip(S,E,F):
#     se = partial(s, coefficients=coefficients)
#     v = vmap(se)(S)
#     g = ref_f(S, coefficients=coefficients)
#     dE += (v - E) ** 2
#     dF += (-g - F) ** 2

#     E_term = (kappa / sigE) * jnp.sum(dE)
#     F_term = ((1-kappa) / sigF) * jnp.sum(dF)

#     D = jnp.asarray(get_regularizer_matrix(len(coefficients)))
#     L_term = lam * jnp.sum((D * coefficients)**2)
 
#     return E_term + F_term + L_term


# def per_atom_f(fn, c):
#     def f(x, coefficients=c):
#         s = partial(fn, coefficients=coefficients)
#         tmp = vmap(s)(x)
#         return jnp.sum(tmp)
#     return grad(f)


# neighbor_fn, energy_fn = uf2_neighbor(displacement, box_size, knots=knots, cutoff=3.0)

# neighbor = neighbor_fn.allocate(positions[0], extra_capacity=20)

# @jit
# def train_energy_fn(params, R):
#     _neighbor = neighbor_fn.update(R, neighbor)
#     return energy_fn(R, _neighbor, coefficients=params)

# vectorized_energy_fn = vmap(train_energy_fn, (None, 0))

# grad_fn = grad(train_energy_fn, argnums=1)
# force_fn = lambda params, R, **kwargs: -grad_fn(params, R)
# vectorized_force_fn = vmap(force_fn, (None, 0))

# params = np.zeros(len(knots)-k)

# @jit
# def energy_loss(params, R, energy_targets):
#   return np.mean((vectorized_energy_fn(params, R) - energy_targets) ** 2)

# @jit
# def force_loss(params, R, force_targets):
#   dforces = vectorized_force_fn(params, R) - force_targets
#   return np.mean(np.sum(dforces ** 2, axis=(1, 2)))

# @jit
# def loss(params, R, targets):
#   return energy_loss(params, R, targets[0]) + force_loss(params, R, targets[1])

# opt = optax.chain(optax.clip_by_global_norm(1.0),
#                   optax.adam(1e-3))

# @jit
# def update_step(params, opt_state, R, labels):
#   updates, opt_state = opt.update(grad(loss)(params, R, labels),
#                                   opt_state)
#   return optax.apply_updates(params, updates), opt_state

# @jit
# def update_epoch(params_and_opt_state, batches):
#   def inner_update(params_and_opt_state, batch):
#     params, opt_state = params_and_opt_state
#     b_xs, b_labels = batch

#     return update_step(params, opt_state, b_xs, b_labels), 0
#   return lax.scan(inner_update, params_and_opt_state, batches)[0]

# dataset_size = positions.shape[0]
# batch_size = 128

# lookup = onp.arange(dataset_size)
# onp.random.shuffle(lookup)

# @jit
# def make_batches(lookup):
#   batch_Rs = []
#   batch_Es = []
#   batch_Fs = []

#   for i in range(0, len(lookup), batch_size):
#     if i + batch_size > len(lookup):
#       break

#     idx = lookup[i:i + batch_size]

#     batch_Rs += [positions[idx]]
#     batch_Es += [energies[idx]]
#     batch_Fs += [forces[idx]]

#   return np.stack(batch_Rs), np.stack(batch_Es), np.stack(batch_Fs)

# batch_Rs, batch_Es, batch_Fs = make_batches(lookup)