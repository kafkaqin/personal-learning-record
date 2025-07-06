import jax
import jax.numpy as jnp

def f(x):
    return x**2+jnp.sin(x)

df = jax.grad(f)
x  = 0.0
print("f'(0) =",df(x))

d2f = jax.grad(jax.grad(f))
print("d2f(0) =",d2f(x))

vmap_df = jax.vmap(df)
xs = jnp.array([0.0,jnp.pi/2,jnp.pi])
print("xs=",vmap_df(xs))
