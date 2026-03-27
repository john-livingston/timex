import jax
jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

__all__ = ['io', 'util', 'model', 'plot', 'fit', 'optim']