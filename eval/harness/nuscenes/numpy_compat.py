"""Let the nuScenes devkit's detection metrics run under NumPy >= 1.24.

The devkit's detection/algo.py still calls np.float, which NumPy removed. The alias is
restricted to that one module; global NumPy is left unchanged.
"""


def patch_nuscenes_numpy():
    import numpy as np
    from nuscenes.eval.detection import algo
    if 'float' not in np.__dict__:
        class NumpyCompat:
            float = float

            def __getattr__(self, name):
                return getattr(np, name)
        algo.np = NumpyCompat()
