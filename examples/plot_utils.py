from functools import wraps
import numpy as np
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import pandas as pd
import logomaker


def density_scatter(x, y, ax=None, sort=True, bins=100, cmap="plasma", **kwargs):
    """
    Scatter plot colored by 2d histogram

    Source: https://stackoverflow.com/a/53865762/9047715
    """
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn(
        (0.5 * (x_e[1:] + x_e[:-1]), 0.5 * (y_e[1:] + y_e[:-1])),
        data,
        np.vstack([x, y]).T,
        method="splinef2d",
        bounds_error=False,
    )

    # To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last (on top)
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.scatter(x, y, c=z, cmap=cmap, **kwargs)
    return ax


@wraps(logomaker.Logo)
def make_logo_from_one_hot(one_hot: np.ndarray, alphabet: str = "ACGT", **kwargs_logo):
    """Plot weights as a sequence logo."""
    assert one_hot.shape[-1] == len(alphabet), f"{one_hot.shape[-1]} != {len(alphabet)}"
    df = pd.DataFrame(one_hot, columns=list(alphabet))
    return logomaker.Logo(df, **kwargs_logo)
