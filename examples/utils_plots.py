import numpy as np
from scipy.interpolate import interpn
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.colors import Normalize


def density_scatter(x, y, ax, fig, sort=True, bins=100, cmap="magma", **kwargs):
    """Scatter plot colored by 2d histogram"""
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

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    # norm = Normalize(vmin=np.min(z), vmax=np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    # cbar.ax.set_ylabel("Density")

    return ax
