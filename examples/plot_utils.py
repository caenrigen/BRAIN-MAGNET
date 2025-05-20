from typing import Optional
import numpy as np
from scipy.interpolate import interpn
from deeplift.visualization import viz_sequence


def density_scatter(x, y, ax, sort=True, bins=100, cmap="magma", **kwargs):
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

    ax.scatter(x, y, c=z, cmap=cmap, **kwargs)


def plot_weights(
    inputs,
    shap_vals,
    start: Optional[int] = None,
    end: Optional[int] = None,
    hypothetical: bool = False,
):
    segment = inputs[:, start:end]
    hyp_imp_scores_segment = shap_vals[:, start:end]
    # * The actual importance scores can be computed using an element-wise product of
    # * the hypothetical importance scores and the actual importance scores
    if not hypothetical:
        scores = hyp_imp_scores_segment * segment
    else:
        scores = hyp_imp_scores_segment
    viz_sequence.plot_weights(scores, subticks_frequency=20)
