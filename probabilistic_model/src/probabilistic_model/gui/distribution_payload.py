"""
PDF/CDF/mode/expectation data for a numeric variable's marginal, as plain JSON-able
data.

Computes the same values :meth:`~probabilistic_model.gui.plotting.ProbabilisticModelPlotWidget.plot_1d_numeric`
plots with QtCharts, without depending on Qt -- shared by the desktop GUI's own export
tooling and by anything else that needs the same distribution data outside a Qt widget.
"""

from __future__ import annotations

import numpy as np
from random_events.variable import Variable

from probabilistic_model.constants import SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT
from probabilistic_model.probabilistic_model import ProbabilisticModel
from probabilistic_model.utils import neighbouring_points

NUMBER_OF_SAMPLES = 1000
"""
Matches :class:`~probabilistic_model.gui.plotting.ProbabilisticModelPlotWidget`'s own
default sample count.
"""


def numeric_variable_distribution_payload(
    model: ProbabilisticModel,
    variable: Variable,
    number_of_samples: int = NUMBER_OF_SAMPLES,
) -> dict:
    """
    PDF/CDF/mode/expectation data for ``variable``'s marginal under ``model``.

    :param model: The full model ``variable`` is marginalized out of.
    :param variable: The variable to compute the distribution of.
    :param number_of_samples: How many samples to draw before adding the support's
        boundary points.
    :return: A JSON-able dict with ``samples``, ``pdf``, ``cdf``, ``expectation``,
        ``modes`` and ``modeHeight``.
    """
    marginal = model.marginal([variable])
    samples = marginal.sample(number_of_samples)[:, 0]

    supporting_interval = marginal.support.simple_sets[0][variable]
    for simple_interval in supporting_interval.simple_sets:
        lower, upper = simple_interval.lower, simple_interval.upper
        if lower > -np.inf:
            samples = np.concatenate((samples, neighbouring_points(lower)))
        if upper < np.inf:
            samples = np.concatenate((samples, neighbouring_points(upper)))

    samples = np.sort(samples)
    lowest, highest = float(samples[0]), float(samples[-1])
    size = (highest - lowest) or 1.0
    samples = np.concatenate(([lowest - size * 0.05], samples, [highest + size * 0.05]))

    pdf = marginal.likelihood(samples.reshape(-1, 1))
    cdf = marginal.cumulative_distribution_function(samples.reshape(-1, 1))

    try:
        mode_event, max_likelihood = marginal.mode()
    except Exception:
        mode_event, max_likelihood = None, float(np.max(pdf))
    height = float(max_likelihood) * SCALING_FACTOR_FOR_EXPECTATION_IN_PLOT

    try:
        expectation = marginal.expectation([variable])[variable]
    except Exception:
        expectation = None

    modes = []
    if mode_event is not None:
        for simple_event in mode_event.simple_sets:
            interval = simple_event[variable]
            for simple_interval in interval.simple_sets:
                modes.append(
                    [float(simple_interval.lower), float(simple_interval.upper)]
                )

    return {
        "samples": [float(value) for value in samples],
        "pdf": [float(value) for value in pdf],
        "cdf": [float(value) for value in cdf],
        "expectation": float(expectation) if expectation is not None else None,
        "modes": modes,
        "modeHeight": height,
    }
