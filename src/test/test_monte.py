from doublespike import IsoData, monterun, dsinversion, errorestimate
import numpy as np


def test_monterun():
    isodata = IsoData("Fe")

    spike = [0, 0, 0.5, 0.5]
    prop = 0.5
    alpha = -0.2
    beta = 1.8
    n = 100

    measured = monterun(isodata, prop, spike, alpha, beta, n)

    out = dsinversion(isodata, measured, spike)

    monte_error = np.std(out["alpha"])
    predicted_error, _ = errorestimate(isodata, prop, spike, alpha=alpha, beta=beta)

    ratio = monte_error / predicted_error
    # with statistical fluctuations this won't be exactly 1 so have a
    # wide tolerance
    assert ratio < 1.3 and ratio > 0.7
