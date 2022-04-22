from doublespike import spike_calibration, IsoData
from doublespike.isodata import normalise_composition
import numpy as np


def test_calibrate():
    # Check the calibration routine does the right thing for perfect measurements
    isodata = IsoData("Ca")
    isodata.isoinv = [42, 43, 44, 46]
    true_spike = np.array([1e-3, 1e-2, 0.4, 0.6, 1e-4, 1e-6])
    true_spike = true_spike / sum(true_spike)

    beta_spike = 0.8
    spike_measurement = normalise_composition(true_spike * isodata.mass**beta_spike)

    prop1, beta1 = 0.3, 1.2
    mixture1_measurement = prop1 * true_spike + (1 - prop1) * isodata.standard
    mixture1_measurement = normalise_composition(
        mixture1_measurement * isodata.mass**beta1
    )

    prop2, beta2 = 0.6, -0.6
    mixture2_measurement = prop2 * true_spike + (1 - prop2) * isodata.standard
    mixture2_measurement = normalise_composition(
        mixture2_measurement * isodata.mass**beta2
    )

    mixture_measurement = np.vstack([mixture1_measurement, mixture2_measurement])

    out = spike_calibration(isodata, spike_measurement, mixture_measurement)

    assert np.all(abs(out["calibrated_spike"] - true_spike) < 1e-10)
