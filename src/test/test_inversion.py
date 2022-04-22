from doublespike import dsinversion, IsoData
from doublespike.isodata import normalise_composition
import pytest

test_data = [
    ("Fe", [0.0, 0.0, 0.5, 0.5], None, 0.5, -0.2, 1.7),
    ("Ca", [0.031, 0.367, 0.001, 0.002, 0.0, 0.597], [40, 42, 44, 48], 0.3, 0.1, -0.2)
]

@pytest.mark.parametrize("element, spike, isoinv, prop, alpha, beta",test_data)
def test_inversion(element, spike, isoinv, prop, alpha, beta):
    isodata = IsoData(element)
    isodata.spike = spike
    if isoinv is not None:
        isodata.isoinv = isoinv
    sample = normalise_composition(isodata.standard * isodata.mass**(-alpha))
    mixture = prop * isodata.spike + (1.0 - prop) * sample
    measured = normalise_composition(mixture * isodata.mass**beta)
    
    z = dsinversion(isodata, measured)
    assert( abs(z["alpha"] - alpha) <  1e-7)
    assert( abs(z["beta"] - beta) <  1e-7)
    assert( abs(z["prop"] - prop) <  1e-7)
