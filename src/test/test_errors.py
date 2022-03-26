from doublespike import errorestimate, IsoData

def test_errorestimate():
    isodata = IsoData("Fe")
    isodata.set_spike([0.0, 0.0, 0.5, 0.5])
    alpha_err, ppm_err = errorestimate(isodata, prop=0.5, alpha=-0.2, beta=1.8)
    assert(abs(alpha_err - 0.0036364520) < 1e-9)
