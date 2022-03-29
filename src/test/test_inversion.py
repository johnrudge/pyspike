from doublespike import dsinversion, IsoData


def test_inversion():
    isodata_fe = IsoData("Fe")
    isodata_fe.set_spike([0.0, 0.0, 0.5, 0.5])
    measured = [0.2658, 4.4861, 2.6302, 2.6180]

    z = dsinversion(isodata_fe, measured)
    assert abs(z["alpha"] + 0.1907413) < 1e-7
