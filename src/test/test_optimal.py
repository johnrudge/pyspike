from doublespike import IsoData, optimalspike

def test_optimalspike():
    isodata = IsoData("Ca")
    isoinv = [40, 42, 44, 48]
    isospike = [2, 5]
    opt = optimalspike(isodata, "real", isoinv=isoinv, isospike=isospike)
    assert(abs(opt['optppmperamu'][0] - 53.20139232)<1e-8)
