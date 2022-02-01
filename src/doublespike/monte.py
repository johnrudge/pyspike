"""Routines for Monte Carlo mass spec simulation."""
import numpy as np
from numpy.random import default_rng
from .isodata import IsoData
from .inversion import dsinversion
    
def monterun(isodata, prop = None, spike = None, alpha = 0.0, beta = 0.0, n = 1000): 
    """Generate a fake mass spectrometer run by Monte-Carlo simulation.
    
    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        prop (float/array): proportion of spike in double spike-sample mix.
            A vector of values can be specified if desired, to reflect changes over a run.
        spike (array): a composition vector for the spike. e.g. [0, 0, 0.5, 0.5] is a 50-50
            mix of 57Fe and 58Fe. If None this is read from isodata.
        alpha (float/array): the natural fractionations (can be float or array of values)
        beta (float/array): the instrumental fractionations (can be float or array of values)
        n (int): number of Monte-Carlo samples to take. Default is 1000.

        Note that behaviour depends on the error model specified in isodata.errormodel. 
    
    Example:
        >>> isodata = IsoData('Fe')
        >>> measured=monterun(isodata,0.5,[0, 0, 0.5, 0.5])
    """
    # Get spike from isodata if not supplied as argument
    if spike is None:
        if isodata.spike is None:
            raise Exception("No spike given")
        else:
            spike = isodata.spike
    spike = np.array(spike)
            
    if isodata.errormodel == {}:
        raise Exception("Need to set errormodel")
    
    standard = isodata.standard
    mass = isodata.mass
    emodel = isodata.errormodel
    spike = spike/sum(spike)
    nisos = isodata.nisos()
    
    if isinstance(alpha, float):
        alpha = alpha * np.ones(n)
    if isinstance(beta, float):
        beta = beta * np.ones(n)
    if isinstance(prop, float):
        prop = prop * np.ones(n)
     
    P = np.log(mass)[np.newaxis,:]
    alpha = alpha[:,np.newaxis]
    beta = beta[:,np.newaxis]
    prop = prop[:,np.newaxis]
    
    standard = standard[np.newaxis,:]
    spike = spike[np.newaxis,:]
    
    
    sample = standard * np.exp( - P * alpha)
    sample = normalise_composition(sample)
    
    mixture = prop * spike + (1-prop) * sample
    
    measured = mixture * np.exp( + P*beta)
    measured = normalise_composition(measured)
    
    def calc_var(data, emod):
        datai = data *  emod['intensity']    
        if emod['type'] == 'fixed-sample':
            datai = datai / (1.0 - prop)
        dataivar = emod['a'] + emod['b'] * datai + emod['c'] * (datai **2)
        return datai, dataivar
    
    measuredi, measuredivar = calc_var(measured, emodel['measured'])
    standardi, standardivar = calc_var(standard, emodel['standard'])
    spikei, spikeivar = calc_var(spike, emodel['spike'])

    rng = default_rng()
    measuredv = rng.normal(loc = measuredi, scale = np.sqrt(measuredivar))
    standardv = None
    spikev = None

    if np.any(standardivar>0.0):
        standardi = np.tile(standardi, (n,1))
        standardivar = np.tile(standardivar, (n,1))
        standardv = rng.normal(loc = standardi, scale = np.sqrt(standardivar))

    if np.any(spikeivar>0.0):
        spikei = np.tile(spikei, (n,1))
        spikeivar = np.tile(spikeivar, (n,1))
        spikev = rng.normal(loc = spikei, scale = np.sqrt(spikeivar))
    
    if standardv is None and spikev is None:
        return measuredv
    elif standardv is not None:
        return measuredv, standardv
    else:
        return measuredv, standardv, spikev

    
def normalise_composition(comp):
    """Normalise an array so rows have unit sum, i.e. rows are compositional vectors."""
    return comp / comp.sum(axis=1)[:, np.newaxis]    
    
if __name__=="__main__":
    idat = IsoData('Fe')
    idat.set_spike([0.0, 0.0, 0.5, 0.5])
    idat.set_errormodel()
    measuredv = monterun(idat, prop = 0.5, alpha = -0.2, beta = 1.8, n = 1000)
    print(measuredv[0:10,:])
    
    out = dsinversion(idat, measuredv)
    import matplotlib.pyplot as plt
    plt.plot(out['alpha'])
    plt.ylabel(r'$\alpha$')
    plt.show()
