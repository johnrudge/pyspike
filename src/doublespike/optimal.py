"""Module for determining optimal double spikes."""

import numpy as np
import itertools
from scipy.special import binom
from scipy.optimize import minimize, LinearConstraint
from scipy.special import expit, logit
from .errors import errorestimate

def optimalspike(isodata,type_ = 'pure',isospike = None,isoinv = None,errorratio = None,alpha = 0.0,beta = 0.0): 
    """Find the optimal double spike composition and double spike-sample mixture proportions.
    
    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
                This is the only mandatory argument.
        type(str): type of spike, 'pure' or 'real'. Real spikes, such as those from
            Oak Ridge National Labs, contain impurities. See isodata.rawspike
            for their assumed compositions. By default pure spikes are used.
        isospike (array): the isotopes used in the double spike e.g. [54, 57].
            By default all choices of 2 isotopes are tried.
        isoinv (array): the isotopes used in the inversion, e.g. [54, 56, 57, 58].
        errorratio (array): by default, the optimal double spike is chosen as that which
            minimises the error on the natural fractionation factor (known as
            alpha). Instead, the optimiser can be told to minimise the
            error on a particular ratio by setting errorratio. e.g.
            setting errorratio=[58, 56] will minimise the error on 58Fe/56Fe.
        alpha, beta (floats): there is a small dependance of the error on the fractionation
        factors (instrumental and natural, or alpha and beta). Values of alpha and
        beta can be set here if desired, although the effect on the optimal spikes
        is slight unless the fractionations are very large.

    Returns:
        All the outputs are provided as matrices. Each column represents an isotope
        (see isodata.isonum for the isotope numbers) e.g. for Fe the columns
        correspond to the isotopes 54Fe, 56Fe, 57Fe, 58Fe. The rows represent the
        different combinations of double spikes and isotopes being tried, in order of
        error: The first row is the best double spike, and the last row is the worst.
        optspike: the proportions of each isotope in the optimal double spike.
        optprop: the optimal proportion of spike in the double spike-sample mix.
        opterr: the error in the fractionation factor (or ratio if specified)
                for the optimal spike.
        optisoinv: the 4 isotopes used in the inversion.
        optspikeprop: the proportion of each raw spike in the optimal double spike.
        optppmperamu: an alternative expression of the error in terms of ppm per amu.

    Example:
        >>> isodata_fe = IsoData('Fe')
        >>> optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike(isodata_fe,'pure')
    """
    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isospike = isodata.isoindex(isospike)
    isoinv = isodata.isoindex(isoinv)
    
    # If don't specify inversion isotopes, do all possible combinations
    if isoinv is None:
        isoinv = list(itertools.combinations(np.arange(isodata.nisos()), 4))
    else:
        isoinv = list([np.array(isoinv)])
    
    # Work out all combinations of inversion isotopes and spiking isotopes
    isoinvvals = []
    isospikevals = []
    for i in range(len(isoinv)):
        if isospike is None:
            if type_ == 'pure':
                # look at all combinations of spikes from the inversion isotopes
                isospikev = list(itertools.combinations(isoinv[i], 2))
            else:
                if isodata.nrawspikes() == 0:
                    return {} # can't proceed if no single spikes to use
                # look at all combinations of spikes from the all rawspikes
                isospikev = list(itertools.combinations(np.arange(isodata.nrawspikes()), 2))
        else:
            isospikev = list([isospike])
            #if len(set(isospike).intersection(set(isoinv[i]))) == 2:
                #isospikev = list([isospike])
            #else:
                #isospikev = None
        
        if isospikev is not None:
            isospikevals.append(isospikev)
            isoinvvals.append(np.tile(isoinv[i],(len(isospikev),1)))
    isoinvvals = np.vstack(isoinvvals)
    isospikevals = np.vstack(isospikevals)
   
    optspikes = []
    optprops = []
    opterrs = []
    optppmperamus = []
    optspikeprops = []
    
    for i in range(len(isoinvvals)):
        try:
            optspike,optprop,opterr,optspikeprop,optppmperamu = singleoptimalspike(isodata,type_,isospikevals[i,:],isoinvvals[i,:],errorratio,alpha,beta)
        except:
            # try to fail gracefully
            optspike = np.zeros(isodata.nisos())
            optprop = 0.0
            opterr = 1e32
            optspikeprop = np.zeros(isodata.nisos())
            optppmperamu = 1e32
                
        optspikes.append(optspike)
        optprops.append(optprop)
        opterrs.append(opterr)
        optppmperamus.append(optppmperamu)
        optspikeprops.append(optspikeprop)
    
    optspike = np.vstack(optspikes)
    optspikeprop = np.vstack(optspikeprops)
    optprop = np.array(optprops)
    opterr = np.array(opterrs)
    optppmperamu = np.array(optppmperamus)
    optisoinv = isoinvvals
    
    ## Sort in ascending order of error
    ix = np.argsort(opterr)
    
    # avoid masses of output by limiting to all possibilites in case of pure spikes
    max_noutput = min(len(ix), int(6*binom(isodata.nisos(),4)))
    ix = ix[0:max_noutput]
    
    out = {'optspike': optspike[ix,:],
           'optprop': optprop[ix],
           'opterr': opterr[ix],
           'optisoinv': isodata.isonum[optisoinv[ix,:]],
           'optspikeprop': optspikeprop[ix,:],
           'optppmperamu': optppmperamu[ix]}
           
    return out
    
def singleoptimalspike(isodata, type_ = 'pure',isospike = None,isoinv = None,errorratio = None, alpha = 0.0, beta = 0.0): 
    """Calculate the composition of the optimal double spike given the isotopes used in the inversion and of those the isotopes we are spiking."""
    if type_ == "pure":
        spikevector1 = np.zeros(isodata.nisos())
        spikevector1[isospike[0]] = 1.0
        spikevector2 = np.zeros(isodata.nisos())
        spikevector2[isospike[1]] = 1.0
    else:
        spikevector1 = isodata.rawspike[isospike[0], :]
        spikevector2 = isodata.rawspike[isospike[1], :]
    
    # Helpful to rescale the error, to make everything roughly order 1 for the optimiser
    initialerror, _ = errorestimate(isodata,0.5,0.5*spikevector1 + (1 - 0.5)*spikevector2,isoinv,errorratio,beta,alpha)

    def objective(y):
        p = expit(y[0])  # use expit transformation to keep things in range [0,1]
        q = expit(y[1])
        
        error, ppmperamu = errorestimate(isodata,p,q*spikevector1 + (1.0 - q)*spikevector2,isoinv,errorratio,beta,alpha)
        return error/initialerror 
    
    y0 = np.array([0.0, 0.0])
    res = minimize(objective, y0, tol = 1e-9)
    
    y = res.x
    p = expit(y[0])
    q = expit(y[1])
    
    optprop = p
    optspike = q * spikevector1 + (1 - q) * spikevector2
    opterr,optppmperamu = errorestimate(isodata,p,q*spikevector1 + (1 - q)*spikevector2,isoinv,errorratio,beta,alpha)
    
    optspikeprop=np.zeros_like(optspike)
    optspikeprop[isospike[0]]=q
    optspikeprop[isospike[1]]=1-q
    
    return optspike,optprop,opterr,optspikeprop,optppmperamu

if __name__=="__main__":
    from .isodata import IsoData
    ##isodata = IsoData('Fe')
    isodata_ca = IsoData('Ca')
    isoinv = [40, 42, 44, 48]
    
    isospike = [2,5]
    
    print(optimalspike(isodata_ca,'real', isoinv = isoinv, isospike = isospike))
