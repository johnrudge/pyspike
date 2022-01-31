import numpy as np
import matplotlib.pyplot as plt
from .errors import errorestimate, optimalspike
    
def errorcurve2d(isodata, type_ = 'pure', isospike = None, isoinv = None, errorratio = None, alpha = 0.0, beta = 0.0, resolution = 100,threshold = 0.25,ncontour = 25,plottype = 'default', **kwargs): 
    """A 2D contour plot of error as a function of double spike composition and double spike-sample proportions
    
            element -- element used in double spike, e.g. 'Fe'
            type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
               Oak Ridge National Labs, contain impurities (see 'maininput.csv'
               or ISODATA.(element).rawspike) for the assumed compositions.
               By default pure spikes are used.
            isospike -- the isotopes used in the double spike e.g. [54 57].
               By default the first two isotopes are chosen.
            isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
               By default the first four isotopes are chosen.
            errorratio -- by default, the error on the natural fractionation
               factor (known as alpha) is given. Instead, the error on a
                particular ratio can be given by setting errorratio. e.g.
               setting errorratio=[58 56] will give the error on 58Fe/56Fe.
            alpha, beta -- there is a small dependance of the error on the fractionation
               factors (natural and instrumental). Values of alpha and
               beta can be set here if desired, although the effect on the optimal spikes
               is slight unless the fractionations are very large. Default is zero.
            resolution -- number of grid points in x and y. Default is 100.
            threshold -- maximum contour to plot, relative to the minimum error.
               Default is 0.25 i.e. 25# in excess of the minimum.
            ncontour -- number of countours. Default is 25.
            plottype -- by default, the error is plotted. By setting this to 'ppmperamu'
               an estimate of the ppm per amu is plotted instead.
            ... -- additional arguments are passed to contour command.
    
    Note that a number of parameters are specified in the global variable ISODATA.
    
    Example
        errorcurve2d('Fe','pure',[57, 58])"""
        
    # Get data from isodata if not supplied as arguments
    if isoinv is None:
        if hasattr(isodata, 'isoinv'):
            isoinv = isodata.isoinv
        else:
            isoinv = isodata.isonum[0:4]

    if isospike is None:
        isospike = np.array([0,1])
    else:
        isodata.isoindex(isospike)
    
    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isoinv = isodata.isoindex(isoinv)
    isospike = isodata.isoindex(isospike)

    if type_ == 'pure':
        spike1 = np.zeros(isodata.nisos())
        spike2 = np.zeros(isodata.nisos())
        spike1[isospike[0]] = 1
        spike2[isospike[1]] = 1
    else:
        spike1 = isodata.rawspike[isospike[0],:]
        spike2 = isodata.rawspike[isospike[1],:]
    
    prop = np.linspace(0.001,0.999,resolution)
    spikeprop = np.linspace(0.001,0.999,resolution)
    iv,jv = np.meshgrid(np.arange(len(prop)),np.arange(len(spikeprop)))
    
    def fun(i,j):
        return errorestimate(isodata,prop[i],spikeprop[j]*spike1 + (1 - spikeprop[j])*spike2,isoinv,errorratio,alpha,beta)
    vfun = np.vectorize(fun)
    errvals,ppmperamuvals = vfun(iv, jv)
    
    optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike(isodata,type_,isospike,isoinv,errorratio,alpha,beta)
    
    if plottype == 'ppmperamu':
        C = plt.contour(prop,spikeprop,ppmperamuvals,np.linspace(optppmperamu,(1 + threshold) * optppmperamu,ncontour + 1),**kwargs)
    else:
        C = plt.contour(prop,spikeprop,errvals,np.linspace(opterr[0],(1 + threshold) * opterr[0],ncontour + 1),**kwargs)
    
    plt.xlim(np.array([0,1]))
    plt.ylim(np.array([0,1]))
    plt.xlabel('proportion of double spike in double spike-sample mix')
    
    isolabel = isodata.isolabel()
    if type_ == 'pure':
        plt.ylabel('proportion of '+isolabel[isospike[0]]+' in '+isolabel[isospike[0]]+'-'+isolabel[isospike[1]]+' double spike')
    else:
        rawspikelabel = isodata.rawspikelabel()
        plt.ylabel('proportion of '+rawspikelabel[isospike[0]]+' in '+rawspikelabel[isospike[0]]+'-'+rawspikelabel[isospike[1]]+' double spike')
    isoinv = isodata.isoindex(isoinv)
    invisostring = isolabel[isoinv[0]]+', '+isolabel[isoinv[1]]+', '+isolabel[isoinv[2]]+', '+isolabel[isoinv[3]]+' inversion'
    if errorratio is None:
        plt.title(r'Error in $\alpha$ (1SD) with '+invisostring)
    else:
        plt.title('Error in '+isolabel[errorratio[0]]+'/',isolabel[errorratio[1]]+' (1SD) with '+invisostring)
    
    plt.plot(optprop[0],optspikeprop[0,isospike[0]],'rx')
    
if __name__=="__main__":
    from .isodata import IsoData
    idat = IsoData('Fe')
    idat.set_errormodel()
    errorcurve2d(idat,'pure',[57,58], resolution = 100);
    plt.show()
    
