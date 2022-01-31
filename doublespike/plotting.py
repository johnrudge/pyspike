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

def errorcurve(isodata,spike = None,isoinv = None,errorratio = None,alpha = 0.0,beta = 0.0,plottype = 'default',**kwargs): 
    """ERRORCURVE    A plot of error as a function of double spike-sample proportions for a given double spike composition
    ERRORCURVE(element,spike,isoinv,errorratio,alpha,beta,...)
            element -- element used in double spike, e.g. 'Fe'
            spike -- the composition of the double spike as a composition vector e.g. [0 0 0.5 0.5]
               represents a 50-50 mixture of the third and fourth isotopes (57-58 for Fe).
            isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
               By default the first four isotopes are chosen.
            errorratio -- by default, the error on the natural fractionation
               factor (known as alpha) is given. Instead, the error on a
                particular ratio can be given by setting errorratio. e.g.
               setting errorratio=[58 56] will give the error on 58Fe/56Fe.
            alpha, beta -- there is a small dependance of the error on the fractionation
               factors (natural and instrumental). Values of beta and
               alpha can be set here if desired, although the effect on the optimal spikes
               is slight unless the fractionations are very large. Default is zero.
            plottype -- by default, the error is plotted. By setting this to 'ppmperamu'
               an estimate of the ppm per amu is plotted instead.
            ... -- additional arguments are passed to the plot command.
    
    Note that a number of parameters are specified in the global variable ISODATA.
    
    Example
        errorcurve('Fe',[0 0 0.5 0.5])
    
    See also errorestimate, errorcurve2"""
    
    # Get data from isodata if not supplied as arguments
    if isoinv is None:
        if hasattr(isodata, 'isoinv'):
            isoinv = isodata.isoinv
        else:
            isoinv = isodata.isonum[0:4]
    spike = np.array(spike)
    spike = spike / sum(spike)
    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isoinv = isodata.isoindex(isoinv)
    pvals = np.linspace(0.001,0.999,1000)
    errvals = np.zeros(len(pvals))
    ppmperamuvals = np.zeros(len(pvals))
    for i in range(len(pvals)):
        errvals[i],ppmperamuvals[i] = errorestimate(isodata,pvals[i],spike,isoinv,errorratio,alpha,beta)
    
    if plottype=='ppmperamu':
        plotvals = ppmperamuvals
    else:
        plotvals = errvals
    
    plt.plot(pvals,plotvals,**kwargs)
    mine = np.amin(plotvals)
    plt.xlim(np.array([0,1]))
    plt.ylim(np.array([0,5 * mine]))
    plt.xlabel('proportion of double spike in double spike-sample mix')
    isolabel = isodata.isolabel()
    if errorratio is None:
        plt.ylabel(r'Error in $\alpha$ (1SD)')
    else:
        plt.ylabel('Error in '+isodata.isolabel[errorratio[0]]+'/'+isolabel[errorratio[1]]+' (1SD)')
    
    plt.title(isolabel[isoinv[0]]+', '+isolabel[isoinv[1]]+', '+isolabel[isoinv[2]]+', '+isolabel[isoinv[3]]+' inversion')


def errorcurve2(isodata,type_ = 'pure',prop = 0.5,isospike = None, isoinv = None,errorratio = None,alpha = 0.0,beta = 0.0,plottype = 'default',**kwargs): 
    """ERRORCURVE2    A plot of error as a function of double spike proportions for a given double spike-sample proportion
 ERRORCURVE2(element,type,prop,isospike,isoinv,errorratio,alpha,beta,...)
            element -- element used in double spike, e.g. 'Fe'
            type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
               Oak Ridge National Labs, contain impurities (see 'data/maininput.csv'
               or ISODATA.(element).rawspike) for their assumed compositions.
               By default pure spikes are used.
            prop -- proportion of double spike in double spike-sample mixture e.g. 0.5.
            isospike -- the isotopes used in the double spike e.g. [54 57].
               By default the first two isotopes are chosen.
            isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
               By default the first four isotopes are chosen.
            errorratio -- by default, the error on the natural fractionation
               factor (known as alpha or alpha) is given. Instead, the
               error on a particular ratio can be given by setting errorratio. e.g.
               setting errorratio=[58 56] will give the error on 58Fe/56Fe.
            alpha, beta -- there is a small dependance of the error on the fractionation
               factors (natural and instrumental). Values of beta and
               alpha can be set here if desired, although the effect on the optimal spikes
               is slight unless the fractionations are very large. Default is zero.
            plottype -- by default, the error is plotted. By setting this to 'ppmperamu'
               an estimate of the ppm per amu is plotted instead.
            ... -- additional arguments are passed to the plot command.
    
    Note that a number of parameters are specified in the global variable ISODATA.
    
    Example
        errorcurve2('Fe','real',0.5,[54 57])
    
    See also errorestimate, errorcurve"""
    
    # Get data from isodata if not supplied as arguments
    if isoinv is None:
        if hasattr(isodata, 'isoinv'):
            isoinv = isodata.isoinv
        else:
            isoinv = isodata.isonum[0:4]
    
    if isospike is None:
        isospike = np.array([1,2])
    
    rawspike = isodata.rawspike
    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isoinv = isodata.isoindex(isoinv)
    isospike = isodata.isoindex(isospike)
    qvals = np.linspace(0.001,0.999,1000)
    errvals = np.zeros(len(qvals))
    ppmperamuvals = np.zeros(len(qvals))
    for i in range(len(qvals)):
        #spike = (np.multiply(qvals(i),rawspike(isospike(1),:))) + (np.multiply((1 - qvals(i)),rawspike(isospike(2),:)))
        # ? real only?
        # NEED TO FIX THIS BUG
        spike = qvals[i] * rawspike[isospike[0],:] + (1-qvals[i]) * rawspike[isospike[1],:]
        errvals[i],ppmperamuvals[i] = errorestimate(isodata,prop,spike,isoinv,errorratio,alpha,beta)
    
    if plottype=='ppmperamu':
        plotvals = ppmperamuvals
    else:
        plotvals = errvals
    
    isolabel = isodata.isolabel()
    plt.plot(qvals,plotvals,**kwargs)
    mine = np.amin(plotvals)
    plt.xlim(np.array([0,1]))
    plt.ylim(np.array([0,5 * mine]))
    if type_ == 'pure':
        plt.xlabel('proportion of '+isolabel[isospike[0]]+' in '+isolabel[isospike[0]]+'-'+isolabel[isospike[1]]+' double spike')
    else:
        plt.xlabel('proportion of first rawspike in double spike')
    
    if errorratio is None:
        plt.ylabel(r'Error in $\alpha$ (1SD)')
    else:
        plt.ylabel('Error in '+isolabel[errorratio[0]]+'/'+isolabel[errorratio[1]]+' (1SD)')
    
    plt.title(isolabel[isoinv[0]]+', '+isolabel[isoinv[1]]+', '+isolabel[isoinv[2]]+', '+isolabel[isoinv[3]]+' inversion')


def errorcurveoptimalspike(isodata,type_ = 'pure',isospike = None,isoinv = None,errorratio = None,alpha = 0.0,beta = 0.0,plottype = 'default',**kwargs): 
    """ERRORCURVEOPTIMALSPIKE    Find the optimal double spike compositions and plot the corresponding error curves
[optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu]
 =ERRORCURVEOPTIMALSPIKE(element,type,isoinv,isospike,errorratio,alpha,beta,...)
            element -- element used in double spike, e.g. 'Fe'
               This is the only mandatory argument.
            type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
               Oak Ridge National Labs, contain impurities (see 'maininput.csv'
               or ISODATA.(element).rawspike) for their assumed compositions.
               By default pure spikes are used.
            isospike -- the isotopes used in the double spike e.g. [54 57].
               By default all choices of 2 isotopes are tried.
            isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
               By default the first four isotopes are used.
            errorratio -- by default, the optimal spike is chosen as that which
               minimises the error on the natural fractionation factor (known as
               alpha). Instead, the optimiser can be told to minimise the
               error on a particular ratio by setting errorratio. e.g.
               setting errorratio=[58 56] will minimise the error on 58Fe/56Fe.
            alpha, beta -- there is a small dependance of the error on the fractionation
               factors (natural and instrumental). Values of alpha and
               beta can be set here if desired, although the effect on the optimal spikes
               is slight unless the fractionations are very large. Default is zero.
            plottype -- by default, the error is plotted. By setting this to 'ppmperamu'
               an estimate of the ppm per amu is plotted instead.
            ... -- additional arguments are passed to the plot command
    
    Outputs are the same as those of optimalspike.
    
    Example
        errorcurveoptimalspike('Fe')
    
    See also optimalspike, errorcurve"""
    
    if isoinv is None:
        if hasattr(isodata, 'isoinv'):
            isoinv = isodata.isoinv
        else:
            isoinv = isodata.isonum[0:4]
    
    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isospike = isodata.isoindex(isospike)
    isoinv = isodata.isoindex(isoinv)
    # Find the optimal spikes
    optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike(isodata,type_,isospike,isoinv,errorratio,alpha,beta)
    isolabel = isodata.isolabel()
    rawspikelabel = isodata.rawspikelabel()
    for j in range(optspike.shape[0]):
        if type_ == 'pure':
            spiked = np.where(optspike[j,:] > 0)[0]
            leglabel = isolabel[spiked[0]]+'-'+isolabel[spiked[1]]
        else:
            spiked = np.where(optspikeprop[j,:] > 0)[0]
            leglabel =rawspikelabel[spiked[0]]+'-'+rawspikelabel[spiked[1]]
        errorcurve(isodata,optspike[j,:],isoinv,errorratio,alpha,beta,plottype,label=leglabel)
        
    if plottype=='ppmperamu':
        plt.ylim(np.array([0,5 * np.amin(optppmperamu)]))
    else:
        plt.ylim(np.array([0,5 * np.amin(opterr)]))
    
    #hold('off')
    plt.legend(loc='upper right')
    return optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu

if __name__=="__main__":
    from .isodata import IsoData
    idat = IsoData('Fe')
    idat.set_errormodel()
    #errorcurve2d(idat,'real',[2, 3], resolution = 100);
    #spike = [0.0, 0.0, 0.5, 0.5]
    #errorcurve(idat, spike)
    #errorcurve2(idat, 'pure', 0.5, [57, 58])
    
    errorcurveoptimalspike(idat,'real')
    
    plt.show()
    
