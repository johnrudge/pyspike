"""Plotting routines using matplotlib."""
import numpy as np
import matplotlib.pyplot as plt
from .errors import errorestimate
from .optimal import optimalspike
    
def errorcurve2d(isodata, type_ = 'pure', isospike = None, isoinv = None, errorratio = None, alpha = 0.0, beta = 0.0, resolution = 100,threshold = 0.25,ncontour = 25,plottype = 'default', **kwargs): 
    """2D contour plot of error as a function of double spike composition and double spike-sample proportions.
    
    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        type (str): type of spike, 'pure' or 'real'. Real spikes, such as those from
            Oak Ridge National Labs, contain impurities. See isodata.rawspike for the assumed compositions.
            By default pure spikes are used.
        isospike(array): the isotopes used in the double spike e.g. [54, 57].
        isoinv (array): the isotopes used in the inversion, e.g. [54, 56, 57, 58].
        errorratio (array): by default, the error on the natural fractionation
            factor (known as alpha) is given. Instead, the error on a
            particular ratio can be given by setting errorratio. e.g.
            setting errorratio=[58, 56] will give the error on 58Fe/56Fe.
        alpha, beta (float): there is a small dependance of the error on the fractionation
            factors (natural and instrumental). Values of alpha and
            beta can be set here if desired, although the effect on the optimal spikes
            is slight unless the fractionations are very large. Default is zero.
        resolution (int): number of grid points in x and y. Default is 100.
        threshold (float): maximum contour to plot, relative to the minimum error.
            Default is 0.25 i.e. 25% in excess of the minimum.
        ncontour (int): number of countours. Default is 25.
        plottype (str): by default, the error is plotted. By setting this to 'ppmperamu'
            an estimate of the ppm per amu is plotted instead.
        **kwargs: additional keyword arguments are passed to contour command.
    
    Example:
        >>> isodata_fe = IsoData('Fe')
        >>> errorcurve2d(isodata_fe,'pure',[57, 58])
        
    See also errorestimate
    """
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
    
    # Ensure working with numpy arrays
    isoinv = np.array(isoinv)
    isospike = np.array(isospike)
    
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
    
    os = optimalspike(isodata,type_,isospike,isoinv,errorratio,alpha,beta)
    
    if plottype == 'ppmperamu':
        C = plt.contour(prop,spikeprop,ppmperamuvals,np.linspace(os['optppmperamu'],(1 + threshold) * os['optppmperamu'],ncontour + 1),**kwargs)
    else:
        C = plt.contour(prop,spikeprop,errvals,np.linspace(os['opterr'][0],(1 + threshold) * os['opterr'][0],ncontour + 1),**kwargs)
    
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
    
    plt.plot(os['optprop'][0],os['optspikeprop'][0,isospike[0]],'rx')

def errorcurve(isodata,spike = None,isoinv = None,errorratio = None,alpha = 0.0,beta = 0.0,plottype = 'default',**kwargs): 
    """Plot of error as a function of double spike-sample proportions for a given double spike composition.
    
    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        spike (array): the composition of the double spike as a composition vector e.g. [0, 0, 0.5, 0.5]
            represents a 50-50 mixture of the third and fourth isotopes (57-58 for Fe).
        isoinv (array): the isotopes used in the inversion, e.g. [54, 56, 57, 58].
            By default the first four isotopes are chosen.
        errorratio (array): by default, the error on the natural fractionation
            factor (known as alpha) is given. Instead, the error on a
            particular ratio can be given by setting errorratio. e.g.
            setting errorratio=[58, 56] will give the error on 58Fe/56Fe.
        alpha, beta (float): there is a small dependance of the error on the fractionation
            factors (natural and instrumental). Values of beta and
            alpha can be set here if desired, although the effect on the optimal spikes
            is slight unless the fractionations are very large.
        plottype (str): by default, the error is plotted. By setting this to 'ppmperamu'
            an estimate of the ppm per amu is plotted instead.
        **kwargs -- additional keyword arguments are passed to the plot command.
    
    Example:
        >>> isodata_fe = IsoData('Fe')
        >>> errorcurve(isodata_fe,[0, 0, 0.5, 0.5])
    
    See also errorestimate, errorcurve2
    """
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
    """Plot of error as a function of double spike proportions for a given double spike-sample proportion.
    
    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        type (str): type of spike, 'pure' or 'real'. Real spikes, such as those from
            Oak Ridge National Labs, contain impurities. See isodata.rawspike
            for their assumed compositions. By default pure spikes are used.
        prop (float): proportion of double spike in double spike-sample mixture e.g. 0.5.
        isospike (array): the isotopes used in the double spike e.g. [54, 57].
        isoinv (array): the isotopes used in the inversion, e.g. [54, 56, 57, 58].
        errorratio (array):by default, the error on the natural fractionation
            factor (known as alpha or alpha) is given. Instead, the
            error on a particular ratio can be given by setting errorratio. e.g.
            setting errorratio=[58, 56] will give the error on 58Fe/56Fe.
        alpha, beta (float): there is a small dependance of the error on the fractionation
            factors (natural and instrumental). Values of beta and
            alpha can be set here if desired, although the effect on the optimal spikes
            is slight unless the fractionations are very large. 
        plottype (str): by default, the error is plotted. By setting this to 'ppmperamu'
            an estimate of the ppm per amu is plotted instead.
        **kwargs -- additional arguments are passed to the plot command.
    
    Example:
        >>> isodata_fe = IsoData('Fe')
        >>> errorcurve2(isodata_fe,'real',0.5,[54,57])
    
    See also errorestimate, errorcurve
    """
    # Get data from isodata if not supplied as arguments
    if isoinv is None:
        if hasattr(isodata, 'isoinv'):
            isoinv = isodata.isoinv
        else:
            raise Exception('Inversion isotopes not set')
    
    if isospike is None:
        raise Exception('isospike not set')
    
    rawspike = isodata.rawspike
    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isoinv = isodata.isoindex(isoinv)
    isospike = isodata.isoindex(isospike)
    qvals = np.linspace(0.001,0.999,1000)
    errvals = np.zeros(len(qvals))
    ppmperamuvals = np.zeros(len(qvals))
    
    if type_ == "pure":
        spikevector1 = np.zeros(isodata.nisos())
        spikevector1[isospike[0]] = 1.0
        spikevector2 = np.zeros(isodata.nisos())
        spikevector2[isospike[1]] = 1.0
    else:
        spikevector1 = isodata.rawspike[isospike[0], :]
        spikevector2 = isodata.rawspike[isospike[1], :]
    
    for i in range(len(qvals)):
        spike = qvals[i] * spikevector1 + (1-qvals[i]) * spikevector2
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
    """Find the optimal double spike compositions and plot the corresponding error curves.
    
    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
                This is the only mandatory argument.
        type_ (str): type of spike, 'pure' or 'real'. Real spikes, such as those from
            Oak Ridge National Labs, contain impurities. See isodata.rawspike for 
            their assumed compositions. By default pure spikes are used.
        isospike (array): the isotopes used in the double spike e.g. [54, 57].
            By default all choices of 2 isotopes are tried.
        isoinv (array): the isotopes used in the inversion, e.g. [54, 56, 57, 58].
        errorratio (array): by default, the optimal spike is chosen as that which
            minimises the error on the natural fractionation factor (known as
            alpha). Instead, the optimiser can be told to minimise the
            error on a particular ratio by setting errorratio. e.g.
            setting errorratio=[58, 56] will minimise the error on 58Fe/56Fe.
        alpha, beta (float): there is a small dependance of the error on the fractionation
            factors (natural and instrumental). Values of alpha and
            beta can be set here if desired, although the effect on the optimal spikes
            is slight unless the fractionations are very large.
        plottype (str): by default, the error is plotted. By setting this to 'ppmperamu'
            an estimate of the ppm per amu is plotted instead.
        **kwargs: additional arguments are passed to the plot command
    
    Example:
        >>> errorcurveoptimalspike(IsoData('Fe'))
    
    See also optimalspike, errorcurve
    """
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
    os = optimalspike(isodata,type_,isospike,isoinv,errorratio,alpha,beta)
    isolabel = isodata.isolabel()
    rawspikelabel = isodata.rawspikelabel()
    for j in range(os['optspike'].shape[0]):
        if type_ == 'pure':
            spiked = np.where(os['optspike'][j,:] > 0)[0]
            leglabel = isolabel[spiked[0]]+'-'+isolabel[spiked[1]]
        else:
            spiked = np.where(os['optspikeprop'][j,:] > 0)[0]
            leglabel =rawspikelabel[spiked[0]]+'-'+rawspikelabel[spiked[1]]
        errorcurve(isodata,os['optspike'][j,:],isoinv,errorratio,alpha,beta,plottype,label=leglabel)
        
    if plottype=='ppmperamu':
        plt.ylim(np.array([0,5 * np.amin(os['optppmperamu'])]))
    else:
        plt.ylim(np.array([0,5 * np.amin(os['opterr'])]))
    
    plt.legend(loc='upper right')

if __name__=="__main__":
    from .isodata import IsoData
    idat = IsoData('Fe')    
    errorcurveoptimalspike(idat,'real')
    
    plt.show()
    
