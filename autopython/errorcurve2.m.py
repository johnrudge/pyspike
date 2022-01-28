import numpy as np
import matplotlib.pyplot as plt
    
def errorcurve2(element = None,type_ = None,prop = None,isospike = None,isoinv = None,errorratio = None,alpha = None,beta = None,plottype = None,varargin = None): 
    #ERRORCURVE2    A plot of error as a function of double spike proportions for a given double spike-sample proportion
#  ERRORCURVE2(element,type,prop,isospike,isoinv,errorratio,alpha,beta,...)
#             element -- element used in double spike, e.g. 'Fe'
#             type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
#                Oak Ridge National Labs, contain impurities (see 'data/maininput.csv'
#                or ISODATA.(element).rawspike) for their assumed compositions.
#                By default pure spikes are used.
#             prop -- proportion of double spike in double spike-sample mixture e.g. 0.5.
#             isospike -- the isotopes used in the double spike e.g. [54 57].
#                By default the first two isotopes are chosen.
#             isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
#                By default the first four isotopes are chosen.
#             errorratio -- by default, the error on the natural fractionation
#                factor (known as alpha or alpha) is given. Instead, the
#                error on a particular ratio can be given by setting errorratio. e.g.
#                setting errorratio=[58 56] will give the error on 58Fe/56Fe.
#             alpha, beta -- there is a small dependance of the error on the fractionation
#                factors (natural and instrumental). Values of beta and
#                alpha can be set here if desired, although the effect on the optimal spikes
#                is slight unless the fractionations are very large. Default is zero.
#             plottype -- by default, the error is plotted. By setting this to 'ppmperamu'
#                an estimate of the ppm per amu is plotted instead.
#             ... -- additional arguments are passed to the plot command.
    
    # Note that a number of parameters are specified in the global variable ISODATA.
    
    # Example
#    errorcurve2('Fe','real',0.5,[54 57])
    
    # See also errorestimate, errorcurve
    global ISODATA
    if len(ISODATA)==0:
        dsstartup
    
    if (len(varargin) < 9) or len(plottype)==0:
        plottype = 'default'
    
    if (len(varargin) < 8) or len(beta)==0:
        beta = 0
    
    if (len(varargin) < 7) or len(alpha)==0:
        alpha = 0
    
    if (len(varargin) < 6) or len(errorratio)==0:
        errorratio = []
    
    if (len(varargin) < 5) or len(isoinv)==0:
        isoinv = np.array([1,2,3,4])
    
    if (len(varargin) < 4) or len(isospike)==0:
        isospike = np.array([1,2])
    
    if (len(varargin) < 3) or len(prop)==0:
        prop = 0.5
    
    if (len(varargin) < 2) or len(type_)==0:
        type_ = 'pure'
    
    rawdata = getattr(ISODATA,(element))
    rawspike = rawdata.rawspike
    # Convert isotope mass numbers to index numbers
    errorratio = rawdata.isoindex(errorratio)
    isoinv = rawdata.isoindex(isoinv)
    isospike = rawdata.isoindex(isospike)
    qvals = np.linspace(0.001,0.999,1000)
    errvals = np.zeros((qvals.shape,qvals.shape))
    ppmperamuvals = np.zeros((qvals.shape,qvals.shape))
    for i in np.arange(1,len(qvals)+1).reshape(-1):
        spike = (np.multiply(qvals(i),rawspike(isospike(1),:))) + (np.multiply((1 - qvals(i)),rawspike(isospike(2),:)))
        errvals[i],ppmperamuvals[i] = errorestimate(element,prop,spike,isoinv,errorratio,alpha,beta)
    
    if plottype=='ppmperamu':
        plotvals = ppmperamuvals
    else:
        plotvals = errvals
    
    plt.plot(qvals,plotvals,varargin[:])
    mine = np.amin(plotvals)
    plt.xlim(np.array([0,1]))
    plt.ylim(np.array([0,5 * mine]))
    if str(type_) == str('pure'):
        plt.xlabel(np.array(['proportion of ',rawdata.isolabel[isospike(1)],' in ',rawdata.isolabel[isospike(1)],'-',rawdata.isolabel[isospike(2)],' double spike']))
    else:
        plt.xlabel('proportion of first rawspike in double spike')
    
    if len(errorratio)==0:
        plt.ylabel('Error in \alpha (1SD)')
    else:
        plt.ylabel(np.array(['Error in ',rawdata.isolabel[errorratio(1)],'/',rawdata.isolabel[errorratio(2)],' (1SD)']))
    
    plt.title(np.array([rawdata.isolabel[isoinv(1)],', ',rawdata.isolabel[isoinv(2)],', ',rawdata.isolabel[isoinv(3)],', ',rawdata.isolabel[isoinv(4)],' inversion']))