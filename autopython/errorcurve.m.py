import numpy as np
import matplotlib.pyplot as plt
    
def errorcurve(element = None,spike = None,isoinv = None,errorratio = None,alpha = None,beta = None,plottype = None,varargin = None): 
    #ERRORCURVE    A plot of error as a function of double spike-sample proportions for a given double spike composition
#  ERRORCURVE(element,spike,isoinv,errorratio,alpha,beta,...)
#             element -- element used in double spike, e.g. 'Fe'
#             spike -- the composition of the double spike as a composition vector e.g. [0 0 0.5 0.5]
#                represents a 50-50 mixture of the third and fourth isotopes (57-58 for Fe).
#             isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
#                By default the first four isotopes are chosen.
#             errorratio -- by default, the error on the natural fractionation
#                factor (known as alpha) is given. Instead, the error on a
#                 particular ratio can be given by setting errorratio. e.g.
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
#    errorcurve('Fe',[0 0 0.5 0.5])
    
    # See also errorestimate, errorcurve2
    global ISODATA
    if len(ISODATA)==0:
        dsstartup
    
    if (len(varargin) < 7) or len(plottype)==0:
        plottype = 'default'
    
    if (len(varargin) < 6) or len(beta)==0:
        beta = 0
    
    if (len(varargin) < 5) or len(alpha)==0:
        alpha = 0
    
    if (len(varargin) < 4) or len(errorratio)==0:
        errorratio = []
    
    if (len(varargin) < 3) or len(isoinv)==0:
        isoinv = np.array([1,2,3,4])
    
    rawdata = getattr(ISODATA,(element))
    spike = spike / sum(spike)
    # Convert isotope mass numbers to index numbers
    errorratio = rawdata.isoindex(errorratio)
    isoinv = rawdata.isoindex(isoinv)
    pvals = np.linspace(0.001,0.999,1000)
    errvals = np.zeros((pvals.shape,pvals.shape))
    ppmperamuvals = np.zeros((pvals.shape,pvals.shape))
    for i in np.arange(1,len(pvals)+1).reshape(-1):
        errvals[i],ppmperamuvals[i] = errorestimate(element,pvals(i),spike,isoinv,errorratio,alpha,beta)
    
    if plottype=='ppmperamu':
        plotvals = ppmperamuvals
    else:
        plotvals = errvals
    
    plt.plot(pvals,plotvals,varargin[:])
    mine = np.amin(plotvals)
    plt.xlim(np.array([0,1]))
    plt.ylim(np.array([0,5 * mine]))
    plt.xlabel('proportion of double spike in double spike-sample mix')
    if len(errorratio)==0:
        plt.ylabel('Error in \alpha (1SD)')
    else:
        plt.ylabel(np.array(['Error in ',rawdata.isolabel[errorratio(1)],'/',rawdata.isolabel[errorratio(2)],' (1SD)']))
    
    plt.title(np.array([rawdata.isolabel[isoinv(1)],', ',rawdata.isolabel[isoinv(2)],', ',rawdata.isolabel[isoinv(3)],', ',rawdata.isolabel[isoinv(4)],' inversion']))