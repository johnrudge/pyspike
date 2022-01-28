import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
    
def errorcurveoptimalspike(element = None,type_ = None,isospike = None,isoinv = None,errorratio = None,alpha = None,beta = None,plottype = None,varargin = None): 
    #ERRORCURVEOPTIMALSPIKE    Find the optimal double spike compositions and plot the corresponding error curves
# [optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu]
#  =ERRORCURVEOPTIMALSPIKE(element,type,isoinv,isospike,errorratio,alpha,beta,...)
#             element -- element used in double spike, e.g. 'Fe'
#                This is the only mandatory argument.
#             type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
#                Oak Ridge National Labs, contain impurities (see 'maininput.csv'
#                or ISODATA.(element).rawspike) for their assumed compositions.
#                By default pure spikes are used.
#             isospike -- the isotopes used in the double spike e.g. [54 57].
#                By default all choices of 2 isotopes are tried.
#             isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
#                By default the first four isotopes are used.
#             errorratio -- by default, the optimal spike is chosen as that which
#                minimises the error on the natural fractionation factor (known as
#                alpha). Instead, the optimiser can be told to minimise the
#                error on a particular ratio by setting errorratio. e.g.
#                setting errorratio=[58 56] will minimise the error on 58Fe/56Fe.
#             alpha, beta -- there is a small dependance of the error on the fractionation
#                factors (natural and instrumental). Values of alpha and
#                beta can be set here if desired, although the effect on the optimal spikes
#                is slight unless the fractionations are very large. Default is zero.
#             plottype -- by default, the error is plotted. By setting this to 'ppmperamu'
#                an estimate of the ppm per amu is plotted instead.
#             ... -- additional arguments are passed to the plot command
    
    # Outputs are the same as those of optimalspike.
    
    # Example
#   errorcurveoptimalspike('Fe')
    
    # See also optimalspike, errorcurve
    
    global ISODATA
    # Have some default arguments
    if len(ISODATA)==0:
        dsstartup
    
    if (len(varargin) < 8) or len(plottype)==0:
        plottype = 'default'
    
    if (len(varargin) < 7) or len(beta)==0:
        beta = 0
    
    if (len(varargin) < 6) or len(alpha)==0:
        alpha = 0
    
    if (len(varargin) < 5) or len(errorratio)==0:
        errorratio = []
    
    if (len(varargin) < 4) or len(isoinv)==0:
        isoinv = np.array([1,2,3,4])
    
    if (len(varargin) < 3) or len(isospike)==0:
        isospike = []
    
    if (len(varargin) < 2) or len(type_)==0:
        type_ = 'pure'
    
    rawdata = getattr(ISODATA,(element))
    # Convert isotope mass numbers to index numbers
    errorratio = rawdata.isoindex(errorratio)
    isospike = rawdata.isoindex(isospike)
    isoinv = rawdata.isoindex(isoinv)
    # Find the optimal spikes
    optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike(element,type_,isospike,isoinv,errorratio,alpha,beta)
    cols = np.matlib.repmat('brgcmky',1,optspike.shape[1-1])
    for j in np.arange(1,optspike.shape[1-1]+1).reshape(-1):
        if str(type_) == str('pure'):
            spiked = find(optspike(j,:) > 0)
            leglabel = np.array([rawdata.isolabel[spiked(1)],'-',rawdata.isolabel[spiked(2)]])
        else:
            spiked = find(optspikeprop(j,:) > 0)
            leglabel = np.array([rawdata.rawspikelabel[spiked(1)],'-',rawdata.rawspikelabel[spiked(2)]])
            #leglabel=['rawspikes ' num2str(spiked(1)) '-' num2str(spiked(2))];
#  		if (size(optspikeprop,2)==2)
#  			leglabel=['rawspikes ' num2str(isospike(1)) '-' num2str(isospike(2))];
#  		else
#  			leglabel=['best real spike'];
#  		end
        #errorcurve(element,alpha,beta,optspike(j,:),errorratio,isoinv,cols(j),'DisplayName',leglabel);
        errorcurve(element,optspike(j,:),isoinv,errorratio,alpha,beta,plottype,cols(j),varargin[:],'DisplayName',leglabel)
        hold('on')
    
    if plottype=='ppmperamu':
        plt.ylim(np.array([0,5 * np.amin(optppmperamu)]))
    else:
        plt.ylim(np.array([0,5 * np.amin(opterr)]))
    
    hold('off')
    plt.legend('show')
    return optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu