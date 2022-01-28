import numpy as np
import matplotlib.pyplot as plt
    
def errorcurve2d(element = None,type_ = None,isospike = None,isoinv = None,errorratio = None,alpha = None,beta = None,resolution = None,threshold = None,ncontour = None,plottype = None,varargin = None): 
    #ERRORCURVE2D    A 2D contour plot of error as a function of double spike composition and double spike-sample proportions
#  ERRORCURVE2D(element,type,isospike,isoinv,errorratio,alpha,beta,resolution,threshold,ncontour,plottype,...)
#             element -- element used in double spike, e.g. 'Fe'
#             type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
#                Oak Ridge National Labs, contain impurities (see 'maininput.csv'
#                or ISODATA.(element).rawspike) for the assumed compositions.
#                By default pure spikes are used.
#             isospike -- the isotopes used in the double spike e.g. [54 57].
#                By default the first two isotopes are chosen.
#             isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
#                By default the first four isotopes are chosen.
#             errorratio -- by default, the error on the natural fractionation
#                factor (known as alpha) is given. Instead, the error on a
#                 particular ratio can be given by setting errorratio. e.g.
#                setting errorratio=[58 56] will give the error on 58Fe/56Fe.
#             alpha, beta -- there is a small dependance of the error on the fractionation
#                factors (natural and instrumental). Values of alpha and
#                beta can be set here if desired, although the effect on the optimal spikes
#                is slight unless the fractionations are very large. Default is zero.
#             resolution -- number of grid points in x and y. Default is 100.
#             threshold -- maximum contour to plot, relative to the minimum error.
#                Default is 0.25 i.e. 25# in excess of the minimum.
#             ncontour -- number of countours. Default is 25.
#             plottype -- by default, the error is plotted. By setting this to 'ppmperamu'
#                an estimate of the ppm per amu is plotted instead.
#             ... -- additional arguments are passed to contour command.
    
    # Note that a number of parameters are specified in the global variable ISODATA.
    
    # Example
#    errorcurve2d('Fe','pure',[57 58])
    
    # See also errorestimate, contour
    global ISODATA
    # Set some default values
    if len(ISODATA)==0:
        dsstartup
    
    if (len(varargin) < 11) or len(plottype)==0:
        plottype = 'default'
    
    if (len(varargin) < 10) or len(ncontour)==0:
        ncontour = 25
    
    if (len(varargin) < 9) or len(threshold)==0:
        threshold = 0.25
    
    if (len(varargin) < 8) or len(resolution)==0:
        resolution = 100
    
    if (len(varargin) < 7) or len(beta)==0:
        beta = 0
    
    if (len(varargin) < 6) or len(alpha)==0:
        alpha = 0
    
    if (len(varargin) < 5) or len(errorratio)==0:
        errorratio = []
    
    if (len(varargin) < 4) or len(isoinv)==0:
        isoinv = np.array([1,2,3,4])
    
    if (len(varargin) < 3) or len(isospike)==0:
        isospike = np.array([1,2])
    
    if (len(varargin) < 2) or len(type_)==0:
        type_ = 'pure'
    
    rawdata = getattr(ISODATA,(element))
    # Convert isotope mass numbers to index numbers
    errorratio = rawdata.isoindex(errorratio)
    isoinv = rawdata.isoindex(isoinv)
    isospike = rawdata.isoindex(isospike)
    if str(type_) == str('pure'):
        spike1 = np.zeros((1,rawdata.nisos))
        spike2 = np.zeros((1,rawdata.nisos))
        spike1[isospike[1]] = 1
        spike2[isospike[2]] = 1
    else:
        spike1 = rawdata.rawspike(isospike(1),:)
        spike2 = rawdata.rawspike(isospike(2),:)
    
    prop = np.linspace(0.001,0.999,resolution)
    spikeprop = np.linspace(0.001,0.999,resolution)
    iv,jv = np.meshgrid(np.arange(1,len(prop)+1),np.arange(1,len(spikeprop)+1))
    errvals,ppmperamuvals = arrayfun(lambda i = None,j = None: errorestimate(element,prop(i),(np.multiply(spikeprop(j),spike1) + np.multiply((1 - spikeprop(j)),spike2)),isoinv,errorratio,alpha,beta),iv,jv)
    optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike(element,type_,isospike,isoinv,errorratio,alpha,beta)
    #pcolor(prop,spikeprop,errvals);
#shading flat;
#caxis([opterr (1+threshold)*opterr]);
    
    if str(plottype) == str('ppmperamu'):
        C = plt.contour(prop,spikeprop,ppmperamuvals,np.linspace(optppmperamu,(1 + threshold) * optppmperamu,ncontour + 1),varargin[:])
    else:
        C = plt.contour(prop,spikeprop,errvals,np.linspace(opterr,(1 + threshold) * opterr,ncontour + 1),varargin[:])
    
    plt.xlim(np.array([0,1]))
    plt.ylim(np.array([0,1]))
    plt.xlabel('proportion of double spike in double spike-sample mix')
    if str(type_) == str('pure'):
        plt.ylabel(np.array(['proportion of ',rawdata.isolabel[isospike(1)],' in ',rawdata.isolabel[isospike(1)],'-',rawdata.isolabel[isospike(2)],' double spike']))
    else:
        #	ylabel('proportion of first rawspike in double spike');
        plt.ylabel(np.array(['proportion of ',rawdata.rawspikelabel[isospike(1)],' in ',rawdata.rawspikelabel[isospike(1)],'-',rawdata.rawspikelabel[isospike(2)],' double spike']))
    
    invisostring = np.array([rawdata.isolabel[isoinv(1)],', ',rawdata.isolabel[isoinv(2)],', ',rawdata.isolabel[isoinv(3)],', ',rawdata.isolabel[isoinv(4)],' inversion'])
    if len(errorratio)==0:
        plt.title(np.array(['Error in \alpha (1SD) with ',invisostring]))
    else:
        plt.title(np.array(['Error in ',rawdata.isolabel[errorratio(1)],'/',rawdata.isolabel[errorratio(2)],' (1SD) with ',invisostring]))
    
    hold('on')
    plt.plot(optprop(1),optspikeprop(isospike(1)),'rx')
    hold('off')