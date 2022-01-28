import numpy as np
import numpy.matlib
    
def optimalpurespike(element = None,beta = None,alpha = None,errorratio = None,isospike = None,isoinv = None): 
    #OPTIMALPURESPIKE    Finds the best pure spike
#    OPTIMALPURESPIKE(rawdata,beta,alpha,errorratio,isospike,isoinv)
#             rawdata -- data about a particular element
#             beta -- instrumental fractionation
#             alpha -- natural fractionation
#             errorratio -- the ratio whose error we are targeting
#             isospike -- the isotopes to spike
#             isoinv -- the isotopes used in the inversion
    global ISODATA
    rawdata = getattr(ISODATA,(element))
    # Have some default arguments
    if (len(varargin) < 6) or len(isoinv)==0:
        isoinv = []
    
    if (len(varargin) < 5) or len(isospike)==0:
        isospike = []
    
    if (len(varargin) < 4) or len(errorratio)==0:
        errorratio = []
    
    if (len(varargin) < 3) or len(alpha)==0:
        alpha = 0
    
    if (len(varargin) < 2) or len(beta)==0:
        beta = 0
    
    # Convert isotope mass numbers to index numbers
    errorratio = rawdata.isoindex(errorratio)
    isospike = rawdata.isoindex(isospike)
    isoinv = rawdata.isoindex(isoinv)
    if (len(isoinv)==0):
        isoinv = combnk(np.arange(1,rawdata.nisos+1),4)
    
    isoinvvals = []
    isospikevals = []
    for i in np.arange(1,isoinv.shape[1-1]+1).reshape(-1):
        if len(isospike)==0:
            isospikev = combnk(isoinv(i,:),2)
        else:
            if len(intersect(isospike,isoinv(i,:))) == 2:
                isospikev = isospike
            else:
                isospikev = []
        isospikevals = np.array([[isospikevals],[isospikev]])
        isoinvvals = np.array([[isoinvvals],[np.matlib.repmat(isoinv(i,:),isospikev.shape[1-1],1)]])
    
    for i in np.arange(1,isoinvvals.shape[1-1]+1).reshape(-1):
        optspike[i,:],optprop[i,:],opterr[i,:],optppmperamu[i,:] = singlepureoptimalspike(element,beta,alpha,errorratio,isospikevals(i,:),isoinvvals(i,:))
    
    optisoinv = isoinvvals
    # Sort in ascending order of error
    opterr,ix = __builtint__.sorted(opterr)
    optppmperamu = optppmperamu(ix,:)
    optspike = optspike(ix,:)
    optprop = optprop(ix,:)
    optisoinv = optisoinv(ix,:)
    optisoinv = rawdata.isonum(optisoinv)
    #optspikeprop=optspikeprop(ix,:);
    optspikeprop = optspike
    
def singlepureoptimalspike(element = None,beta = None,alpha = None,errorratio = None,isospike = None,isoinv = None): 
    # Calculate the composition of the optimal double spike given the isotopes used in the inversion
# and of those the isotopes we are spiking
    global ISODATA
    rawdata = getattr(ISODATA,(element))
    spikevector1 = np.zeros((1,rawdata.nisos))
    spikevector1[isospike[1]] = 1
    spikevector2 = np.zeros((1,rawdata.nisos))
    spikevector2[isospike[2]] = 1
    if (verLessThan('optim','4.0')):
        options = optimset('Display','notify','TolX',1e-08,'TolFun',1e-10,'TolCon',1e-06,'LargeScale','off','MaxFunEvals',10000)
    else:
        options = optimset('Display','notify','TolX',1e-08,'TolFun',1e-10,'TolCon',1e-06,'Algorithm','active-set','MaxFunEvals',10000)
    
    tol = 2e-05
    
    lb = np.array([[tol],[tol]])
    ub = np.array([[1 - tol],[1 - tol]])
    y0 = np.transpose(np.array([0.5,0.5]))
    # Helpful to rescale the error, to make everything roughly order 1 for the optimiser
    initialerror = errorestimate(element,y0(1),np.multiply(y0(2),spikevector1) + np.multiply((1 - y0(2)),spikevector2),isoinv,errorratio,beta,alpha)
    y,opterr = fmincon(lambda y = None: errorestimate(element,y(1),np.multiply(y(2),spikevector1) + np.multiply((1 - y(2)),spikevector2),isoinv,errorratio,beta,alpha) / initialerror,y0,[],[],[],[],lb,ub,[],options)
    opterr,optppmperamu = errorestimate(element,y(1),np.multiply(y(2),spikevector1) + np.multiply((1 - y(2)),spikevector2),isoinv,errorratio,beta,alpha)
    optprop = y(1)
    optspike = y(2) * spikevector1 + (1 - y(2)) * spikevector2
    return optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu