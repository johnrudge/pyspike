import numpy as np
    
def errorestimate(element = None,prop = None,spike = None,isoinv = None,errorratio = None,alpha = None,beta = None): 
    #ERRORESTIMATE Calculates the error in the natural fractionation factor or a chosen ratio by linear error propagation
#    ERRORESTIMATE(element,prop,spike,isoinv,errorratio,alpha,beta)
#             element -- element used in double spike, e.g. 'Fe'
#             prop -- proportion of double spike in double spike-sample mix.
#             spike -- the isotopic composition of the spike e.g. [0 0.5 0 0.5]
#                corresponds to a 50-50 mixture of the 2nd and 4th isotopes
#                (56Fe and 58Fe) in the case of Fe.
#             isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
#                By default the first 4 isotopes are used.
#             errorratio -- by default, the error on the natural fractionation
#                factor (known as alpha) is given. Instead, the error on a
#                particular ratio can be given by setting errorratio. e.g.
#                setting errorratio=[58 56] will give the error on 58Fe/56Fe.
#             alpha, beta -- there is a small dependance of the error on the fractionation
#                factors (instrumental and natural, or alpha and beta). Values of alpha and
#                beta can be set here if desired, although the effect on the optimal spikes
#                is slight unless the fractionations are very large. Default is zero.
    
    # Output: error -- the error on the fractionation factor, or the specified ratio.
#         ppmperamu -- the error converted to an approximate ppm per atomic mass unit
    
    # Note that a number of parameters are specified in the global variable ISODATA.
    
    # Example
#   error=errorestimate('Fe',0.5,[0 0.5 0 0.5])
    
    # See also dsstartup
    global ISODATA
    if len(ISODATA)==0:
        dsstartup
    
    if (len(varargin) < 7) or len(beta)==0:
        beta = 0
    
    if (len(varargin) < 6) or len(alpha)==0:
        alpha = 0
    
    if (len(varargin) < 5) or len(errorratio)==0:
        errorratio = []
    
    if (len(varargin) < 4) or len(isoinv)==0:
        isoinv = np.array([1,2,3,4])
    
    rawdata = getattr(ISODATA,(element))
    spike = spike / sum(spike)
    # Convert isotope mass numbers to index numbers
    errorratio = rawdata.isoindex(errorratio)
    isoinv = rawdata.isoindex(isoinv)
    m,ix = np.amax(spike(isoinv))
    
    d = isoinv(ix)
    n = isoinv(isoinv != d)
    isoinv = np.array([d,n])
    
    # Calculate ratios
    in_ = calcratioeddata(element,isoinv)
    
    in_.AT = spike(in_.Ani) / spike(in_.di)
    
    # Now calculate sample ratio, lambda etc
    in_.AN = np.multiply(in_.An,np.exp(- in_.AP * alpha))
    lambda_ = realproptoratioprop(np.array([prop,1 - prop]),np.array([[in_.AT],[in_.AN]]))
    lambda_ = lambda_(1)
    z = np.array([lambda_,alpha,beta])
    in_.AM = np.multiply(lambda_,in_.AT) + np.multiply((1 - lambda_),in_.AN)
    in_.Am = np.multiply(in_.AM,np.exp(in_.AP * beta))
    # Error propagation
    measured = np.ones((1,rawdata.nisos))
    measured[in_.Ani] = in_.Am
    measured = measured / sum(measured)
    #isonorm=isoinv;  # normalise so that only the sum of beams used in the inversion is the mean intensity
    isonorm = np.arange(1,rawdata.nisos+1)
    
    in_.VAn = calcratiocov(rawdata.standard,rawdata.errormodel.standard,in_.di,isonorm,prop)
    in_.VAT = calcratiocov(spike,rawdata.errormodel.spike,in_.di,isonorm,prop)
    in_.VAm = calcratiocov(measured,rawdata.errormodel.measured,in_.di,isonorm,prop)
    Vz,VAN = fcerrorpropagation(z,in_.AP,in_.An,in_.AT,in_.Am,in_.VAn,in_.VAT,in_.VAm,in_.srat)
    # Error to return
    if len(errorratio)==0:
        error = np.sqrt(Vz(2,2))
    else:
        # Now change coordinates to get variance of ratio we're interested in
        newVAN = changedenomcov(in_.AN,VAN,in_.di,errorratio(2))
        isonums = np.arange(1,getattr(ISODATA,(element)).nisos+1)
        newAni = isonums(isonums != errorratio(2))
        erat = find(errorratio(1) == newAni)
        error = np.sqrt(newVAN(erat,erat))
    
    if (len(errorratio)==0):
        ppmperamu = (1000000.0 * error) / mean(getattr(ISODATA,(element)).mass)
    else:
        stdratio = getattr(ISODATA,(element)).standard(errorratio(1)) / getattr(ISODATA,(element)).standard(errorratio(2))
        massdiff = np.abs(getattr(ISODATA,(element)).mass(errorratio(1)) - getattr(ISODATA,(element)).mass(errorratio(2)))
        ppmperamu = (1000000.0 * error) / (stdratio * massdiff)
    
    return error,ppmperamu