import numpy as np
import itertools
from scipy.optimize import minimize, LinearConstraint
from scipy.special import expit, logit

from .inversion import realproptoratioprop, ratio

def errorestimate(isodata, prop = None, spike = None, isoinv = None, errorratio = None, alpha = 0.0, beta = 0.0): 
    """ Calculates the error in the natural fractionation factor or a chosen ratio by linear error propagation

            isodata -- object of class IsoData, e.g. IsoData('Fe')
            prop -- proportion of double spike in double spike-sample mix.
            spike -- the isotopic composition of the spike e.g. [0 0.5 0 0.5]
               corresponds to a 50-50 mixture of the 2nd and 4th isotopes
               (56Fe and 58Fe) in the case of Fe.
            isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
               By default the first 4 isotopes are used.
            errorratio -- by default, the error on the natural fractionation
               factor (known as alpha) is given. Instead, the error on a
               particular ratio can be given by setting errorratio. e.g.
               setting errorratio=[58 56] will give the error on 58Fe/56Fe.
            alpha, beta -- there is a small dependance of the error on the fractionation
               factors (instrumental and natural, or alpha and beta). Values of alpha and
               beta can be set here if desired, although the effect on the optimal spikes
               is slight unless the fractionations are very large. Default is zero.
    
    Output: error -- the error on the fractionation factor, or the specified ratio.
        ppmperamu -- the error converted to an approximate ppm per atomic mass unit
    
    Note that a number of parameters are specified in the global variable ISODATA.
    
    Example
        error=errorestimate('Fe',0.5,[0 0.5 0 0.5])"""

    # Get data from isodata if not supplied as arguments
    if spike is None:
        if isodata.spike is None:
            raise Exception("No spike given")
        else:
            spike = isodata.spike

    if isoinv is None:
        if hasattr(isodata, 'isoinv'):
            isoinv = isodata.isoinv
        else:
            isoinv = isodata.isonum[0:4]
    
    standard = isodata.standard
    spike = spike / sum(spike)
    
    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isoinv = isodata.isoindex(isoinv)
    
    # Choose denominator from largest spike value
    ix = np.argmax(spike[isoinv])
    deno = isoinv[ix]
    nume = isoinv[isoinv != deno]
    isoinv = np.concatenate((np.array([deno]),nume))
    
    isonum = np.arange(isodata.nisos())
    isonum = isonum[isonum != isoinv[0]]
    isonum = np.concatenate((np.array([isoinv[0]]), isonum))
    
    # Calculate ratios
    AP = np.log(ratio(isodata.mass, isonum))
    AT = ratio(spike, isonum)
    An = ratio(standard, isonum)
    
    # Now calculate sample ratio, lambda etc
    AN = An * np.exp(-AP*alpha)
    lambda_ = realproptoratioprop(prop, AT, AN)
    z = np.array([lambda_,alpha,beta])
    AM = lambda_*AT + (1-lambda_)*AN
    Am = AM * np.exp(AP*beta)
    
    # Error propagation
    measured = np.ones(isodata.nisos())
    measured[isonum[1:]] = Am
    measured = measured / sum(measured)
    isonorm = np.arange(isodata.nisos()) # normalise so that the sum of all beams is the mean intensity
    
    di = deno
    VAn = calcratiocov(isodata.standard,isodata.errormodel['standard'],di,isonorm,prop)
    VAT = calcratiocov(spike,isodata.errormodel['spike'],di,isonorm,prop)
    VAm = calcratiocov(measured,isodata.errormodel['measured'],di,isonorm,prop)
    
    # srat gives indices of ratios used in inversion
    srat = np.array([np.where(isonum == i)[0][0] for i in isoinv])
    srat = srat[1:] - 1
    
    Vz, VAN, _ = fcerrorpropagation(z,AP,An,AT,Am,VAn,VAT,VAm,srat)

    # Error to return
    if errorratio is None:
        error = np.sqrt(Vz[1,1])
    else:
        # Now change coordinates to get variance of ratio we're interested in
        newVAN = changedenomcov(AN,VAN,di,errorratio[1])
        isonums = np.arange(isodata.nisos())
        newAni = isonums[isonums != errorratio[1]]
        erat = np.where(errorratio[0] == newAni)[0][0]
        error = np.sqrt(newVAN[erat,erat])
    
    if errorratio is None:
        ppmperamu = (1000000.0 * error) / np.mean(isodata.mass)
    else:
        stdratio = isodata.standard[errorratio[0]] / isodata.standard[errorratio[1]]
        massdiff = np.abs(isodata.mass[errorratio[0]] - isodata.mass[errorratio[1]])
        ppmperamu = (1000000.0 * error) / (stdratio * massdiff)
    
    return error, ppmperamu 

def calcratiocov(composition = None, errormodel = None, di = None, isonorm = None, prop = 0.0): 
    # Calculate the covariance matrix of the ratios based on the given error model and composition
    # di is the isotope with which to denominator
    # isonorm are the isotopes to use in the normalisation
    # prop is the proportion of spike in the spike-sample mix
    
    if isonorm is None:
        isonorm = np.arange(composition.shape[0])
    
    # first normalise composition so it is really a composition (unit sum)
    composition = composition / sum(composition)
    
    meanbeams = errormodel['intensity']*composition / sum(composition[isonorm])
    
    if errormodel['type']=='fixed-sample':
        meanbeams = meanbeams / (1.0 - prop)

    covbeams = calcbeamcov(meanbeams,errormodel)
    V = covbeamtoratio(meanbeams,covbeams,di)
    return V
    
def calcbeamcov(meanbeams = None,errormodel = None): 
    # the beam covariance matrix
    beamvar = errormodel['a'] + errormodel['b']*meanbeams + errormodel['c']*(meanbeams ** 2)
    return np.diag(beamvar)
    
def covbeamtoratio(meanbeams = None,covbeams = None,di = None): 
    # converts a covariance matrix for beams to one for ratios
    # di is the isotope to denominator with
    # assumes last row and column of M correspond to denominator
    isonums = np.arange(len(meanbeams))
    ni = isonums[isonums != di]
    n = meanbeams[ni]
    d = meanbeams[di]
    ii = np.concatenate((ni, np.array([di]))) # move denominator to end
    M = covbeams[ii, :][:,ii]
    
    D = np.diag(1/d * np.ones(len(n)))
    S = - np.transpose(n) / (d ** 2)
    A = np.hstack((D, S[:,np.newaxis]))
    
    V = (A @ M) @ (np.transpose(A))
    return V

def changedenomcov(data = None, datacov = None, olddi = None, newdi = None): 
    # change denominator of covariance matrix for given set of ratios
    
    nisos = len(data) + 1
    oldni = np.concatenate((np.arange(olddi),np.arange(olddi+1,nisos)))
    dataplus = np.concatenate((data[0:olddi],np.array([1]),data[olddi:]))  
    
    newni = np.concatenate((np.arange(newdi),np.arange(newdi+1,nisos)))
    
    datacovplus = np.zeros((nisos,nisos))
    datacovplus[:,oldni][oldni,:] = datacov
    A = np.eye(nisos) / dataplus[newdi]
    A[:,newdi] = A[:,newdi] - dataplus.T / (dataplus[newdi] ** 2)
    newdatacovplus = A @ datacovplus @ A.T
    newdatacov = newdatacovplus[:,newni][newni,:]
    return newdatacov


def fcerrorpropagation(z,AP,An,AT,Am,VAn,VAT,VAm,srat): 
    """linear error propagation for the fractionation correction"""
    
    lambda_ = z[0]
    alpha = z[1]
    beta = z[2]
    AM = Am * np.exp(- AP * beta)
    AN = An * np.exp(- AP * alpha)
    
    # Select appropriate ratios
    P = AP[srat]
    N = AN[srat]
    T = AT[srat]
    M = AM[srat]
    VT = VAT[srat,:][:,srat]
    Vm = VAm[srat,:][:,srat]
    Vn = VAn[srat,:][:,srat]
    
    # calculate various Jacobian matrices
    dfdlambda = T - N*(1 + alpha*P)
    dfdu = -N*P
    dfdbeta = M*P
    dfdy = np.array([dfdlambda,dfdu,dfdbeta]).T
    dfdT = lambda_*np.eye(3)
    dfdm = - np.diag(np.exp(- beta * P))
    dfdn = (1 - lambda_) * np.diag(np.exp(- alpha * P))
    
    ## matrix to convert from (lambda, (1-lambda)alpha,beta) to (lambda,alpha,beta)
    K = np.array([[1,0,0],
                  [(alpha / (1 - lambda_)),(1 / (1 - lambda_)),0],
                  [0,0,1]])
    dzdT = - K @ (np.linalg.solve(dfdy,dfdT))
    dzdm = - K @ (np.linalg.solve(dfdy,dfdm))
    dzdn = - K @ (np.linalg.solve(dfdy,dfdn))
    
    # Covariance matix for (lambda,beta,alpha)
    Vz = dzdT @ VT @ dzdT.T + dzdm @ Vm @ dzdm.T + dzdn @ Vn @ dzdn.T
    
    # full matrices for all ratios
    nratios = len(An)
    dzdAT = np.zeros((3,nratios))
    dzdAn = np.zeros((3,nratios))
    dzdAm = np.zeros((3,nratios))
    dzdAT[0:3,:][:,srat] = dzdT
    dzdAn[0:3,:][:,srat] = dzdn
    dzdAm[0:3,:][:,srat] = dzdm
    
    # Covariance matrix of sample
    dalphadAT = dzdAT[1,:]
    dalphadAn = dzdAn[1,:]
    dalphadAm = dzdAm[1,:]
    
    NP = AN*AP
    NP = NP[:,np.newaxis]
    dANdAT = - NP @ dalphadAT[np.newaxis,:]
    dANdAn = np.diag(np.exp(np.multiply(- AP,alpha))) - NP @ dalphadAn[np.newaxis,:]
    dANdAm = - NP @ dalphadAm[np.newaxis,:]

    VAN = dANdAn @ VAn @ dANdAn.T + dANdAT @ VAT @ dANdAT.T + dANdAm @ VAm @ dANdAm.T
    
    # Covariance matrix of mixture
    dbetadAT = dzdAT[2,:]
    dbetadAn = dzdAn[2,:]
    dbetadAm = dzdAm[2,:]
    
    MP = AM*AP
    MP = MP[:,np.newaxis]
    dAMdAT = - MP @ dbetadAT[np.newaxis,:]
    dAMdAn = - MP @ dbetadAn[np.newaxis,:]
    dAMdAm = np.diag(np.exp(np.multiply(- beta,AP))) - MP @ dbetadAm[np.newaxis,:]
    VAM = dAMdAn @ VAn @ dAMdAn.T + dAMdAT @ VAT @ dAMdAT.T + dAMdAm @ VAm @ dAMdAm.T
    return Vz, VAN, VAM


def optimalspike(isodata,type_ = 'pure',isospike = None,isoinv = None,errorratio = None,alpha = 0.0,beta = 0.0): 
    """OPTIMALSPIKE    Find the optimal double spike composition and double spike-sample mixture proportions
    [optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu]
    =OPTIMALSPIKE(element,type,isospike,isoinv,errorratio,alpha,beta)
                element -- element used in double spike, e.g. 'Fe'
                This is the only mandatory argument.
                type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
                Oak Ridge National Labs, contain impurities (see 'data/maininput.csv'
                or ISODATA.(element).rawspike) for their assumed compositions.
                By default pure spikes are used.
                isospike -- the isotopes used in the double spike e.g. [54 57].
                By default all choices of 2 isotopes are tried.
                isoinv -- the isotopes used in the inversion, e.g. [54 56 57 58].
                By default all choices of 4 isotopes are tried.
                errorratio -- by default, the optimal double spike is chosen as that which
                minimises the error on the natural fractionation factor (known as
                alpha). Instead, the optimiser can be told to minimise the
                error on a particular ratio by setting errorratio. e.g.
                setting errorratio=[58 56] will minimise the error on 58Fe/56Fe.
                alpha, beta -- there is a small dependance of the error on the fractionation
                factors (instrumental and natural, or alpha and beta). Values of alpha and
                beta can be set here if desired, although the effect on the optimal spikes
                is slight unless the fractionations are very large. Default is zero.
        
        All the outputs are provided as matrices. Each column represents an isotope
    (see ISODATA.(element).isonum for the isotope numbers) e.g. for Fe the columns
    correspond to the isotopes 54Fe, 56Fe, 57Fe, 58Fe. The rows represent the
    different combinations of double spikes and isotopes being tried, in order of
    error: The first row is the best double spike, and the last row is the worst.
        optspike -- the proportions of each isotope in the optimal double spike.
        optprop -- the optimal proportion of spike in the double spike-sample mix.
        opterr -- the error in the fractionation factor (or ratio if specified)
                for the optimal spike.
        optisoinv -- the 4 isotopes used in the inversion.
        optspikeprop -- the proportion of each raw spike in the optimal double spike.
        optppmperamu -- an alternative expression of the error in terms of ppm per amu.
        
        Note that a number of parameters are specified in the global variable ISODATA.
        
        Example
    [optspike,optprop,opterr]=optimalspike('Fe')"""
    
    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    if type_ == 'pure':
        optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalpurespike(isodata,beta,alpha,errorratio,isospike,isoinv)
    else:
        optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalrealspike(isodata,beta,alpha,errorratio,isospike,isoinv)
    
    return optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu

def optimalpurespike(isodata,beta = 0.0,alpha = 0.0,errorratio = None,isospike = None,isoinv = None): 
    #OPTIMALPURESPIKE    Finds the best pure spike
#    OPTIMALPURESPIKE(isodata,beta,alpha,errorratio,isospike,isoinv)
#             isodata -- data about a particular element
#             beta -- instrumental fractionation
#             alpha -- natural fractionation
#             errorratio -- the ratio whose error we are targeting
#             isospike -- the isotopes to spike
#             isoinv -- the isotopes used in the inversion

    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isospike = isodata.isoindex(isospike)
    isoinv = isodata.isoindex(isoinv)
    
    # If don't specify inversion isotopes, do all possible combinations
    if isoinv is None:
        isoinv = list(itertools.combinations(np.arange(isodata.nisos()), 4))
    else:
        isoinv = list([isoinv])
    
    # Work out all combinations of inversion isotopes and spiking isotopes
    isoinvvals = []
    isospikevals = []
    for i in range(len(isoinv)):
        if isospike is None:
            #isospikev = combnk(isoinv(i,:),2)
            isospikev = list(itertools.combinations(isoinv[i], 2))
        else:
            if len(set(isospike).intersection(set(isoinv[i]))) == 2:
                isospikev = list([isospike])
            else:
                isospikev = None
        
        if isospikev is not None:
            isospikevals.append(isospikev)
            isoinvvals.append(np.tile(isoinv[i],(len(isospikev),1)))
    isoinvvals = np.vstack(isoinvvals)
    isospikevals = np.vstack(isospikevals)
   
    optspikes = []
    optprops = []
    opterrs = []
    optppmperamus = []
    
    for i in range(len(isoinvvals)):
        optspike,optprop,opterr,optppmperamu = singlepureoptimalspike(isodata,beta,alpha,errorratio,isospikevals[i,:],isoinvvals[i,:])
        optspikes.append(optspike)
        optprops.append(optprop)
        opterrs.append(opterr)
        optppmperamus.append(optppmperamu)
    
    optspike = np.vstack(optspikes)
    optprop = np.array(optprops)
    opterr = np.array(opterrs)
    optppmperamu = np.array(optppmperamus)
    optisoinv = isoinvvals
    
    ## Sort in ascending order of error
    ix = np.argsort(opterr)
    opterr = opterr[ix]
    optppmperamu = optppmperamu[ix]
    optspike = optspike[ix,:]
    optprop = optprop[ix]
    optisoinv = optisoinv[ix,:]
    optisoinv = isodata.isonum[optisoinv]
    optspikeprop = optspike
    
    return optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu
    
def singlepureoptimalspike(isodata,beta = 0.0,alpha = 0.0,errorratio = None,isospike = None,isoinv = None): 
    # Calculate the composition of the optimal double spike given the isotopes used in the inversion
    # and of those the isotopes we are spiking
    
    spikevector1 = np.zeros(isodata.nisos())
    spikevector1[isospike[0]] = 1.0
    spikevector2 = np.zeros(isodata.nisos())
    spikevector2[isospike[1]] = 1.0
    
    # Helpful to rescale the error, to make everything roughly order 1 for the optimiser
    initialerror, _ = errorestimate(isodata,0.5,0.5*spikevector1 + (1 - 0.5)*spikevector2,isoinv,errorratio,beta,alpha)
    
    def objective(y):
        p = expit(y[0])  # use expit transformation to keep things in range
        q = expit(y[1])
        
        error, ppmperamu = errorestimate(isodata,p,q*spikevector1 + (1 - q)*spikevector2,isoinv,errorratio,beta,alpha)
        return error/initialerror 
    
    y0 = np.array([0.0, 0.0])
    res = minimize(objective, y0, tol = 1e-10)
    
    y = res.x
    p = expit(y[0])
    q = expit(y[1])
    
    optprop = p
    optspike = q * spikevector1 + (1 - q) * spikevector2
    opterr,optppmperamu = errorestimate(isodata,p,q*spikevector1 + (1 - q)*spikevector2,isoinv,errorratio,beta,alpha)
    
    return optspike,optprop,opterr,optppmperamu


def optimalrealspike(isodata,beta = 0.0,alpha = 0.0,errorratio = None,isospike = None,isoinv = None): 
    #OPTIMALREALSPIKE    Finds the best pure spike
#    OPTIMALREALSPIKE(isodata,beta,alpha,errorratio,isospike,isoinv)
#             isodata -- data about a particular element
#             beta -- instrumental fractionation
#             alpha -- natural fractionation
#             errorratio -- the ratio whose error we are targeting
#             isospike -- the isotopes to spike
#             isoinv -- the isotopes used in the inversion

    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isospike = isodata.isoindex(isospike)
    isoinv = isodata.isoindex(isoinv)
    
    # If don't specify inversion isotopes, do all possible combinations
    if isoinv is None:
        isoinv = list(itertools.combinations(np.arange(isodata.nisos()), 4))
    else:
        isoinv = list([isoinv])
    
    # Work out all combinations of inversion isotopes and spiking isotopes
    isoinvvals = []
    isospikevals = []
    for i in range(len(isoinv)):
        if isospike is None:
            #isospikev = combnk(isoinv(i,:),2)
            isospikev = list(itertools.combinations(isoinv[i], 2))
        else:
            if len(set(isospike).intersection(set(isoinv[i]))) == 2:
                isospikev = list([isospike])
            else:
                isospikev = None
        
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
        optspike,optprop,opterr,optspikeprop,optppmperamu = singlerealoptimalspike(isodata,beta,alpha,errorratio,isospikevals[i,:],isoinvvals[i,:])
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
    opterr = opterr[ix]
    optppmperamu = optppmperamu[ix]
    optspike = optspike[ix,:]
    optprop = optprop[ix]
    optisoinv = optisoinv[ix,:]
    optisoinv = isodata.isonum[optisoinv]
    optspikeprop = optspikeprop[ix,:]
    
    return optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu
    
# A lot of copy-paste here --- need to clean-up
def singlerealoptimalspike(isodata,beta = 0.0,alpha = 0.0,errorratio = None,isospike = None,isoinv = None): 
    # Calculate the composition of the optimal double spike given the isotopes used in the inversion
    # and of those the isotopes we are spiking
    
    spikevector1 = isodata.rawspike[isospike[0], :]
    spikevector2 = isodata.rawspike[isospike[1], :]
    
    # Helpful to rescale the error, to make everything roughly order 1 for the optimiser
    initialerror, _ = errorestimate(isodata,0.5,0.5*spikevector1 + (1 - 0.5)*spikevector2,isoinv,errorratio,beta,alpha)
    
    tol = 1e-05
    lb = np.array([logit(tol), logit(tol)])
    ub = np.array([1-logit(tol), 1-logit(tol)])
    
    con = LinearConstraint(np.eye(2), lb, ub)
    
    def objective(y):
        p = expit(y[0])  # use expit transformation to keep things in range
        q = expit(y[1])
        
        error, ppmperamu = errorestimate(isodata,p,q*spikevector1 + (1.0 - q)*spikevector2,isoinv,errorratio,beta,alpha)
        return error/initialerror 
    
    y0 = np.array([0.0, 0.0])
    #res = minimize(objective, y0, tol = 1e-16, constraints = {con})
    res = minimize(objective, y0, tol = 1e-9)
    
    y = res.x
    p = expit(y[0])
    q = expit(y[1])
    
    optprop = p
    optspike = q * spikevector1 + (1 - q) * spikevector2
    opterr,optppmperamu = errorestimate(isodata,p,q*spikevector1 + (1 - q)*spikevector2,isoinv,errorratio,beta,alpha)
    
    optspikeprop=np.zeros(isodata.nrawspikes())
    optspikeprop[isospike[0]]=q
    optspikeprop[isospike[1]]=1-q
    
    return optspike,optprop,opterr,optspikeprop,optppmperamu


if __name__=="__main__":
    from .isodata import IsoData
    #isodata = IsoData('Fe')
    isodata = IsoData('Ca')
    isoinv=[40, 44, 46, 48]
    
    #spike = np.array([[1e-9, 0.1, 0.4, 0.4],[1e-9, 0.1, 0.4, 0.4]])    
    #isodata.set_spike([0.0, 0.0, 0.5, 0.5])
    isodata.set_errormodel()

    #alpha_err, ppm_err = errorestimate(isodata, prop = 0.5, alpha = -0.2, beta = 1.8 )
    
    optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalrealspike(isodata, isospike = [0,3], isoinv = isoinv)
    
    #optspike,optprop,opterr,optspikeprop,optppmperamu = singlerealoptimalspike(isodata, isospike = np.array([0, 1, 2, 3]))
    
    print(optspike)
    print(optprop)
    print(opterr)
    #print(optisoinv)
    print(optspikeprop)
    print(optppmperamu)

