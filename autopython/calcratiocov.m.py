import numpy as np
    
def calcratiocov(composition = None,errormodel = None,di = None,isonorm = None,prop = None): 
    # Calculate the covariance matrix of the ratios based on the given error model and composition
# di is the isotope with which to denominator
# isonorm are the isotopes to use in the normalisation
# prop is the proportion of spike in the spike-sample mix
    
    if (len(varargin) < 5) or len(isonorm)==0:
        prop = 0.0
    
    if (len(varargin) < 4) or len(isonorm)==0:
        isonorm = np.arange(1,composition.shape[2-1]+1)
    
    # first normalise composition so it is really a composition (unit sum)
    composition = composition / sum(composition)
    meanbeams = np.multiply(composition,errormodel.intensity) / sum(composition(isonorm))
    if errormodel.type=='fixed-sample':
        meanbeams = meanbeams / (1.0 - prop)
    
    covbeams = calcbeamcov(meanbeams,errormodel)
    V = covbeamtoratio(meanbeams,covbeams,di)
    
def calcbeamcov(meanbeams = None,errormodel = None): 
    # the beam covariance matrix
    beamvar = errormodel.a + np.multiply(meanbeams,errormodel.b) + np.multiply((meanbeams ** 2),errormodel.c)
    beamcov = diag(beamvar)
    
def covbeamtoratio(meanbeams = None,covbeams = None,di = None): 
    # converts a covariance matrix for beams to one for ratios
# di is the isotope to denominator with
# assumes last row and column of M correspond to denominator
    isonums = np.arange(1,len(meanbeams)+1)
    ni = isonums(isonums != di)
    n = meanbeams(ni)
    d = meanbeams(di)
    M = covbeams(np.array([ni,di]),np.array([ni,di]))
    
    #A=[diag(repmat(1/d,1,length(n))) -n'./(d^2)];
#A=[(1/d).*eye(length(n)) -n'./(d^2)];
    A = np.array([diag(np.multiply((1 / d),np.ones((1,len(n))))),- np.transpose(n) / (d ** 2)])
    V = (A * M) * (np.transpose(A))
    return V