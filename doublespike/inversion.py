from isodata import IsoData
import numpy as np
from scipy.optimize import fsolve 

def dsinversion(isodata, measured, spike=None, isoinv=None, standard=None):
#    DSINVERSION    Do the double spike inversion for a given set of measurements
#    out=DSINVERSION(element,measured,spike,isoinv,standard)
#             element -- element used in double spike, e.g. 'Fe'
#             measured -- a matrix of beam intensities. Columns correspond to the
#                 different isotopes e.g. for Fe, first column is 54Fe, second is 56Fe,
#                 third is 57Fe, fourth is 58Fe. The matrix should have the same number
#                 of columns as there are isotopes available.
#             spike -- a composition vector for the spike. e.g. [0 0 0.5 0.5] is a 50-50
#                 mix of 57Fe and 58Fe.
#             isoinv -- the four isotopes to use in the inversion, e.g [54 56 57 58]. This
#                 defaults to the first four isotopes.
#             standard -- standard composition or unspiked run data. Defaults to the
#                 value in ISODATA.(element).standard if not specified.
# This routine performs the double spike inversion on measured data to return the
# "true" composition of the sample. Output is returned as a structure with the
# following fields
#             alpha -- the inferred natural fractionations
#             beta -- the inferred instrumental fractionations
#             prop -- the inferred proportions of spike to sample
#             sample -- the inferred compositions of the sample
#             mixture -- the inferred compositions of the mixture
#   Example
#   out=dsinversion('Fe',measured,[0 0 0.5 0.5],[54 56 57 58]);
#   plot(out.alpha);

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
    if standard is None:
        standard = isodata.standard

    # Duplicate so all matrices same size
    nspike, nstandard, nmeasured = 1, 1, 1
    if spike.ndim >1:
        nspike = spike.shape[0]
    if measured.ndim >1:
        nmeasured = measured.shape[0]
    if standard.ndim >1:
        nstandard = standard.shape[0]
    nobs = max(nspike, nmeasured, nstandard)
    
    if spike.ndim==1:
        spike = np.tile(spike, (nobs,1))
    if measured.ndim==1:
        measured = np.tile(measured, (nobs,1))    
    if standard.ndim==1:
        standard = np.tile(standard, (nobs,1))  

    # Avoid division by zero errors for small values
    isoinv = isodata.isoindex(isoinv)
    if any(spike[0,isoinv] < 0.001):
        ix = np.argmax(spike[0,isoinv])
        deno = isoinv[ix]
        nume = isoinv[isoinv != deno]
        isoinv = np.concatenate((np.array([deno]),nume))

    # Take ratios based on the isotopes we are inverting
    P = np.log(ratio(idat.mass, isoinv))
    n = ratio(standard, isoinv)
    T = ratio(spike, isoinv)
    m = ratio(measured,isoinv)
    
    z = np.zeros((nmeasured,3))
    for i in range(nobs):
        z[i,:] = dscorrection(P,n[i,:],T[i,:],m[i,:],xtol=1e-12)
    out = {}
    lambda_ = z[:,0]
    out['alpha'] = z[:,1]
    out['beta'] = z[:,2]
    
    isonum = np.arange(isodata.nisos())
    isonum = isonum[isonum != isoinv[0]]
    isonum = np.concatenate((np.array([isoinv[0]]), isonum))
    
    AP = np.log(ratio(idat.mass, isonum))
    AT = ratio(spike, isonum)
    An = ratio(standard, isonum)
    Am = ratio(measured, isonum)
    
    # Calculate sample and mixture proportion, and proportion by mole
    AM = np.zeros_like(Am)
    AN = np.zeros_like(Am)
    for i in np.arange(nobs):
        AM[i,:] = Am[i,:]*np.exp(- AP * out['beta'][i])
        AN[i,:] = Am[i,:]*np.exp(- AP * out['alpha'][i])
        prop = ratioproptorealprop(np.array([lambda_[i],(1 - lambda_[i])]),np.array([AT[i,:],AN[i,:]]))
        out.prop[i,0] = prop[0]
    
    return out

def ratioproptorealprop(ratprop, ratios): 
    # convert a proportion in ratio space to one per mole
    nratios = ratios.shape[1]
    nratprops = len(ratprop)

    invpropdenom = np.sum(np.vstack((np.ones(nratios),ratios)), axis = 0)
    print(invpropdenom)
    #newinvpropdenom = np.matlib.repmat(invpropdenom,np.array([nratprops,1]))
    newinvpropdenom = np.tile(invpropdenom,(nratprops,1))
    print(newinvpropdenom)                          
                              
    prop = ratprop*newinvpropdenom.T
    print("prop1",prop)
    sprop = np.sum(prop, 0)
    print(sprop)
    #sprop = np.matlib.repmat(sprop,np.array([1,nratios]))
    sprop = np.tile(sprop,(nratios,1))
    prop = prop / sprop
    print("prop",prop)
    return prop



def dscorrection(P, n, T, m, **kwargs): 
    # Routine for double spike fractionation correction
    # takes as input ratios:
    #       P -- log of ratio of atomic masses
    #       n -- ratio of standard/ unspiked run
    #       T -- ratio of spike
    #       m -- ratio of measured
    # outputs enriched ratio proportion (lambda), natural fractionation (alpha),
    # and instrumental fractionation (beta) as a vector z=(lambda, (1-lambda)*alpha, beta)

    # start by solving the linear problem
    b = np.transpose((m - n))
    A = np.array([np.transpose((T - n)),np.transpose((np.multiply(- n,P))),np.transpose((np.multiply(m,P)))])
    y0 = np.linalg.solve(A,b)
    
    # by starting at the linear solution, solve the non-linear problem    
    y = fsolve(F, y0, args=(P,n,T,m), fprime=J, **kwargs)
    z = y
    z[1] = y[1] / (1 - y[0])
    return z

def F_params(y, P, n, T, m):
    # variables in the objective function
    lambda_ = y[0]
    alpha = y[1] / (1 - lambda_)
    beta = y[2]
    N = n * np.exp(- alpha * P)
    M = m * np.exp(- beta * P)
    return lambda_, alpha, beta, N, M
    
def F(y, P, n, T, m): 
    # The nonlinear equations to solve
    lambda_, alpha, beta, N, M = F_params(y, P, n, T, m)
    fval = lambda_*T + (1-lambda_)*N - M
    return fval

def J(y, P, n, T, m):
    # The Jacobian of the nonlinear equations -- can speed up root finding, but is not required
    lambda_, alpha, beta, N, M = F_params(y, P, n, T, m)
    dfdlambdaprime = T - N*(1 + alpha*P)
    dfdu = -N*P
    dfdbeta = M*P
    Jac = np.array([dfdlambdaprime,dfdu,dfdbeta]).T
    return Jac


def ratio(data, isoidx):
    # convert data to isotope ratios based on choice of isotopes
    # first index is the denominator index
    di = isoidx[0]
    ni = isoidx[1:]
    return data[...,ni]/data[...,di,np.newaxis]


if __name__=="__main__":
    idat = IsoData('Fe')
    
    #spike = np.array([[1e-9, 0.1, 0.4, 0.4],[1e-9, 0.1, 0.4, 0.4]])    
    idat.set_spike([0.0, 0.0, 0.5, 0.5])
    measured = np.array([0.2658, 4.4861, 2.6302, 2.6180])

    z = dsinversion(idat, measured)
    print(z)
