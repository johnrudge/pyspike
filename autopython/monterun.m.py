import numpy as np
import numpy.matlib
    
def monterun(element = None,prop = None,spike = None,alpha = None,beta = None,n = None): 
    #MONTERUN Generate a fake mass spectrometer run by Monte-Carlo simulation
#    [measuredv standardv spikev]=MONTERUN(element,prop,spike,alpha,beta,n)
#             element -- element used in double spike, e.g. 'Fe'
#             prop -- proportion of spike in double spike-sample mix.
#                A vector of values can be specified if desired, to reflect changes over a run.
#             spike -- the isotopic composition of the double spike e.g. [0 0.5 0 0.5]
#                corresponds to a 50-50 mixture of the 2nd and 4th isotopes
#                (56Fe and 58Fe) in the case of Fe.
#             alpha, beta -- there is a small dependance of the error on the fractionation
#                factors (instrumental and natural, or alpha and beta). Values of beta and
#                alpha can be set here if desired, although the effect on the optimal spikes
#                is slight unless the fractionations are very large. Default is zero.
#                A vector of values can be specified if desired, to reflect changes over a run.
#             n -- number of Monte-Carlo samples to take. Default is 1000.
    
    # Note that a number of parameters are specified in the global variable ISODATA.
    
    # This function produces a fake mass spectrometer run using the error model specified in
# ISODATA.(element).errormodel. The output is given as ion beam intensities for measured,
# standard, and double spike.
    
    # Example
#    measured=monterun('Fe',0.5,[0 0 0.5 0.5]);
    
    # See also dsinversion
    global ISODATA
    # Set some defaults
    if len(ISODATA)==0:
        dsstartup
    
    if (len(varargin) < 6) or (len(n)==0):
        n = 1000
    
    if (len(varargin) < 5) or (len(beta)==0):
        beta = 0
    
    if (len(varargin) < 4) or (len(alpha)==0):
        alpha = 0
    
    rawdata = getattr(ISODATA,(element))
    standard = rawdata.standard
    mass = rawdata.mass
    emodel = rawdata.errormodel
    spike = spike / sum(spike)
    nisos = getattr(ISODATA,(element)).nisos
    if (len(alpha) == 1):
        alpha = np.matlib.repmat(alpha,np.array([n,1]))
    
    if (len(beta) == 1):
        beta = np.matlib.repmat(beta,np.array([n,1]))
    
    if (len(prop) == 1):
        prop = np.matlib.repmat(prop,np.array([n,1]))
    
    # This code needs vectorising...
# calculate expected sample composition
    sample = np.zeros((n,nisos))
    mixture = np.zeros((n,nisos))
    measured = np.zeros((n,nisos))
    for i in np.arange(1,n+1).reshape(-1):
        sample[i,:] = np.multiply(standard,np.exp(np.multiply(- np.log(mass),alpha(i))))
        sample[i,:] = sample(i,:) / sum(sample(i,:))
        mixture[i,:] = np.multiply(prop(i),spike) + np.multiply((1 - prop(i)),sample(i,:))
        measured[i,:] = np.multiply(mixture(i,:),np.exp(np.multiply(np.log(mass),beta(i))))
        measured[i,:] = measured(i,:) / sum(measured(i,:))
        measuredi[i,:] = np.multiply(measured(i,:),emodel.measured.intensity)
        if emodel.measured.type=='fixed-sample':
            measuredi[i,:] = measuredi(i,:) / (1.0 - prop(i))
        measuredivar[i,:] = emodel.measured.a + np.multiply(emodel.measured.b,measuredi(i,:)) + np.multiply(emodel.measured.c,(measuredi(i,:) ** 2))
        measuredicov[:,:,i] = diag(measuredivar(i,:))
        standardi[i,:] = np.multiply(standard,emodel.standard.intensity)
        standardivar[i,:] = emodel.standard.a + np.multiply(emodel.standard.b,standardi(i,:)) + np.multiply(emodel.standard.c,(standardi(i,:) ** 2))
        standardicov[:,:,i] = diag(standardivar(i,:))
        spikei[i,:] = np.multiply(spike,emodel.spike.intensity)
        spikeivar[i,:] = emodel.spike.a + np.multiply(emodel.spike.b,spikei(i,:)) + np.multiply(emodel.spike.c,(spikei(i,:) ** 2))
        spikeicov[:,:,i] = diag(spikeivar(i,:))
    
    measuredv = mvnrnd(measuredi,measuredicov,n)
    standardv = mvnrnd(standardi,standardicov,n)
    spikev = mvnrnd(spikei,spikeicov,n)
    return measuredv,standardv,spikev