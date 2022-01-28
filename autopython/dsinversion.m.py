import numpy as np
import numpy.matlib
    
def dsinversion(element = None,measured = None,spike = None,isoinv = None,standard = None): 
    #DSINVERSION    Do the double spike inversion for a given set of measurements
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
    
    # Example
#   out=dsinversion('Fe',measured,[0 0 0.5 0.5],[54 56 57 58]);
#   plot(out.alpha);
    
    # See also dsstartup
    
    global ISODATA
    # Set some defaults
    if len(ISODATA)==0:
        dsstartup
    
    if (len(varargin) < 5) or (len(standard)==0):
        standard = getattr(ISODATA,(element)).standard
    
    if (len(varargin) < 4) or (len(isoinv)==0):
        isoinv = np.array([1,2,3,4])
    
    rawdata = getattr(ISODATA,(element))
    isoinv = rawdata.isoindex(isoinv)
    # Avoid division by zero errors for small values
    if np.any(spike(1,isoinv) < 0.001):
        m,ix = np.amax(spike(1,isoinv))
        deno = isoinv(ix)
        nume = isoinv(isoinv != deno)
        isoinv = np.array([deno,nume])
    
    # Duplicate so all vectors are same length
    nmeasured = np.amax(np.array([spike.shape[1-1],standard.shape[1-1],measured.shape[1-1]]))
    if (standard.shape[1-1] == 1):
        standard = np.matlib.repmat(standard,np.array([nmeasured,1]))
    
    if (spike.shape[1-1] == 1):
        spike = np.matlib.repmat(spike,np.array([nmeasured,1]))
    
    if (measured.shape[1-1] == 1):
        measured = np.matlib.repmat(measured,np.array([nmeasured,1]))
    
    # Take ratios based on the isotopes we are inverting
    in_ = calcratioeddata(element,isoinv)
    in_.AT = spike(:,in_.Ani) / np.matlib.repmat(spike(:,in_.di),np.array([1,in_.nratios]))
    in_.An = standard(:,in_.Ani) / np.matlib.repmat(standard(:,in_.di),np.array([1,in_.nratios]))
    in_.Am = measured(:,in_.Ani) / np.matlib.repmat(measured(:,in_.di),np.array([1,in_.nratios]))
    # Do the fractionation correction
    z = fractionationcorrection(in_.AP,in_.An,in_.AT,in_.Am,in_.srat)
    # Create a structure of outputs for things we're interested in
    lambda_ = z(:,1)
    out.alpha = z(:,2)
    out.beta = z(:,3)
    # Calculate sample and mixture proportion, and proportion by mole
    AM = np.zeros((in_.Am.shape,in_.Am.shape))
    AN = np.zeros((in_.Am.shape,in_.Am.shape))
    for i in np.arange(1,nmeasured+1).reshape(-1):
        AM[i,:] = np.multiply(in_.Am(i,:),np.exp(- in_.AP * out.beta(i)))
        AN[i,:] = np.multiply(in_.An(i,:),np.exp(- in_.AP * out.alpha(i)))
        prop = ratioproptorealprop(np.array([lambda_(i),(1 - lambda_(i))]),np.array([[in_.AT(i,:)],[AN(i,:)]]))
        out.prop[i,1] = prop(1)
    
    out.sample = np.zeros((measured.shape,measured.shape))
    out.mixture = np.zeros((measured.shape,measured.shape))
    out.sample[:,in_.Ani] = AN
    out.sample[:,in_.di] = 1
    out.mixture[:,in_.Ani] = AM
    out.mixture[:,in_.di] = 1
    out.sample = out.sample / np.matlib.repmat(np.sum(out.sample, 2-1),np.array([1,rawdata.nisos]))
    out.mixture = out.mixture / np.matlib.repmat(np.sum(out.mixture, 2-1),np.array([1,rawdata.nisos]))
    return out