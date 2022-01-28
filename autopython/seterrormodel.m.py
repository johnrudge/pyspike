#SETERRORMODEL   Sets the coefficients of the error model in the global variable ISODATA
#    SETERRORMODEL(intensity,deltat,R,T,radiogenicisos)
#             intensity -- mean total beam intensity (volts). Default is 10 V.
#             deltat -- integration time (seconds). Default is 8 s.
#             R -- resistance (ohms). Default is 10^11 ohms.
#             T -- temperature (Kelvin). Default is 300 K.
#             radiogenicisos -- which isotopes use the radiogenic error model
#                  (errors on standard). Default is {'Pb','Sr','Hf','Os','Nd'}
#
# This function generates the coefficients of the ISODATA.(element).errormodel
# By default, only the measurements of the mixture have errors. For radiogenic
# isotopes, the standard (i.e. unspiked run) also has errors.
#
# The error model for the beams takes the form
#     sigma_^2 = a + b * mu + c * mu^2
# Each beam is assumed independent.
#
# Example
#    seterrormodel(15,4);   # set 15 V total beam with 4 second integrations
#
# See also dsstartup
import numpy as np
    
def seterrormodel(intensity = None,deltat = None,R = None,T = None,radiogenicisos = None,type_ = None): 
    global ISODATA
    # Fundamental constants
    elementarycharge = 1.60217646e-19
    
    k = 1.3806504e-23
    
    if (len(varargin) < 6):
        type_ = 'fixed-total'
    
    if (len(varargin) < 5):
        radiogenicisos = np.array(['Pb','Sr','Hf','Os','Nd'])
    
    # Mass spec properties
    if (len(varargin) < 4) or len(T)==0:
        T = 300
    
    if (len(varargin) < 3) or len(R)==0:
        R = 100000000000.0
    
    if (len(varargin) < 2) or len(deltat)==0:
        deltat = 8
    
    if (len(varargin) < 1) or len(intensity)==0:
        intensity = 10
    
    a = 4 * k * T * R / (deltat)
    
    b = elementarycharge * R / (deltat)
    
    els = fieldnames(ISODATA)
    for i in np.arange(1,len(els)+1).reshape(-1):
        element = els[i]
        nisos = getattr(ISODATA,(element)).nisos
        # by default assume Johnson noise and counting statistics
        errormodel.measured.type = 'fixed-total'
        errormodel.measured.intensity = intensity
        errormodel.measured.a = np.multiply(a,np.ones((1,nisos)))
        errormodel.measured.b = np.multiply(b,np.ones((1,nisos)))
        errormodel.measured.c = 0.0 * np.ones((1,nisos))
        # by default, assume no noise on the spike
        errormodel.spike.type = 'fixed-total'
        errormodel.spike.intensity = intensity
        errormodel.spike.a = 0.0 * np.ones((1,nisos))
        errormodel.spike.b = 0.0 * np.ones((1,nisos))
        errormodel.spike.c = 0.0 * np.ones((1,nisos))
        # by default, assume no noise on standard unless it is radiogenic
        errormodel.standard.type = 'fixed-total'
        errormodel.standard.intensity = intensity
        if len(intersect(radiogenicisos,np.array([element])))==0:
            errormodel.standard.a = 0.0 * np.ones((1,nisos))
            errormodel.standard.b = 0.0 * np.ones((1,nisos))
        else:
            errormodel.standard.a = np.multiply(a,np.ones((1,nisos)))
            errormodel.standard.b = np.multiply(b,np.ones((1,nisos)))
        errormodel.standard.c = 0.0 * np.ones((1,nisos))
        getattr(ISODATA,(element)).errormodel = errormodel
    