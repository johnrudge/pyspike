import numpy as np
    
def errorpropagation(z = None,AP = None,An = None,AT = None,Am = None,AVn = None,AVT = None,AVm = None,srat = None): 
    # does error propagation for the fractionation correction
# srat specifies which subset (3 ratios) you'd like to use
    
    nratios = len(An)
    nmeasured = len(Am(:,1))
    Vz = np.zeros((nmeasured,3,3))
    VAN = np.zeros((nmeasured,nratios,nratios))
    VAM = np.zeros((nmeasured,nratios,nratios))
    for i in np.arange(1,nmeasured+1).reshape(-1):
        Vz[i,:,:],VAN[i,:,:],VAM[i,:,:] = fcerrorpropagation(z(i,:),AP,An,AT,Am(i,:),AVn,AVT,np.squeeze(AVm(i,:,:)),srat)
    
    return Vz,VAN,VAM