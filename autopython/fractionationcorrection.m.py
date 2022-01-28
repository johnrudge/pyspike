import numpy as np
import numpy.matlib
    
def fractionationcorrection(AP = None,An = None,AT = None,Am = None,srat = None): 
    # does fractionation correction for a vector of measured values
# srat specifies which 3 ratios you'd like to use
    
    # Select appropriate ratios
    P = AP(:,srat)
    
    n = An(:,srat)
    
    T = AT(:,srat)
    
    m = Am(:,srat)
    
    nmeasured = np.amax(np.array([m.shape[1-1],T.shape[1-1],n.shape[1-1]]))
    # Duplicate so all matrices are same length
    if (n.shape[1-1] == 1):
        n = np.matlib.repmat(n,np.array([nmeasured,1]))
    
    if (T.shape[1-1] == 1):
        T = np.matlib.repmat(T,np.array([nmeasured,1]))
    
    if (m.shape[1-1] == 1):
        m = np.matlib.repmat(m,np.array([nmeasured,1]))
    
    # dscorrection only works with a single value, here run over array
    options = optimset('Display','off','Jacobian','on','TolFun',1e-15)
    z = np.zeros((nmeasured,3))
    for i in np.arange(1,nmeasured+1).reshape(-1):
        z[i,:] = dscorrection(P,n(i,:),T(i,:),m(i,:),options)
    
    return z