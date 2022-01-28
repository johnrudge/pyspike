import numpy as np
import numpy.matlib
    
def realproptoratioprop(realprop = None,ratios = None): 
    # convert a proportion in moles to a proportion in ratio space
    
    nratios = len(ratios(:,1))
    nrealprops = len(realprop(:,1))
    invpropdenom = 1.0 / np.transpose((np.sum(np.array([np.ones((nratios,1)),ratios]), 2-1)))
    newinvpropdenom = np.matlib.repmat(invpropdenom,np.array([nrealprops,1]))
    prop = (np.multiply(realprop,newinvpropdenom))
    sprop = np.sum(prop, 2-1)
    sprop = np.matlib.repmat(sprop,np.array([1,nratios]))
    prop = prop / sprop
    return prop