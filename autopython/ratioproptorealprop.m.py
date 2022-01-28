import numpy as np
import numpy.matlib
    
def ratioproptorealprop(ratprop = None,ratios = None): 
    # convert a proportion in ratio space to one per mole
    
    nratios = len(ratios(:,1))
    nratprops = len(ratprop(:,1))
    invpropdenom = np.transpose((np.sum(np.array([np.ones((nratios,1)),ratios]), 2-1)))
    newinvpropdenom = np.matlib.repmat(invpropdenom,np.array([nratprops,1]))
    prop = (np.multiply(ratprop,newinvpropdenom))
    sprop = np.sum(prop, 2-1)
    sprop = np.matlib.repmat(sprop,np.array([1,nratios]))
    prop = prop / sprop
    return prop