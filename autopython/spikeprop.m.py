import numpy as np
    
def spikeprop(lambda_ = None,X1 = None,X2 = None): 
    # convert proportion in ratio space to proportion by moles
# X = lambda*X1 + (1-lambda)*X2 in ratio land
    
    s1 = (1 + sum(X1))
    s2 = (1 + sum(X2))
    p1 = np.multiply(lambda_,s1) / ((np.multiply((1 - lambda_),s2)) + (np.multiply(lambda_,s1)))
    prop = ratioproptorealprop(np.array([np.transpose(lambda_),np.transpose((1 - lambda_))]),np.array([[X1],[X2]]))
    p2 = np.transpose(prop(:,1))
    p = p1
    return p