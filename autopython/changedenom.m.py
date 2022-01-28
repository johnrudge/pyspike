import numpy as np
import numpy.matlib
    
def changedenom(data = None,olddi = None,newdi = None): 
    # change denominator for given set of ratios
    nisos = len(data(1,:)) + 1
    ndata = len(data(:,1))
    dataplus = np.array([data(:,np.arange(1,(olddi - 1)+1)),np.ones((np.array([ndata,1]),np.array([ndata,1]))),data(:,np.arange((olddi),end()+1))])
    denomplus = np.matlib.repmat(dataplus(:,newdi),np.array([1,nisos]))
    newdataplus = dataplus / denomplus
    newni = np.array([np.arange(1,(newdi - 1)+1),np.arange((newdi + 1),nisos+1)])
    newdata = newdataplus(:,newni)
    return newdata