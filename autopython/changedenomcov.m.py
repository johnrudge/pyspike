import numpy as np
    
def changedenomcov(data = None,datacov = None,olddi = None,newdi = None): 
    # change denominator of covariance matrix for given set of ratios
    
    nisos = len(data) + 1
    oldni = np.array([np.arange(1,(olddi - 1)+1),np.arange((olddi + 1),nisos+1)])
    dataplus = np.array([data(np.arange(1,(olddi - 1)+1)),1,data(np.arange((olddi),end()+1))])
    newni = np.array([np.arange(1,(newdi - 1)+1),np.arange((newdi + 1),nisos+1)])
    datacovplus = np.zeros((nisos,nisos))
    datacovplus[oldni,oldni] = datacov
    A = np.eye(nisos) / dataplus(newdi)
    A[:,newdi] = A(:,newdi) - np.transpose(dataplus) / (dataplus(newdi) ** 2)
    newdatacovplus = A * datacovplus * (np.transpose(A))
    newdatacov = newdatacovplus(newni,newni)
    return newdatacov