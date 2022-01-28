import numpy as np
    
def fcerrorpropagation(z = None,AP = None,An = None,AT = None,Am = None,VAn = None,VAT = None,VAm = None,srat = None): 
    # linear error propagation for the fractionation correction
    
    lambda_ = z(1)
    alpha = z(2)
    beta = z(3)
    AM = np.multiply(Am,np.exp(- AP * beta))
    AN = np.multiply(An,np.exp(- AP * alpha))
    # Select appropriate ratios
    P = AP(srat)
    
    N = AN(srat)
    
    T = AT(srat)
    
    M = AM(srat)
    
    VT = VAT(srat,srat)
    
    Vm = VAm(srat,srat)
    
    Vn = VAn(srat,srat)
    
    # calculate various Jacobian matrices
    dfdlambda = T - (np.multiply(N,(1 + np.multiply(alpha,P))))
    dfdu = np.multiply(- N,P)
    dfdbeta = np.multiply(M,P)
    dfdy = np.array([np.transpose(dfdlambda),np.transpose(dfdu),np.transpose(dfdbeta)])
    dfdT = np.multiply(lambda_,np.eye(3))
    dfdm = - diag(np.exp(- beta * P))
    dfdn = (1 - lambda_) * diag(np.exp(- alpha * P))
    # matrix to convert from (lambda, (1-lambda)alpha,beta) to (lambda,alpha,beta)
    K = np.array([1,0,0(alpha / (1 - lambda_)),(1 / (1 - lambda_)),0,0,0,1])
    dzdT = - K * (np.linalg.solve(dfdy,dfdT))
    dzdm = - K * (np.linalg.solve(dfdy,dfdm))
    dzdn = - K * (np.linalg.solve(dfdy,dfdn))
    # Covariance matix for (lambda,beta,alpha)
    Vz = dzdT * VT * (np.transpose(dzdT)) + dzdm * Vm * (np.transpose(dzdm)) + dzdn * Vn * (np.transpose(dzdn))
    # full matrices for all ratios
    nratios = len(An)
    dzdAT = np.zeros((3,nratios))
    dzdAn = np.zeros((3,nratios))
    dzdAm = np.zeros((3,nratios))
    dzdAT[np.arange[1,3+1],srat] = dzdT
    dzdAn[np.arange[1,3+1],srat] = dzdn
    dzdAm[np.arange[1,3+1],srat] = dzdm
    # Covariance matrix of sample
    dalphadAT = dzdAT(2,:)
    dalphadAn = dzdAn(2,:)
    dalphadAm = dzdAm(2,:)
    dANdAT = - ((np.transpose((np.multiply(AN,AP)))) * dalphadAT)
    dANdAn = diag(np.exp(np.multiply(- AP,alpha))) - ((np.transpose((np.multiply(AN,AP)))) * dalphadAn)
    dANdAm = - ((np.transpose((np.multiply(AN,AP)))) * dalphadAm)
    VAN = dANdAn * VAn * (np.transpose(dANdAn)) + dANdAT * VAT * (np.transpose(dANdAT)) + dANdAm * VAm * (np.transpose(dANdAm))
    # Covariance matrix of mixture
    dbetadAT = dzdAT(3,:)
    dbetadAn = dzdAn(3,:)
    dbetadAm = dzdAm(3,:)
    dAMdAT = - ((np.transpose((np.multiply(AM,AP)))) * dbetadAT)
    dAMdAn = - ((np.transpose((np.multiply(AM,AP)))) * dbetadAn)
    dAMdAm = diag(np.exp(np.multiply(- beta,AP))) - ((np.transpose((np.multiply(AM,AP)))) * dbetadAm)
    VAM = dAMdAn * VAn * (np.transpose(dAMdAn)) + dAMdAT * VAT * (np.transpose(dAMdAT)) + dAMdAm * VAm * (np.transpose(dAMdAm))
    return Vz,VAN,VAM