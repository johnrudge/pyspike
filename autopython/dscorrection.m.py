import numpy as np
    
def dscorrection(P = None,n = None,T = None,m = None,options = None): 
    # Routine for double spike fractionation correction
# takes as input ratios:
#       P -- log of ratio of atomic masses
#       n -- ratio of standard/ unspiked run
#       T -- ratio of spike
#       m -- ratio of measured
# outputs enriched ratio proportion (lambda), natural fractionation (alpha),
# and instrumental fractionation (beta) as a vector z=(lambda, (1-lambda)*alpha, beta)
    
    # start by solving the linear problem
    b = np.transpose((m - n))
    A = np.array([np.transpose((T - n)),np.transpose((np.multiply(- n,P))),np.transpose((np.multiply(m,P)))])
    y0 = np.linalg.solve(A,b)
    # by starting at the linear solution, solve the non-linear problem
    y,fval,exitflag,output = fsolve(lambda y = None: F(y,P,n,T,m),y0,options)
    z = y
    z[2] = y(2) / (1 - y(1))
    
    
def F(y = None,P = None,n = None,T = None,m = None): 
    # The nonlinear equations to solve
    lambda_ = y(1)
    alpha = y(2) / (1 - lambda_)
    beta = y(3)
    N = np.multiply(n,np.exp(np.multiply(- alpha,P)))
    M = np.multiply(m,np.exp(np.multiply(- beta,P)))
    fval = (np.multiply(lambda_,T)) + (np.multiply((1 - lambda_),N)) - M
    # The Jacobian of the nonlinear equations -- can speed up root finding, but is not required
    dfdlambdaprime = T - (np.multiply(N,(1 + np.multiply(alpha,P))))
    dfdu = np.multiply(- N,P)
    dfdbeta = np.multiply(M,P)
    Jac = np.array([np.transpose(dfdlambdaprime),np.transpose(dfdu),np.transpose(dfdbeta)])
    return z