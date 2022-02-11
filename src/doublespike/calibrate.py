from .isodata import IsoData, normalise_composition, realproptoratioprop, ratioproptorealprop
from .inversion import dscorrection, dsinversion
from .errors import calcratiocov
from scipy.optimize import minimize
import numpy as np

def spike_calibration(isodata, spike_measurement, mixture_measurement, isoinv = None, standard=None):
    """A simple least squares routine for calibrating a double spike from spike-standard mixtures."""
    if isoinv is None:
        if isodata.isoinv is None:
            raise Exception("Inversion isotopes not specified.")
        else:
            isoinv = isodata.isoinv
    if standard is None:
        standard = isodata.standard
    
    # make sure working with numpy arrays
    spike_measurement = np.array(spike_measurement)
    mixture_measurement = np.array(mixture_measurement)
    
    # normalise so have compositional vectors
    spike_measurement = normalise_composition(spike_measurement)
    mixture_measurement = normalise_composition(mixture_measurement)
    
    # choose isotope to denominator by using largest isotope in spike
    isoinv = isodata.isoindex(isoinv)
    ix = np.argmax(spike_measurement[isoinv])
    deno = isoinv[ix]
    nume = isoinv[isoinv != deno]
    isoinv = np.concatenate((np.array([deno]),nume))
    
    invrat = isodata.invrat(isoinv)
    
    An = isodata.ratio(standard, deno)
    At = isodata.ratio(spike_measurement, deno)
    Am = isodata.ratio(mixture_measurement, deno)
    AP = np.log(isodata.ratio(isodata.mass, deno))
    
    n_obs = mixture_measurement.shape[0]
    
    emod_mixture = isodata.errormodel['measured']
    VAms = [calcratiocov(mixture_measurement[i,:],emod_mixture,deno) for i in range(n_obs)]
    emod_spike = isodata.errormodel['measured']
    VAt = calcratiocov(spike_measurement,emod_spike,deno)
        
    n = An[invrat]
    P = AP[invrat]
    t = At[invrat]
    m = Am[:, invrat]
    
    VMs = [V[np.ix_(invrat,invrat)] for V in VAms]
    Vt = VAt[np.ix_(invrat,invrat)]
    
    SMs = [np.linalg.inv(V) for V in VMs]
    St = np.linalg.inv(Vt)
    
    # form initial guess of model parameters. guess a 50-50 mix, with no fractionation
    prop0 = 0.5
    lambda0 = realproptoratioprop(prop0, At, An) * np.ones(m.shape[0])
    beta0 = 0.0*np.ones(m.shape[0])
    betaT0 = np.array([0.0])
    T0 = t
    z0 = np.concatenate((lambda0, beta0, betaT0, T0))
    df = (1+ m.shape[0])*len(invrat) - len(z0)  # degrees of freedom 


    res = minimize(objective, z0, args=(t, m, P, n, SMs, St), jac =True, tol = 1e-16, options={'disp':False ,'gtol':1e-8,'eps':1e-12})
    z = res.x
    misfit = res.fun
    n_ratios, n_obs, lambda_, beta, betat, T = zP_to_params(z, P)
    
    # Reconstruct spike vector
    calibrated_spike = np.zeros_like(spike_measurement)
    calibrated_spike[deno] = 1.0
    calibrated_spike[nume] = T
    
    # For isotopes that were not used in inversion, work out an expectation based on known betat
    isonum = np.arange(isodata.nisos())
    unused = np.array(list(set(isonum).difference(set(isoinv))))
    if len(unused)>0:
        expected_spike = spike_measurement*np.exp(-np.log(isodata.mass)*betat)
        expected_spike = normalise_composition(expected_spike)
        print("expected spike", expected_spike)
        expected_unused = expected_spike[unused]/expected_spike[deno]
        calibrated_spike[unused] = expected_unused
    
    calibrated_spike = normalise_composition(calibrated_spike)
    
    AT = isodata.ratio(calibrated_spike, deno)
    prop = [ratioproptorealprop(l, AT, An) for l in lambda_]
    
    out = {'calibrated_spike': calibrated_spike,
           'prop_mixture': prop,
           'beta_mixture': beta,
            'beta_spike': betat,
            'misfit' : misfit,
            'df': df}

    return out

def objective(z, t, m, P, n, Wm, Wt):
    me, te = mt_expected(z, P, n)
    res_m = m - me
    res_t = t - te
    
    ob_ms = []
    for i in range(res_m.shape[0]):
        rm = res_m[i,:][np.newaxis,:]
        W = Wm[i]
        ob_ms.append( (rm @ W @ rm.T)[0][0])
    
    rt = res_t[np.newaxis,:]
    ob_t = (rt @ Wt @ rt.T)[0][0]
    
    ob = sum(ob_ms) + ob_t
    
    dmdz, dtdz = dmt_expected_dz(z, P, n)
    
    dob_ms_dzs = []
    for i in range(res_m.shape[0]):
        rm = res_m[i,:][np.newaxis,:]
        W = Wm[i]
        dmidz = dmdz[i,:,:]
        dob_ms_dzs.append(- (2 * rm @ W @ dmidz)[0])
    dob_t_dz = - (2 * rt @ Wt @ dtdz)[0]
    dob_ms_dzs.append(dob_t_dz)
    dob_dz = np.vstack(dob_ms_dzs)
    dob_dz = np.sum(dob_dz, axis = 0)
    
    return ob, dob_dz


def individual_m_expected(lambda_, beta, T, P, n):
    return np.exp(beta*P) * (lambda_*T + (1-lambda_)*n)

def dindividual_m_expected_dlambda(lambda_, beta, T, P, n):
    return np.exp(beta*P) * (T - n)

def dindividual_m_expected_dbeta(lambda_, beta, T, P, n):
    return P * np.exp(beta*P) * (lambda_*T + (1-lambda_)*n)

def dindividual_m_expected_dT(lambda_, beta, T, P, n):
    return np.diag(np.exp(beta*P) * lambda_)

def t_expected(betat, T, P):
    return np.exp(betat*P) * T

def dt_expected_dbetat(betat, T, P):
    return P * np.exp(betat*P) * T

def dt_expected_dT(betat, T, P):
    return np.diag(np.exp(betat*P))


def zP_to_params(z, P):
    n_ratios = len(P)
    n_obs = (len(z) - n_ratios - 1)//2
    lambda_ = z[0:n_obs]
    beta = z[n_obs:2*n_obs]
    betat = z[2*n_obs]
    T = z[2*n_obs+1:] 
    return n_ratios, n_obs, lambda_, beta, betat, T

def mt_expected(z, P, n):
    n_ratios, n_obs, lambda_, beta, betat, T = zP_to_params(z, P)
    m = np.zeros((n_obs, n_ratios))
    
    for i in range(n_obs):
        m[i,:] = individual_m_expected(lambda_[i], beta[i], T, P, n)
    
    t = t_expected(betat, T, P)
    
    return m, t

def dmt_expected_dz(z, P, n):
    n_ratios, n_obs, lambda_, beta, betat, T = zP_to_params(z, P)
    dmdz = np.zeros((n_obs, n_ratios, len(z)))
    
    for i in range(n_obs):
        dmdz[i,:,i] = dindividual_m_expected_dlambda(lambda_[i], beta[i], T, P, n)
        dmdz[i,:,n_obs+i] = dindividual_m_expected_dbeta(lambda_[i], beta[i], T, P, n)
        dmdz[i,:,2*n_obs+1:] = dindividual_m_expected_dT(lambda_[i], beta[i], T, P, n)
    
    #t = t_expected(betat, T, P)
    
    dtdz = np.zeros((n_ratios, len(z)))
    dtdz[:,2*n_obs] = dt_expected_dbetat(betat, T, P)
    dtdz[:,2*n_obs+1:] = dt_expected_dT(betat, T, P)
    
    return dmdz, dtdz


if __name__=="__main__":
    from .monte import monterun
    #isodata = IsoData('Fe')
    #n = int(1e3)
    #true_spike = np.array([1e-3, 1e-2, 0.5, 0.5])
    
    isodata = IsoData('Pt')
    isodata.set_isoinv([194, 195, 196, 198])
    true_spike =np.array([0.000042, 0.004428, 0.333969, 9.962500, 0.618252, 11.898364])
    
    n = int(1e6)
    isodata.set_errormodel(intensity=1e3)
    true_spike = true_spike / sum(true_spike)
    
    
    
    spike_measurements = monterun(isodata, 1.0, true_spike, alpha = 0.0, beta = 0.8, n=n)
    mixture1_measurements = monterun(isodata, 0.7, true_spike, alpha = 0.0, beta = 2.0, n=n)
    mixture2_measurements = monterun(isodata, 0.5, true_spike, alpha = 0.0, beta = 1.5, n=n)
    mixture3_measurements = monterun(isodata, 0.3, true_spike, alpha = 0.0, beta = 1.0, n=n)
    
    spike_measurement = np.mean(spike_measurements, axis = 0)
    mixture1_measurement = np.mean(mixture1_measurements, axis = 0)
    mixture2_measurement = np.mean(mixture2_measurements, axis = 0)
    mixture3_measurement = np.mean(mixture3_measurements, axis = 0)

    mixture_measurement = np.vstack([mixture1_measurement,mixture2_measurement])

    spike_measurement = normalise_composition(spike_measurement)
    mixture_measurement = normalise_composition(mixture_measurement)
    

    #print("true_spike:", true_spike)
    #print("spike measurement:", spike_measurement)
    #print("standard-spike measurements:", mixture_measurement)
    
    out = spike_calibration(isodata, spike_measurement, mixture_measurement)
    print(out)
    
    
    ##test_spike = true_spike
    test_spike = out['calibrated_spike']
    ##test_spike = spike_measurement
    
    m1 = dsinversion(isodata, mixture1_measurement, spike = test_spike)
    m2 = dsinversion(isodata, mixture2_measurement, spike = test_spike)
    m3 = dsinversion(isodata, mixture3_measurement, spike = test_spike)
    print(m1)
    print(m2)
    print(m3)
    
    print("spike_diff", out['calibrated_spike'] - true_spike)
    
