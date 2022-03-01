from .isodata import normalise_composition, realproptoratioprop, ratioproptorealprop
from .errors import calcratiocov
from scipy.optimize import minimize
import numpy as np


def spike_calibration(
    isodata, spike_measurement, mixture_measurement, isoinv=None, standard=None
):
    """A simple least squares routine for calibrating a double spike from spike-standard mixtures.

    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        spike_measurement (array): a matrix of beam intensities for direct measurements of
            the spike. Columns correspond to the different isotopes e.g. for Fe, first
            column is 54Fe, second is 56Fe, third is 57Fe, fourth is 58Fe. The matrix should
            have the same number of columns as there are isotopes available.
        mixture_measurement (array): a matrix of beam intensities for the measurements of
            spike-standarard mixtures.
        isoinv (array): the isotopes to use in the fitting, e.g [54, 56, 57, 58]. If
            None this is read from isodata.
        standard (array): standard composition. If None this is read from isodata.

    Returns:
        This routine estimates the spike composition given a direct measurement of the spike
        and measurements of spike-standard mixtures. The routine minimises the chi-squared
        misfit between the measurements and model, where measurements are weighted
        according to the expected covariance given in isodata.errormodel['measured'].
        Output is returned as a dictionary with the following fields:
            calibrated_spike: the estimated spike composition
            prop_mixture: the proportion of spike in the spike-sample mixtures
            beta_mixture: the fractionation factors for the mixture measurements
            beta_spike: the fractionation factors for the spike measurements
            misfit: the chi-squared misfit
            df: the degrees of freedom for the chi-squared statistic
    """
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

    # make sure working with two dimensional arrays
    if mixture_measurement.ndim == 1:
        mixture_measurement = mixture_measurement[np.newaxis, :]
    if spike_measurement.ndim == 1:
        spike_measurement = spike_measurement[np.newaxis, :]

    # normalise so have compositional vectors
    spike_measurement = normalise_composition(spike_measurement)
    mixture_measurement = normalise_composition(mixture_measurement)

    # choose isotope to denominator by using largest isotope in spike
    isoinv = isodata.isoindex(isoinv)
    ix = np.argmax(spike_measurement[0, isoinv])
    deno = isoinv[ix]
    nume = isoinv[isoinv != deno]
    isoinv = np.concatenate((np.array([deno]), nume))

    invrat = isodata.invrat(isoinv)

    An = isodata.ratio(standard, deno)
    At = isodata.ratio(spike_measurement, deno)
    Am = isodata.ratio(mixture_measurement, deno)
    AP = np.log(isodata.ratio(isodata.mass, deno))

    n_m = mixture_measurement.shape[0]
    n_t = spike_measurement.shape[0]

    emod_mixture = isodata.errormodel["measured"]
    VAms = [
        calcratiocov(mixture_measurement[i, :], emod_mixture, deno) for i in range(n_m)
    ]
    emod_spike = isodata.errormodel["measured"]
    VAts = [calcratiocov(spike_measurement[i, :], emod_spike, deno) for i in range(n_t)]

    n = An[invrat]
    P = AP[invrat]
    t = At[:, invrat]
    m = Am[:, invrat]

    Vms = [V[np.ix_(invrat, invrat)] for V in VAms]
    Vts = [V[np.ix_(invrat, invrat)] for V in VAts]

    Sms = [np.linalg.inv(V) for V in Vms]
    Sts = [np.linalg.inv(V) for V in Vts]

    # form initial guess of model parameters. guess a 50-50 mix, with no fractionation
    prop0 = 0.5
    lambda0 = realproptoratioprop(prop0, At[0, :], An) * np.ones(m.shape[0])
    beta0 = 0.0 * np.ones(m.shape[0])
    betaT0 = 0.0 * np.ones(t.shape[0])
    T0 = t[0, :]
    z0 = np.concatenate((lambda0, beta0, betaT0, T0))
    df = (t.shape[0] + m.shape[0]) * len(invrat) - len(z0)  # degrees of freedom

    res = minimize(
        objective,
        z0,
        args=(m, t, P, n, Sms, Sts, n_m, n_t),
        jac=True,
        tol=1e-16,
        options={"disp": False, "gtol": 1e-8, "eps": 1e-12},
    )
    z = res.x
    misfit = res.fun
    lambda_, beta, betat, T = z_to_params(z, P, n_m, n_t)

    # Reconstruct spike vector
    calibrated_spike = np.zeros_like(spike_measurement[0, :])
    calibrated_spike[deno] = 1.0
    calibrated_spike[nume] = T

    # For isotopes that were not used in inversion, work out an expectation based on known betat
    isonum = np.arange(isodata.nisos())
    unused = np.array(list(set(isonum).difference(set(isoinv))))
    if len(unused) > 0:
        expected_spike_measurement = np.mean(spike_measurement, axis=0)
        expected_betat = np.mean(betat)
        expected_spike = expected_spike_measurement * np.exp(
            -np.log(isodata.mass) * expected_betat
        )
        expected_spike = normalise_composition(expected_spike)
        expected_unused = expected_spike[unused] / expected_spike[deno]
        calibrated_spike[unused] = expected_unused

    calibrated_spike = normalise_composition(calibrated_spike)

    AT = isodata.ratio(calibrated_spike, deno)
    prop = [ratioproptorealprop(l, AT, An) for l in lambda_]

    out = {
        "calibrated_spike": calibrated_spike,
        "prop_mixture": prop,
        "beta_mixture": beta,
        "beta_spike": betat,
        "misfit": misfit,
        "df": df,
    }

    return out


def objective(z, m, t, P, n, Wm, Wt, n_m, n_t):
    """The objective function and its Jacobian for the chi-squared minimization."""
    me, te = mt_expected(z, P, n, n_m, n_t)
    res_m = m - me
    res_t = t - te

    obs = []
    for i in range(res_m.shape[0]):
        rm = res_m[i, :][np.newaxis, :]
        obs.append((rm @ Wm[i] @ rm.T)[0][0])
    for i in range(res_t.shape[0]):
        rt = res_t[i, :][np.newaxis, :]
        obs.append((rt @ Wt[i] @ rt.T)[0][0])
    ob = sum(obs)

    dmdz, dtdz = dmt_expected_dz(z, P, n, n_m, n_t)

    dob_dzs = []
    for i in range(res_m.shape[0]):
        rm = res_m[i, :][np.newaxis, :]
        dmidz = dmdz[i, :, :]
        dob_dzs.append(-(2 * rm @ Wm[i] @ dmidz)[0])

    for i in range(res_t.shape[0]):
        rt = res_t[i, :][np.newaxis, :]
        dtidz = dtdz[i, :, :]
        dob_dzs.append(-(2 * rt @ Wt[i] @ dtidz)[0])

    dob_dz = np.vstack(dob_dzs)
    dob_dz = np.sum(dob_dz, axis=0)

    return ob, dob_dz


def individual_m_expected(lambda_, beta, T, P, n):
    """Mixture measurement."""
    return np.exp(beta * P) * (lambda_ * T + (1 - lambda_) * n)


def dindividual_m_expected_dlambda(lambda_, beta, T, P, n):
    """dm/dlambda."""
    return np.exp(beta * P) * (T - n)


def dindividual_m_expected_dbeta(lambda_, beta, T, P, n):
    """dm/dbeta."""
    return P * np.exp(beta * P) * (lambda_ * T + (1 - lambda_) * n)


def dindividual_m_expected_dT(lambda_, beta, T, P, n):
    """dm/dT."""
    return np.diag(np.exp(beta * P) * lambda_)


def individual_t_expected(betat, T, P):
    """Spike measurement."""
    return np.exp(betat * P) * T


def dindividual_t_expected_dbetat(betat, T, P):
    """dt/dbetat."""
    return P * np.exp(betat * P) * T


def dindividual_t_expected_dT(betat, T, P):
    """dt/dT."""
    return np.diag(np.exp(betat * P))


def z_to_params(z, P, n_m, n_t):
    """Convert single vectors of unknowns z into separate vectors of unknowns."""
    n_ratios = len(P)
    lambda_ = z[0:n_m]
    beta = z[n_m : 2 * n_m]
    betat = z[2 * n_m : 2 * n_m + n_t]
    T = z[2 * n_m + n_t :]
    return lambda_, beta, betat, T


def mt_expected(z, P, n, n_m, n_t):
    """Expected spike and mixture measurements given model parameters in z."""
    lambda_, beta, betat, T = z_to_params(z, P, n_m, n_t)

    m = np.zeros((n_m, len(P)))
    for i in range(n_m):
        m[i, :] = individual_m_expected(lambda_[i], beta[i], T, P, n)

    t = np.zeros((n_t, len(P)))
    for i in range(n_t):
        t[i, :] = individual_t_expected(betat[i], T, P)
    return m, t


def dmt_expected_dz(z, P, n, n_m, n_t):
    """Derivative of expected spike and mixture measurements wrt model parameters z."""
    lambda_, beta, betat, T = z_to_params(z, P, n_m, n_t)

    dmdz = np.zeros((n_m, len(P), len(z)))
    for i in range(n_m):
        dmdz[i, :, i] = dindividual_m_expected_dlambda(lambda_[i], beta[i], T, P, n)
        dmdz[i, :, n_m + i] = dindividual_m_expected_dbeta(lambda_[i], beta[i], T, P, n)
        dmdz[i, :, 2 * n_m + n_t :] = dindividual_m_expected_dT(
            lambda_[i], beta[i], T, P, n
        )

    dtdz = np.zeros((n_t, len(P), len(z)))
    for i in range(n_t):
        dtdz[i, :, 2 * n_m + i] = dindividual_t_expected_dbetat(betat[i], T, P)
        dtdz[i, :, 2 * n_m + n_t :] = dindividual_t_expected_dT(betat[i], T, P)

    return dmdz, dtdz


if __name__ == "__main__":
    from .monte import monterun
    from .isodata import IsoData

    isodata = IsoData("Fe")
    n = int(1e3)
    true_spike = np.array([1e-3, 1e-2, 0.4, 0.6])
    true_spike = true_spike / sum(true_spike)

    spike_measurements = monterun(isodata, 1.0, true_spike, alpha=0.0, beta=0.8, n=n)
    mixture1_measurements = monterun(isodata, 0.7, true_spike, alpha=0.0, beta=2.0, n=n)
    mixture2_measurements = monterun(isodata, 0.5, true_spike, alpha=0.0, beta=1.5, n=n)

    spike_measurement = np.mean(spike_measurements, axis=0)
    mixture1_measurement = np.mean(mixture1_measurements, axis=0)
    mixture2_measurement = np.mean(mixture2_measurements, axis=0)

    mixture_measurement = np.vstack([mixture1_measurement, mixture2_measurement])

    out = spike_calibration(isodata, spike_measurement, mixture_measurement)
    print(out)
    print("True spike", true_spike)
    print("Calibrated spike", out["calibrated_spike"])
    print(
        "Difference between true spike and calibrated:",
        out["calibrated_spike"] - true_spike,
    )
