"""Module for performing linear error propagation."""

import numpy as np
from .isodata import realproptoratioprop, ratio


def errorestimate(
    isodata, prop, spike=None, isoinv=None, errorratio=None, alpha=0.0, beta=0.0
):
    """Calculate the error in the natural fractionation factor or a chosen ratio by linear error propagation.

    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        prop (float): proportion of double spike in double spike-sample mix.
        spike (array): the isotopic composition of the spike e.g. [0, 0.5, 0, 0.5]
            corresponds to a 50-50 mixture of the 2nd and 4th isotopes
            (56Fe and 58Fe) in the case of Fe.
        isoinv (array): the isotopes used in the inversion, e.g. [54, 56, 57, 58].
        errorratio (array): by default, the error on the natural fractionation
            factor (known as alpha) is given. Instead, the error on a
            particular ratio can be given by setting errorratio. e.g.
            setting errorratio=[58, 56] will give the error on 58Fe/56Fe.
        alpha, beta (floats): there is a small dependance of the error on the fractionation
            factors (instrumental and natural, or alpha and beta). Values of alpha and
            beta can be set here if desired, although the effect on the optimal spikes
            is slight unless the fractionations are very large.

        If spike or isoinv are set as None, values from isodata will be used instead.

    Returns:
        error (float): the error on the fractionation factor, or the specified ratio.
        ppmperamu (float): the error converted to an approximate ppm per atomic mass unit

    Example:
        >>> isodata_fe = IsoData('Fe')
        >>> error, ppmperamu = errorestimate(isodata_fe,0.5,[0,0.5,0,0.5])
    """
    # Get data from isodata if not supplied as arguments
    if spike is None:
        if isodata.spike is None:
            raise Exception("No spike given")
        else:
            spike = isodata.spike

    if isoinv is None:
        if isodata.isoinv is not None:
            isoinv = isodata.isoinv
        else:
            raise Exception("No inversion isotopes set")

    # Ensure working with numpy arrays
    spike = np.array(spike)
    spike = spike / sum(spike)
    isoinv = np.array(isoinv)

    # Convert isotope mass numbers to index numbers
    errorratio = isodata.isoindex(errorratio)
    isoinv = isodata.isoindex(isoinv)

    # Choose denominator isotope from largest spike value
    ix = np.argmax(spike[isoinv])
    deno = isoinv[ix]
    nume = isoinv[isoinv != deno]
    isoinv = np.concatenate((np.array([deno]), nume))
    di = isoinv[0]

    # Convert compositional vectors to isotopic ratios
    z, AP, An, AT, Am, AN, AM = ratiodata(isodata, di, prop, spike, alpha, beta)

    # Convert measured back to a compositional vector
    measured = isodata.composition(Am, di)

    # Calculate the covariance matrices of n, T, and m
    VAn = calcratiocov(isodata.standard, isodata.errormodel["standard"], di, None, prop)
    VAT = calcratiocov(spike, isodata.errormodel["spike"], di, None, prop)
    VAm = calcratiocov(measured, isodata.errormodel["measured"], di, None, prop)

    # invrat gives indices of ratios used in inversion
    invrat = isodata.invrat(isoinv)

    Vz, VAN, _ = fcerrorpropagation(z, AP, An, AT, Am, VAn, VAT, VAm, invrat)

    # Error to return
    if errorratio is None:
        error = np.sqrt(Vz[1, 1])
    else:
        # Now change coordinates to get variance of ratio we're interested in
        newVAN = changedenomcov(AN, VAN, di, errorratio[1])

        isonums = np.arange(isodata.nisos())
        newAni = isonums[isonums != errorratio[1]]
        erat = np.where(errorratio[0] == newAni)[0][0]
        error = np.sqrt(newVAN[erat, erat])

    if errorratio is None:
        ppmperamu = (1000000.0 * error) / np.mean(isodata.mass)  # equation (51)
    else:
        stdratio = isodata.standard[errorratio[0]] / isodata.standard[errorratio[1]]
        massdiff = np.abs(isodata.mass[errorratio[0]] - isodata.mass[errorratio[1]])
        ppmperamu = (1000000.0 * error) / (stdratio * massdiff)  # equation (46)

    return error, ppmperamu


def calcratiocov(composition, errormodel, di, isonorm=None, prop=0.0):
    """Calculate the covariance matrix of the ratios based on the given error model and composition."""
    # di is the isotope with which to denominator
    # isonorm are the isotopes to use in the normalisation
    # prop is the proportion of spike in the spike-sample mix

    if isonorm is None:
        # normalise so that the sum of all beams is the mean intensity
        isonorm = np.arange(composition.shape[0])

    # first normalise composition so it is really a composition (unit sum)
    composition = composition / sum(composition)

    meanbeams = errormodel["intensity"] * composition / sum(composition[isonorm])

    if errormodel["type"] == "fixed-sample":
        meanbeams = meanbeams / (1.0 - prop)

    covbeams = calcbeamcov(meanbeams, errormodel)
    V = covbeamtoratio(meanbeams, covbeams, di)
    return V


def calcbeamcov(meanbeams, errormodel):
    """Calculate beam covariance matrix."""
    beamvar = (
        errormodel["a"]
        + errormodel["b"] * meanbeams
        + errormodel["c"] * (meanbeams**2)
    )  # equation (34)
    return np.diag(beamvar)


def covbeamtoratio(meanbeams, covbeams, di):
    """Convert a covariance matrix for beams to one for ratios."""
    # di is the isotope to denominator with
    # assumes last row and column of M correspond to denominator
    isonums = np.arange(len(meanbeams))
    ni = isonums[isonums != di]
    n = meanbeams[ni]
    d = meanbeams[di]
    ii = np.concatenate((ni, np.array([di])))  # move denominator to end
    M = covbeams[np.ix_(ii, ii)]

    D = np.diag(1 / d * np.ones(len(n)))
    S = -n / (d**2)
    A = np.hstack((D, S[:, np.newaxis]))  # equation (38)

    V = A @ M @ A.T
    return V


def changedenomcov(data, datacov, olddi, newdi):
    """Change denominator of covariance matrix for given set of ratios."""
    nisos = len(data) + 1
    oldni = np.concatenate((np.arange(olddi), np.arange(olddi + 1, nisos)))
    dataplus = np.concatenate((data[0:olddi], np.array([1]), data[olddi:]))

    newni = np.concatenate((np.arange(newdi), np.arange(newdi + 1, nisos)))

    datacovplus = np.zeros((nisos, nisos))
    datacovplus[np.ix_(oldni, oldni)] = datacov

    A = np.eye(nisos) / dataplus[newdi]
    A[:, newdi] = A[:, newdi] - dataplus / (dataplus[newdi] ** 2)
    newdatacovplus = A @ datacovplus @ A.T

    newdatacov = newdatacovplus[np.ix_(newni, newni)]

    return newdatacov


def ratiodata(isodata, di, prop, spike=None, alpha=0.0, beta=0.0):
    """Calculate isotopic ratios describing system.

    Args:
        isodata: object of class IsoData, e.g. IsoData('Fe')
        di (int): denominator isotope, e.g. 56
        prop (float): proportion of double spike in double spike-sample mix.
        spike (array): the isotopic composition of the spike e.g. [0, 0.5, 0, 0.5]
            corresponds to a 50-50 mixture of the 2nd and 4th isotopes
            (56Fe and 58Fe) in the case of Fe. If None taken from isodata.spike
        alpha (float): natural fractionation factor
        beta (float): instrumental fractionation factor

        If spike is set as None, values from isodata will be used instead.

    Returns:
        z (array): vector of model parameters (lambda, alpha, beta)
        AP (array): log of ratio of atomic masses
        An (array): isotopic ratios of standard/ unspiked run
        AT (array): isotopic ratios of spike
        Am (array): isotopic ratios of measurement
        AN (array): isotopic ratios of sample
        AM (array): isotopic ratios of mixture
    """
    if spike is None:
        if isodata.spike is None:
            raise Exception("No spike given")
        else:
            spike = isodata.spike

    # Calculate ratios
    AP = np.log(isodata.ratio(np.array(isodata.mass), di))
    AT = isodata.ratio(np.array(spike), di)
    An = isodata.ratio(np.array(isodata.standard), di)

    # Now calculate sample ratio, lambda etc
    AN = An * np.exp(-AP * alpha)
    lambda_ = realproptoratioprop(prop, AT, AN)
    z = np.array([lambda_, alpha, beta])
    AM = lambda_ * AT + (1 - lambda_) * AN
    Am = AM * np.exp(AP * beta)

    return z, AP, An, AT, Am, AN, AM


def z_sensitivity(z, P, n, T, m):
    """Calculate the derivatives of the z=(lambda,alpha, beta) vector with respect to the input n, T, m ratios."""
    lambda_ = z[0]
    alpha = z[1]
    beta = z[2]

    N = n * np.exp(-P * alpha)
    M = m * np.exp(-P * beta)

    # calculate various Jacobian matrices
    dfdlambda = T - N * (1 + alpha * P)
    dfdu = -N * P
    dfdbeta = M * P
    dfdy = np.array([dfdlambda, dfdu, dfdbeta]).T  # equation (15)
    dfdT = lambda_ * np.eye(3)  # equation (20)
    dfdm = -np.diag(np.exp(-beta * P))  # equation (20)
    dfdn = (1 - lambda_) * np.diag(np.exp(-alpha * P))  # equation (20)

    ## matrix to convert from (lambda, (1-lambda)alpha,beta) to (lambda,alpha,beta), equation (22)
    K = np.array(
        [[1, 0, 0], [(alpha / (1 - lambda_)), (1 / (1 - lambda_)), 0], [0, 0, 1]]
    )
    dzdT = -K @ (np.linalg.solve(dfdy, dfdT))  # equation (19)
    dzdm = -K @ (np.linalg.solve(dfdy, dfdm))  # equation (18)
    dzdn = -K @ (np.linalg.solve(dfdy, dfdn))  # equation (17)

    return dzdn, dzdm, dzdT


def sensitivity(z, AP, An, AT, Am, invrat):
    """Returns the partial derivatives describing the sensitivity of model outputs to model inputs.

    Args:
        z (array): vector of (lambda, alpha, beta)
        AP (array): log of ratio of atomic masses
        An (array): isotopic ratios of standard/ unspiked run
        AT (array): isotopic ratios of spike
        Am (array): isotopic ratios of measurement
        invrat: indices of isotopic ratios used in inversion

    Returns the derivatives:
        dzdAn, dzdAT, dzdAm,
        dANdAn, dANdAT, dANdAm,
        dAMdAn, dAMdAT, dAMdAm,

        where AN are isotopic ratios of sample
        and AM are isotopic ratios of mixture.
    """
    lambda_ = z[0]
    alpha = z[1]
    beta = z[2]
    AM = Am * np.exp(-AP * beta)
    AN = An * np.exp(-AP * alpha)

    P = AP[invrat]
    n = An[invrat]
    T = AT[invrat]
    m = Am[invrat]
    dzdn, dzdm, dzdT = z_sensitivity(z, P, n, T, m)

    # full matrices for all ratios
    nratios = len(An)
    dzdAT = np.zeros((3, nratios))
    dzdAn = np.zeros((3, nratios))
    dzdAm = np.zeros((3, nratios))
    dzdAT[0:3, :][:, invrat] = dzdT
    dzdAn[0:3, :][:, invrat] = dzdn
    dzdAm[0:3, :][:, invrat] = dzdm

    dalphadAT = dzdAT[1, :]
    dalphadAn = dzdAn[1, :]
    dalphadAm = dzdAm[1, :]

    NP = AN * AP
    NP = NP[:, np.newaxis]
    dANdAT = -NP @ dalphadAT[np.newaxis, :]  # equation (26)
    dANdAn = (
        np.diag(np.exp(-alpha * AP)) - NP @ dalphadAn[np.newaxis, :]
    )  # equation (24)
    dANdAm = -NP @ dalphadAm[np.newaxis, :]  # equation (25)

    dbetadAT = dzdAT[2, :]
    dbetadAn = dzdAn[2, :]
    dbetadAm = dzdAm[2, :]

    MP = AM * AP
    MP = MP[:, np.newaxis]
    dAMdAT = -MP @ dbetadAT[np.newaxis, :]  # equation (33)
    dAMdAn = -MP @ dbetadAn[np.newaxis, :]  # equation (31)
    dAMdAm = np.diag(np.exp(-beta * AP)) - MP @ dbetadAm[np.newaxis, :]  # equation (32)

    return dzdAn, dzdAT, dzdAm, dANdAn, dANdAT, dANdAm, dAMdAn, dAMdAT, dAMdAm


def fcerrorpropagation(z, AP, An, AT, Am, VAn, VAT, VAm, invrat):
    """Linear error propagation for the fractionation correction."""
    Vn = VAn[np.ix_(invrat, invrat)]
    VT = VAT[np.ix_(invrat, invrat)]
    Vm = VAm[np.ix_(invrat, invrat)]

    dzdAn, dzdAT, dzdAm, dANdAn, dANdAT, dANdAm, dAMdAn, dAMdAT, dAMdAm = sensitivity(
        z, AP, An, AT, Am, invrat
    )

    # Covariance matix for z=(lambda,beta,alpha), sample, mixture. Equations (16), (23), (30)
    Vz = dzdAn @ VAn @ dzdAn.T + dzdAT @ VAT @ dzdAT.T + dzdAm @ VAm @ dzdAm.T
    VAN = dANdAn @ VAn @ dANdAn.T + dANdAT @ VAT @ dANdAT.T + dANdAm @ VAm @ dANdAm.T
    VAM = dAMdAn @ VAn @ dAMdAn.T + dAMdAT @ VAT @ dAMdAT.T + dAMdAm @ VAm @ dAMdAm.T

    return Vz, VAN, VAM


if __name__ == "__main__":
    from .isodata import IsoData

    isodata = IsoData("Fe")
    isodata.set_spike([0.0, 0.0, 0.5, 0.5])
    alpha_err, ppm_err = errorestimate(isodata, prop=0.5, alpha=-0.2, beta=1.8)
    print(alpha_err, ppm_err)
