## A brief demonstration of the double spike toolbox
# This is a quick guide to the main features of the double spike toolbox.
#
## Startup
# The function _dsstartup_ is used to initialise a number of parameters to default values. These
# are stored in a global variable called ISODATA. This is a structure with fieldnames corresponding
# to the different elements. Information includes the isotope numbers, the atomic masses, the
# standard compositions, and the default error model coefficients. Some of these values are
# loaded in from the file 'maininput.csv' which can modified by the user as necessary.
# _dsstartup_ is called automatically the first time one of the double spike toolbox functions
# is used. The information stored about Fe is shown below.
import os
import matplotlib.pyplot as plt
import numpy as np
dsstartup
global ISODATA
ISODATA.Fe
## Default values
# The default values can be accessed and set through the ISODATA global variable. For example,
# shown below for Fe are the isotope numbers, standard value,
# the atomic masses, and the third and fourth Oak Ridge National Labs (ORNL) spike
# compositions (corresponding to the 57Fe and 58Fe spikes). Note that
# all values are expressed as proportions of each isotope, rather than as isotopic ratios.
ISODATA.Fe.isonum
ISODATA.Fe.standard
ISODATA.Fe.mass
ISODATA.Fe.rawspike(3,:)
ISODATA.Fe.rawspike(4,:)
## 2D Error surfaces 1
# The function _errorcurve2d_ plots a 2D error surface as a contour plot, showing how
# the error varies with both double spike to sample mixture proportions and the proportions in which
# the two single spikes are mixed to make the double spike. The example below is for a 57Fe-58Fe
# double spike, using pure spikes.
plt.figure(1)
errorcurve2d('Fe','pure',np.array([57,58]))
## 2D Error surfaces 2
# Calculations can also be performed using real spikes rather than hypothetical pure spikes. Below
# is an example using the third and fourth ORNL spikes (corresponding to 57Fe and 58Fe spikes).
# This produces Figure 2 in the manuscript. Spike purity usually has only a little effect.
plt.figure(2)
errorcurve2d('Fe','real',np.array([3,4]))
## 2D Error surfaces 3
# Some isotopes have more than four isotopes, and in these cases the isotopes to use in the
# inversion must be specified. Here is an example for Ca, with a 42Ca-48Ca double spike
# and 40Ca, 42Ca, 44Ca, 48Ca used in inversion. The second and sixth ORNL spikes are the 42Ca and 48Ca
# spikes. This produces Figure 5 in the manuscript.
plt.figure(3)
ISODATA.Ca.rawspike(2,:)
ISODATA.Ca.rawspike(6,:)
errorcurve2d('Ca','real',np.array([2,6]),np.array([40,42,44,48]))
## Error curves 1
# The function _errorcurve_ plots the error in either the fractionation factor alpha
# or a chosen ratio as a function of the double spike to sample proportion. Note again that all
# compositions are specified by proportions of each isotope rather than by ratios. Here we look
# at the error curve for a double spike which is 50# 57Fe and 50# 58Fe.
plt.figure(4)
spike = np.array([0,0,0.5,0.5])

errorcurve('Fe',spike)
## Error curves 2
# The function _errorcurve2_ plots the error in either the fractionation factor alpha
# or a chosen ratio as a function of the proportion of the two single spikes that make the double spike.
# The proportion in which double spike and sample are mixed must be specified, as must the single spikes
# to use. We give an example here for a pure 57Fe-58Fe double spike with 50# double spike to 50# sample
plt.figure(5)
errorcurve2('Fe','pure',0.5,np.array([57,58]))
## Optimal spikes 1
# The function _optimalspike_ finds double spike compositions which minimise the error on alpha
# or a chosen ratio. This can be done either for pure spikes, or with the real spikes available
# from Oak Ridge National Labs.
# The following example finds the best 57Fe-58Fe spike using the real spikes available from ORNL.
# The 3rd and 4th spikes correspond to the 57Fe and 58Fe spikes. The optimal double spike turns
# out to be quite close to a 50-50 mix of the available spikes (optspikeprop). The actual double
# spike compositions are in optspike, the optimal double spike-sample mixing proportions in optprop,
# the error estimates in opterr, rescaled error estimates in optppmperamu, and the isotopes that
# were used in the inversion in optisoinv.
optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike('Fe','real',np.array([3,4]))
## Optimal spikes 2
# If the isotopes to spike are not specified, the _optimalspike_ function checks all possible combinations.
# An example for Fe pure spikes is shown below. This produces Table 1 in the manuscript.
optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike('Fe','pure')
## Optimal spikes 3
# An example for Fe ORNL spikes is shown below. This produces Table 2 in the manuscript.
optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike('Fe','real')
## Optimal spikes 4
# By default, _optimalspike_ minimises the error on alpha, but for radiogenic work we often
# wish to minimise the error on a particular ratio. An example of this is Pb. Shown below is the
# result of minimising the error on 206Pb/204Pb. This produces part of Table 3 in the manuscript.
optspike,optprop,opterr,optisoinv,optspikeprop = optimalspike('Pb','pure',[],[],np.array([206,204]))
## Optimal spikes 5
# For elements with more than four isotopes, such as Ca, _optimalspike_
# tries all combinations of four isotopes in the inversion.
# This produces Table 4 in the manuscript. We only display the first 31 rows of the optimal spike
# composition.
optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike('Ca','pure')
optspike(np.arange(1,31+1),:)
## Optimal error curves 1
# The function _errorcurveoptimalspike_ calculates the _optimalspike_, and then plots the _errorcurve_.
plt.figure(6)
errorcurveoptimalspike('Fe','real',np.array([3,4]))
## Optimal error curves 2
# If the isotopes to spike are not specified, all possible combinations are shown.
# This produces Figure 3 in the manuscript.
plt.figure(7)
errorcurveoptimalspike('Fe','real')
## Monte Carlo fake mass spec runs
# _monterun_ performs Monte Carlo fake mass spec runs.
# The example below uses a 50-50 spike-sample mix, the pure 50-50 57Fe-58Fe spike used earlier, a
# natural fractionation of -0.2, an instrumental fractionation of 1.8, with 1000 Monte Carlo samples.
# The first 10 mixture measurements are shown.
measured = monterun('Fe',0.5,spike,- 0.2,1.8,1000)
measured(np.arange(1,10+1),:)
## Double spike inversions
# _dsinversion_ performs the double spike inversion. Here we run
# the double spike inversion on the Monte-Carlo generated data with the chosen spike. We then
# produce a figure showing how the value of alpha varies over the run.
out = dsinversion('Fe',measured,spike)
plt.figure(8)
plt.plot(out.alpha)
plt.ylabel('\alpha')
## Error estimates
# The _errorestimate_ routine estimates the errors by linear error propagation.
# Here we compare the error got from the Monte-Carlo simulation with that
# predicted by linear error propagation. The fact that these are close is a good
# validation of the linear error propagation method.
monteerror = std(out.alpha)
predictederror = errorestimate('Fe',0.5,spike,[],[],- 0.2,1.8)
## Error model 1
# The coefficients of the error model are in contained in the global variable ISODATA.
# The coefficients can be specified for the measured, standard (unspiked run), and double
# spike compositions. See Appendix C of the manuscript for their definition.
ISODATA.Fe.errormodel
ISODATA.Fe.errormodel.measured
## Error model 2
# The function _seterrormodel_ can be used to simply change the intensity and integration time
# for the default error model. In the example below, a doubling of the intensity decreases the error
# by roughly a factor of 1/sqrt(2).
error1 = errorestimate('Fe',0.5,spike,[],[],- 0.2,1.8)
seterrormodel(20,8)

error2 = errorestimate('Fe',0.5,spike,[],[],- 0.2,1.8)
error2 / error1
seterrormodel()

## Error model 3
# The default error model of the double spike toolbox fixes the total voltage of the beams for the overall
# mixture at a given level. When sample-limited it may be more appropriate to consider an error model
# where the voltage for the sample is fixed (see John 2012, J. Anal. At. Spectrom.).
# In the toolbox this can be achieved by changing the error model type from 'fixed-total' to 'fixed-sample'
# The code below recreates Figure 8 of Klaver and Coath 2018, Geostandards and Geoanalytical Research.
isoinv = np.array([58,60,61,62])
spike6062 = np.array([0.0132,0.3295,0.0014,0.6547,0.0012])
spike6162 = np.array([0.0109,0.0081,0.4496,0.5297,0.0017])
seterrormodel()
# fix the sample intensity at 0.5 V for all runs
ISODATA.Ni.errormodel.measured.intensity = 0.5
ISODATA.Ni.errormodel.measured.type = 'fixed-sample'
plt.figure(9)
errorcurve('Ni',spike6062,isoinv)
hold('on')
errorcurve('Ni',spike6162,isoinv)
plt.ylim(np.array([0,0.06]))
plt.legend('^{60}Ni-^{62}Ni spike','^{61}Ni-^{62}Ni spike')
seterrormodel()
