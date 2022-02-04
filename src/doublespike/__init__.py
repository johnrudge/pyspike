"""
The double spike toolbox
========================

Error propagation and data reduction for the double spike technique in mass spectrometry.

The workings of this package are described in:

Rudge J.F., Reynolds B.C., Bourdon B. The double spike toolbox (2009) Chem. Geol. 265:420-431
https://dx.doi.org/10.1016/j.chemgeo.2009.05.010

and at:

https://johnrudge.com/doublespike

Functions:
    cocktail               - Generate double spike cocktail lists
    dsinversion            - Do the double spike inversion for a given set of measurements
    errorcurve             - A plot of error as a function of double spike-sample proportions for a given
                             double spike composition
    errorcurve2            - A plot of error as a function of double spike proportions for a given double
                             spike-sample proportion
    errorcurve2d           - A 2D contour plot of error as a function of double spike composition and
                             double spike-sample proportions
    errorcurveoptimalspike - Plot error curves for the optimal spike composition
    errorestimate          - Calculate the error in the natural fractionation factor or a chosen ratio
                             by linear error propagation
    monterun               - Generate a fake mass spectrometer run by Monte-Carlo simulation
    optimalspike           - Find the optimal double spike composition and double spike-sample mixture
                             proportions
"""

__version__ = '0.9.1'
__author__ = 'John F. Rudge'

from .isodata import IsoData
from .inversion import dsinversion
from .monte import monterun
from .cocktail import cocktail
from .errors import errorestimate, ratiodata, sensitivity
from .optimal import optimalspike
from .plotting import errorcurve, errorcurve2, errorcurve2d, errorcurveoptimalspike
