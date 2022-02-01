"""
The double spike toolbox
========================

Error propagation and data reduction for the double spike technique in mass spectrometry.

The workings of this package are described in:

Rudge J.F., Reynolds B.C., Bourdon B. The double spike toolbox (2009) Chem. Geol. 265:420-431
https://dx.doi.org/10.1016/j.chemgeo.2009.05.010

and at:

https://johnrudge.com/doublespike

"""

__version__ = '0.1.0'
__author__ = 'John F. Rudge'

from .isodata import IsoData
from .inversion import dsinversion
from .monte import monterun
from .cocktail import cocktail
from .errors import errorestimate
from .optimal import optimalspike
from .plotting import errorcurve, errorcurve2, errorcurve2d, errorcurveoptimalspike
