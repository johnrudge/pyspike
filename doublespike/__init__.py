__version__ = '0.1.0'
__author__ = 'John Rudge'

from .isodata import IsoData
from .inversion import dsinversion
from .monte import monterun
from .cocktail import cocktail
from .errors import errorestimate, optimalspike
from .plotting import errorcurve, errorcurve2, errorcurve2d, errorcurveoptimalspike
