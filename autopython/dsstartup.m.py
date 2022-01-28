#DSSTARTUP   Double spike startup - defines a number default parameters in global ISODATA
#    DSSTARTUP(...)
# This function creates the global variable ISODATA, which contains
# a number of parameters used in all the double spike calculations.
# ISODATA is a structure, with fields of the form ISODATA.(element).(property)
# e.g. ISODATA.Fe.isonum contains the isotope numbers for Fe: [54 56 57 58]
# Other properties include
#   mass -- the atomic masses
#   standard -- the standard composition (got from 'maininput.csv').
#   rawspike -- the composition of available spikes e.g. from Oak Ridge National
#               Labs (also got from 'maininput.csv').
#   errormodel -- the coefficients used in the error model.
#
# Any arguments that are specified are passed to seterrormodel(), the function
# which specifies the error model. This is useful for setting the beam intensities,
# integration times etc.
#
# Example
#    dsstartup();
#    global ISODATA;
#
# See also errorestimate, optimalspike, seterrormodel
    
def dsstartup(varargin = None): 
    global ISODATA
    ISODATA = loadrawdata()
    seterrormodel(varargin[:])
    print('Welcome to the double spike toolbox.    John F. Rudge 2009.')
    print('Default parameters are stored in the global variable ISODATA.')
    print('Type "global ISODATA" to access and set these parameters.')