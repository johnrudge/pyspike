"""Module which handles the data on isotope systems."""
import csv
import numpy as np
import pkg_resources

# Fundamental constants
elementarycharge = 1.60217646e-19 # coulombs
k = 1.3806504e-23  # m^2 kg s^-2 K^-1, Boltzmann constant

def loadrawdata(filename = None):
    """Read in isotope system datafile and return a python dictionary with data."""
    if filename is None:
        resource_package = __name__
        resource_path = '/'.join(('data', 'maininput.csv'))  # Do not use os.path.join()
        filename = pkg_resources.resource_filename(resource_package, resource_path)
    
    f = open(filename, 'r')
    csvreader = csv.reader(f)
    
    data = {}

    next(csvreader) # ignore header row
    j = 0
    els = []

    for row in csvreader:
        if not len(row[0])==0:
            el = row[0].strip('"')
            j = j + 1
            els.append(el)
            i = 0
            data[el]={'element':el,'isonum':[], 'mass':[], 'standard':[], 'rawspike':[]}
        data[el]['isonum'].append(int(row[1]))
        data[el]['mass'].append(float(row[2]))
        data[el]['standard'].append(float(row[3]))
        rs = [float(r) for r in row[4:] if len(r)>0]
        data[el]['rawspike'].append(rs)
        
    # Renormalise compositions, convert to numpy format    
    for el in els:
        data[el]['isonum'] = np.array(data[el]['isonum'])
        data[el]['mass'] = np.array(data[el]['mass'])
        
        s = np.array(data[el]['standard'])
        data[el]['standard'] = s / sum(s)

        rs = np.array(data[el]['rawspike'])
        rs = rs/rs.sum(axis=0)
        data[el]['rawspike'] = rs.T
        if len(data[el]['rawspike'])==0:
            data[el]['rawspike'] = None
    
    f.close()
    return data

default_data = loadrawdata()

class IsoData():
    """Object which stores data on an isotope system.
    
    Args:
        element (str): the element string (e.g. 'Fe')
    
    Attributes:
        element (str): the element e.g. 'Fe'
        isonum (array): the isotopes of this element e.g. [54, 56, 57, 58]
        mass (array): the atomic masses
        standard (array): composition of a standard as a vector
        spike (array): composition of a double spike used
        isoinv (array): the 4 isotopes to use the inversion
        rawspike (array): composition of single spikes available
        errormodel (dict): dictionary describing the errormodel
    """
    
    def __init__(self, element):
        """Given an element, initialise with some sensible default values.""" 
        if element in default_data.keys():
            default = default_data[element]
            self.element = default['element']
            self.isonum = default['isonum']
            self.mass = default['mass']
            self.standard = default['standard']
            self.rawspike = default['rawspike']
        else:
            self.element = element
            self.isonum = None
            self.mass = None
            self.standard = None
            self.rawspike = None
        self.spike = None
        if self.nisos() == 4:
            self.isoinv = self.isonum
        else:
            self.isoinv = None
        self.errormodel = {}
        self.set_errormodel()
    
    def __repr__(self):
        return "IsoData()"
    
    def __str__(self):
        txt ='\n '.join("%s: %s" % item for item in vars(self).items())
        return txt
    
    def set_element(self, element):
        self.element = element
        
    def set_mass(self, mass):
        self.mass = np.array(mass)
    
    def set_isonum(self, isonum):
        self.isonum = np.array(isonum, dtype=int)
        
    def set_isoinv(self, isoinv):
        self.isoinv = np.array(isoinv, dtype=int)
    
    def set_standard(self, standard):
        self.standard = np.array(standard)

    def set_spike(self, spike):
        self.spike = np.array(spike)
    
    def set_rawspike(self, rawspike):
        self.rawspike = np.array(rawspike)
    
    def isoindex(self, ix):
        """Give the data index corresponding to a given isotope number e.g. 56->1."""
        
        def isonum_to_idx(k):
            quest = np.where(self.isonum ==k)[0]
            if len(quest)==0:
                return k
            else:
                return quest[0]
        
        if ix is None:
            return None
        elif type(ix) == int:
            return isonum_to_idx(ix)
        elif type(ix) == list:
            return [isonum_to_idx(i) for i in ix]
        else:
            # assume a numpy array
            f = np.vectorize(isonum_to_idx)
            return f(ix)
    
    def isoname(self):
        """Names of the isotopes."""
        return [self.element + str(i) for i in self.isonum]
    
    def isolabel(self):
        """Isotope labels for plotting."""
        return ['$^{' +  str(i) +'}$' + self.element for i in self.isonum]
    
    def rawspikelabel(self):
        """Single spike labels for plotting."""
        return ['spike ' + str(i+1) for i in range(self.rawspike.shape[0])]
    
    def nisos(self):
        """Number of isotopes in system."""
        return len(self.isonum)
    
    def nratios(self):
        """Number of isotope ratios to describe system."""
        return self.nisos() - 1
    
    def nrawspikes(self):
        """Number of single spikes available."""
        if self.rawspike is None:
            return 0
        else:
            return self.rawspike.shape[0]
    
    def set_errormodel(self, intensity = 10.0, deltat = 8.0, R = 1e11, T = 300.0, radiogenic = None, measured_type = 'fixed-total'): 
        """Set the error model used for error estimates and monte carlo runs.
        
        Args:
            intensity (float): Total beam intensity in volts
            delta_t (float): Integration time in seconds
            R (float): resistance in Ohms
            T (float): temperature in K
            radiogenic (str): If True put errors on the standard (unspiked) run, if False do not.
                        If None, then 'Pb','Sr','Hf','Os','Nd' are assumed radiogenic, others not.
            measured_type (str): If 'fixed-total' then total beam intensity of mixture is fixed,
                           if 'fixed-sample' then voltage for the sample is fixed.
        """
        if radiogenic is None:
            if self.element in ['Pb','Sr','Hf','Os','Nd']:
                radiogenic = True
            else:
                radiogenic = False
        
        a = 4 * k * T * R / deltat
        b = elementarycharge * R / deltat
        
        nisos = self.nisos()
        
        # by default assume Johnson noise and counting statistics
        self.errormodel['measured']={
            'type': measured_type,
            'intensity': intensity,
            'a': a*np.ones(nisos),
            'b': b*np.ones(nisos),
            'c': 0.0*np.ones(nisos)
            }
        self.errormodel['spike']={
            'type': 'fixed-total',
            'intensity': intensity,
            'a': 0.0*np.ones(nisos),
            'b': 0.0*np.ones(nisos),
            'c': 0.0*np.ones(nisos)
            }
        if radiogenic:
            # if a radiogenic isotope then put errors on the standard (unspiked) run
            self.errormodel['standard']={
                'type': 'fixed-total',
                'intensity': intensity,
                'a': a*np.ones(nisos),
                'b': b*np.ones(nisos),
                'c': 0.0*np.ones(nisos)
                }
        else:
            # otherwise assume no errors on standard composition
            self.errormodel['standard']={
                'type': 'fixed-total',
                'intensity': intensity,
                'a': 0.0*np.ones(nisos),
                'b': 0.0*np.ones(nisos),
                'c': 0.0*np.ones(nisos)
                }

    def set_custom_errormodel(self, errormodel):
        """Set the error model used for error estimates and monte carlo runs.
        
        Args:
            errormodel: A dictionary giving the complete errormodel.
                    
        See IsoData.errormodel for format of dictionary.
        """
        self.errormodel = errormodel
        
    def ratio(self, composition, denominator_isotope):
        """Convert a compositional array into array of isotopic ratios."""
        di = self.isoindex(denominator_isotope)
        ni = np.arange(self.nisos())
        ni = ni[ni != di]
        return composition[...,ni]/composition[...,di,np.newaxis]
    
    def composition(self, data, denominator_isotope):
        """Convert an array of isotopic ratios to an array of compositional vectors."""
        di = self.isoindex(denominator_isotope)
        ni = np.arange(self.nisos())
        ni = ni[ni != di]
        comp_shape = list(data.shape)
        comp_shape[-1] += 1  
        comp = np.ones(comp_shape)
        comp[ni] = data
        comp = normalise_composition(comp)
        return comp
        
    def invrat(self, isoinv = None):
        """Indices of isotope ratios used in inversion."""
        if isoinv is None:
            isoinv = self.isoinv
            
        isoinv = self.isoindex(isoinv) # convert to indices
        
        isonum = np.arange(self.nisos())
        isonum = isonum[isonum != isoinv[0]]
        isonum = np.concatenate((np.array([isoinv[0]]), isonum))

        invrat = np.array([np.where(isonum == i)[0][0] for i in isoinv])
        invrat = invrat[1:] - 1
        return invrat
    
    def rationame(self, denominator_isotope):
        """Names of the isotopic ratios."""
        di = self.isoindex(denominator_isotope)
        ni = np.arange(self.nisos())
        ni = ni[ni != di]
        return np.array([str(i) + self.element +'/' + str(self.isonum[di]) + self.element   for i in self.isonum[ni]])
    
    def ratioidx(self, numerator_isotope, denominator_isotope):
        """The index corresponding to a particular isotopic ratio."""
        di = self.isoindex(denominator_isotope)
        ni = self.isoindex(numerator_isotope)
        if ni<di:
            return ni
        if ni==di:
            return None
        if ni>di:
            return ni-1
        
def normalise_composition(comp):
    """Normalise rows of an array to unit sum, i.e. rows are compositional vectors."""
    s = comp.sum(axis=-1)
    if type(s) is float:
        return comp/s
    else:
        return comp / s[..., np.newaxis]

def ratioproptorealprop(lambda_, ratio_a, ratio_b): 
    """Convert a proportion in ratio space to one per mole."""
    a = 1 + sum(ratio_a)
    b = 1 + sum(ratio_b)
    return lambda_*a / (lambda_*a+ (1-lambda_)*b)

def realproptoratioprop(prop, ratio_a, ratio_b): 
    """Convert a proportion per mole into ratio space."""
    a = 1 + sum(ratio_a)
    b = 1 + sum(ratio_b)
    return prop*b / (prop*b+ (1-prop)*a) 

def ratio(data, isoidx):
    """Convert data to isotope ratios based on choice of isotopes. First index is the denominator index."""
    di = isoidx[0]
    ni = isoidx[1:]
    return data[...,ni]/data[...,di,np.newaxis]


if __name__=="__main__":
    idat = IsoData('Fe')
    print(idat)

