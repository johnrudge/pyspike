import csv
import numpy as np
import pkg_resources

# Fundamental constants
elementarycharge = 1.60217646e-19 # coulombs
k = 1.3806504e-23  # m^2 kg s^-2 K^-1, Boltzmann constant

def loadrawdata(filename = None):
    # read in isotope system datafile and return a python dictionary with data
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
    return data

default_data = loadrawdata()

class IsoData():
    def __init__(self, element):
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
        self.errormodel = {}
    
    def __repr__(self):
        return "IsoData()"
    
    def __str__(self):
        txt = "element: " + str(self.element) + "\n" \
            + "isonum: " + str(self.isonum) + "\n" \
            + "mass: " + str(self.mass) + "\n" \
            + "standard: " + str(self.standard) + "\n" \
            + "spike: " + str(self.spike)
        return txt
    
    def set_element(self, element):
        self.element = element
        
    def set_mass(self, mass):
        self.mass = np.array(mass)
    
    def set_isonum(self, isonum):
        self.isonum = np.array(isonum, dtype=int)
    
    def set_standard(self, standard):
        self.standard = np.array(standard)

    def set_spike(self, spike):
        self.spike = np.array(spike)
    
    def set_rawspike(self, rawspike):
        self.rawspike = np.array(rawspike)
    
    def isoindex(self, ix):
        """give the data index corresponding to a given isotope number"""
        if ix is None:
            return None
        elif type(ix) == int:
            return isonum_to_idx(self.isonum, ix)
        else:
            # assume an array or list
            return np.array([isonum_to_idx(self.isonum, i) for i in ix])
        
    def isoname(self):
        """names of the isotopes"""
        return [self.element + str(i) for i in self.isonum]
    
    def isolabel(self):
        """isotope labels for plotting"""
        return ['$^{' +  str(i) +'}$' + self.element for i in self.isonum]
    
    def rawspikelabel(self):
        return ['spike ' + str(i+1) for i in range(self.rawspike.shape[0])]
    
    def nisos(self):
        """number of isotopes in system"""
        return len(self.isonum)
    
    def nratios(self):
        """number of isotope ratios to describe system"""
        return self.nisos() - 1
    
    def nrawspikes(self):
        """number of isotope ratios to describe system"""
        return self.rawspike.shape[0]
    
    def set_errormodel(self, intensity = 10, deltat = 8, R = 1e11, T = 300, radiogenic = None, measured_type = 'fixed-total'): 
        """Set the error model used for error estimates and monte carlo runs.
        
            Parameters:
                    intensity: Total beam intensity in volts
                    delta_t: Integration time in seconds
                    R: resistance in Ohms
                    T: temperature in K
                    radiogenic: If True put errors on the standard (unspiked) run, if False do not.
                                If None, then 'Pb','Sr','Hf','Os','Nd' are assumed radiogenic, others not.
                    measured_type: If 'fixed-total' then total beam intensity of mixture is fixed,
                                   if 'fixed-sample' then voltage for the sample is fixed"""
        
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
        
            Parameters:
                    errormodel: A dictionary giving the complete errormodel.
                    
            See IsoData.errormodel for format of dictionary."""
        self.errormodel = errormodel

def isonum_to_idx(isonum, i):
    quest = np.where(isonum ==i)[0]
    if len(quest)==0:
        return i
    else:
        return quest[0]
        
if __name__=="__main__":
    idat = IsoData('Fe')
    print(idat.isoindex([56,57,56]))
    print(idat.isolabel())
    print(idat.nisos())
    print(idat.nratios())

