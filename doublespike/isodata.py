import csv
import numpy as np

def loadrawdata(filename = 'data/maininput.csv'):
    # read in isotope system datafile and return a python dictionary with data
    data = {}

    f = open(filename)
    csvreader = csv.reader(f)

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
        # give the data index corresponding to a given isotope number
        if type(ix) == int:
            return np.where(self.isonum ==ix)[0][0]
        else:
            # assume an array or list
            return np.array([np.where(self.isonum ==i)[0][0] for i in ix])
        
    def isoname(self):
        # names of the isotopes
        return [self.element + str(i) for i in self.isonum]
    
    def isolabel(self):
        # isotope labels for plotting
        return ['^{' +  str(i) +'}' + self.element for i in self.isonum]
    
    def nisos(self):
        # number of isotopes in system
        return len(self.isonum)
    
    def nratios(self):
        # number of isotope ratios to describe system
        return self.nisos() - 1
        
if __name__=="__main__":
    idat = IsoData('Fe')
    print(idat.isoindex([56,57,56]))
    print(idat.isolabel())
    print(idat.nisos())
    print(idat.nratios())

