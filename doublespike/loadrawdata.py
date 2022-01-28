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

data = loadrawdata()
print(data['Fe'])
