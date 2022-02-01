import numpy as np
import csv
from .isodata import IsoData, default_data
from .errors import optimalspike

def cocktail(type_ = 'pure', filename = 'cookbook.csv', elements = None): 
    """COCKTAIL   Generate double spike cocktail lists
        COCKTAIL(type,filename,elements)
            type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
               Oak Ridge National Labs, contain impurities (see 'maininput.csv'
               or ISODATA.(element).rawspike) for their assumed compositions.
               By default pure spikes are used.
            filename -- file to store output (a CSV file).  Default is either 'cookbook_pure.csv'
               or 'cookbook_real.csv' depending on the type.
            elements -- which elements to include in the cookbook. Specify as a cell array
               e.g. {'Ca','Fe'}. Default is all possible elements.
    
    This generates an exhaustive list of all possible double spikes for specified elements
sorted in order of error.
    
    Note that a number of parameters are specified in the global variable ISODATA.
    
    Example
        cocktail('real')"""
    
    
    if elements is None:
        elements = list(default_data.keys())
    
    print('Writing to '+filename)
    f = open(filename, 'w')
    writer = csv.writer(f)
    
    title = 'A double spike cocktail list: '+type_+' spikes'
    writer.writerow([title])
    writer.writerow([])
    
    for element in elements:
        print(element)
        isodata = IsoData(element)
        isodata.set_errormodel()
        if (type_=='pure') or (type_=='real' and isodata.nrawspikes() > 1):
            optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike(isodata,type_)
            optisoinv = isodata.isoindex(optisoinv)
            
            isoname = isodata.isoname()
            
            def iname(i):
                return isoname[i]
            fv = np.vectorize(iname)
            optisonams = fv(optisoinv)
            
            # write output to file
            isohead = ['iso1','iso2','iso3','iso4']
            
            if type_=='pure':
                header = isohead + isoname + ['spike','sample','error','ppmperamu']
                writer.writerow(header)
                
                for i in range(len(opterr)):
                    line = list(optisonams[i,:]) + list(optspike[i,:]) + list([optprop[i],1-optprop[i]])+list([opterr[i]])+list([optppmperamu[i]])
                    writer.writerow(line)
                writer.writerow([])

            else:
                spikename = ['spike' + str(i+1) for i in range(isodata.rawspike.shape[0])]
                
                header = isohead + isoname + spikename + ['spike','sample','error','ppmperamu']
                writer.writerow(header)
                
                for i in range(len(opterr)):
                    line = list(optisonams[i,:]) + list(optspike[i,:]) + list(optspikeprop[i,:]) + list([optprop[i],1-optprop[i]])+list([opterr[i]])+list([optppmperamu[i]])
                    writer.writerow(line)
                writer.writerow([])
    
    f.close()
    print('Output written to '+filename)
    
    
if __name__=="__main__":
    cocktail('real', filename='cocktail_real.csv', elements=['Fe','Ca'])
