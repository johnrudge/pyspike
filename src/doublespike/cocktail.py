"""Module for computing cocktail lists of optimal spikes."""

import numpy as np
import csv
from .isodata import IsoData, default_data
from .optimal import optimalspike

def cocktail(type_ = 'pure', filename = 'cookbook.csv', isodatas = None): 
    """Generate double spike cocktail lists.
    
    Args:
        type (str): type of spike, 'pure' or 'real'. Real spikes, such as those from
               Oak Ridge National Labs, contain impurities. See IsoData(element).rawspike
               for their assumed compositions. 
        filename (str): file to store output (a CSV file).
        isodatas (list of IsoData): data on elements to use in cocktail list
            e.g. [IsoData['Ca'], IsoData['Fe']]. If None, it calculates for all
            possible elements.
            
    This routine generates an exhaustive list of all possible double spikes for specified
    elements sorted in order of error.

    Example:
        >>> cocktail('real')
        
    See also optimalspike
    """
    if isodatas is None:
        elements = list(default_data.keys())
        isodatas = [IsoData(el) for el in elements]
        
    print('Writing to '+filename)
    f = open(filename, 'w')
    writer = csv.writer(f)
    
    title = 'A double spike cocktail list: '+type_+' spikes'
    writer.writerow([title])
    writer.writerow([])
    
    for isodata in isodatas:
        element = isodata.element
        print(element)
        isodata = IsoData(element)
        isodata.set_errormodel()
        if (type_=='pure') or (type_=='real' and isodata.nrawspikes() > 1):
            os = optimalspike(isodata,type_)
            optspike = os['optspike']
            opterr = os['opterr']
            optisoinv = os['optisoinv']
            optprop = os['optprop']
            optspikeprop = os['optspikeprop']
            optppmperamu = os['optppmperamu']

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
    #cocktail('pure', filename='cocktail_pure.csv', isodatas=[IsoData('Fe'),IsoData('Ca')])
    cocktail('real', filename='cocktail_real.csv', isodatas=[IsoData('Fe'),IsoData('Ca')])
