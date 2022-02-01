import numpy as np
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
    title = 'A double spike cocktail list: ',type_,' spikes'
    #fwritecell(filename,'%s','w',np.array([title]))
    #fwritecell(filename,'%s','a',np.array(['']))
    for element in elements:
        print(element)
        isodata = IsoData(element)
        isodata.set_errormodel()
        if (type_=='pure') or (type_=='real' and isodat.nrawspikes() > 1):
            optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike(isodata,type_)
            #optisoinv = isodata.isoindex(optisoinv)
            #optisonams = [isodata.isoname[optisoinv[:,0]], isodata.isoname[optisoinv[:,1]], isodata.isoname[optisoinv[:,2]], isodata.isoname[optisoinv[:,3]] ]
            print(optisoinv)
            ## write output to file
            #isohead = np.transpose(strcat(np.matlib.repmat(np.array(['iso']),4,1),num2str(np.transpose((np.arange(1,4+1))))))
            #if type_=='pure':
                #output = np.array([optisonams,num2cell(np.array([optspike,optprop(1 - optprop),opterr,optppmperamu]))])
                #header = np.array([isohead,in_.isoname,np.array(['spike']),np.array(['sample']),np.array(['error']),np.array(['ppmperamu'])])
                #fwritecell(filename,np.array([np.matlib.repmat('%s,',1,4),np.matlib.repmat('%s,',1,in_.nisos),'%s,%s,%s,%s']),'a',header)
                #fwritecell(filename,np.array([np.matlib.repmat('%s,',1,4),np.matlib.repmat('%f,',1,in_.nisos),'%f,%f,%f,%f']),'a',output)
                #fwritecell(filename,'%s','a',np.array(['']))
            #else:
                #spikehead = np.transpose(strcat(np.matlib.repmat(np.array(['spike']),in_.nspikes,1),num2str(np.transpose((np.arange(1,in_.nspikes+1))))))
                #output = np.array([optisonams,num2cell(np.array([optspike,optspikeprop,optprop(1 - optprop),opterr,optppmperamu]))])
                #header = np.array([isohead,in_.isoname,spikehead,np.array(['spike']),np.array(['sample']),np.array(['error']),np.array(['ppmperamu'])])
                #fwritecell(filename,np.array([np.matlib.repmat('%s,',1,4),np.matlib.repmat('%s,',1,in_.nisos),np.matlib.repmat('%s,',1,in_.nspikes),'%s,%s,%s,%s']),'a',header)
                #fwritecell(filename,np.array([np.matlib.repmat('%s,',1,4),np.matlib.repmat('%f,',1,in_.nisos),np.matlib.repmat('%f,',1,in_.nspikes),'%f,%f,%f,%f']),'a',output)
                #fwritecell(filename,'%s','a',np.array(['']))
    
    print('Output written to '+filename)
    
    
if __name__=="__main__":
    cocktail('pure', filename='cocktail_pure.csv', elements=['Fe'])
