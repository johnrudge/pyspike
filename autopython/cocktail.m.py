import numpy as np
import numpy.matlib
    
def cocktail(type_ = None,filename = None,elements = None): 
    # COCKTAIL   Generate double spike cocktail lists
#  COCKTAIL(type,filename,elements)
#             type -- type of spike, 'pure' or 'real'. Real spikes, such as those from
#                Oak Ridge National Labs, contain impurities (see 'maininput.csv'
#                or ISODATA.(element).rawspike) for their assumed compositions.
#                By default pure spikes are used.
#             filename -- file to store output (a CSV file).  Default is either 'cookbook_pure.csv'
#                or 'cookbook_real.csv' depending on the type.
#             elements -- which elements to include in the cookbook. Specify as a cell array
#                e.g. {'Ca','Fe'}. Default is all possible elements.
    
    # This generates an exhaustive list of all possible double spikes for specified elements
# sorted in order of error.
    
    # Note that a number of parameters are specified in the global variable ISODATA.
    
    # Example
#   cocktail('real')
    
    # See also dsstartup
    global ISODATA
    # default argument
    if len(ISODATA)==0:
        dsstartup
    
    if (len(varargin) < 1) or len(type_)==0:
        type_ = 'pure'
    
    if (len(varargin) < 2) or len(filename)==0:
        filename = np.array(['cookbook_',type_,'.csv'])
    
    if (len(varargin) < 3):
        elements = fieldnames(ISODATA)
    
    print(np.array(['Writing to ',filename]))
    title = np.array(['A double spike cocktail list: ',type_,' spikes'])
    fwritecell(filename,'%s','w',np.array([title]))
    fwritecell(filename,'%s','a',np.array(['']))
    for i in np.arange(1,len(elements)+1).reshape(-1):
        element = elements[i]
        print(element)
        in_ = getattr(ISODATA,(element))
        if (type_=='pure') or (type_=='real' and in_.nspikes > 1):
            optspike,optprop,opterr,optisoinv,optspikeprop,optppmperamu = optimalspike(element,type_)
            optisoinv = in_.isoindex(optisoinv)
            optisonams = np.array([np.transpose(np.array([in_.isoname[optisoinv(:,1)]])),np.transpose(np.array([in_.isoname[optisoinv(:,2)]])),np.transpose(np.array([in_.isoname[optisoinv(:,3)]])),np.transpose(np.array([in_.isoname[optisoinv(:,4)]]))])
            # write output to file
            isohead = np.transpose(strcat(np.matlib.repmat(np.array(['iso']),4,1),num2str(np.transpose((np.arange(1,4+1))))))
            if type_=='pure':
                output = np.array([optisonams,num2cell(np.array([optspike,optprop(1 - optprop),opterr,optppmperamu]))])
                header = np.array([isohead,in_.isoname,np.array(['spike']),np.array(['sample']),np.array(['error']),np.array(['ppmperamu'])])
                fwritecell(filename,np.array([np.matlib.repmat('%s,',1,4),np.matlib.repmat('%s,',1,in_.nisos),'%s,%s,%s,%s']),'a',header)
                fwritecell(filename,np.array([np.matlib.repmat('%s,',1,4),np.matlib.repmat('%f,',1,in_.nisos),'%f,%f,%f,%f']),'a',output)
                fwritecell(filename,'%s','a',np.array(['']))
            else:
                spikehead = np.transpose(strcat(np.matlib.repmat(np.array(['spike']),in_.nspikes,1),num2str(np.transpose((np.arange(1,in_.nspikes+1))))))
                output = np.array([optisonams,num2cell(np.array([optspike,optspikeprop,optprop(1 - optprop),opterr,optppmperamu]))])
                header = np.array([isohead,in_.isoname,spikehead,np.array(['spike']),np.array(['sample']),np.array(['error']),np.array(['ppmperamu'])])
                fwritecell(filename,np.array([np.matlib.repmat('%s,',1,4),np.matlib.repmat('%s,',1,in_.nisos),np.matlib.repmat('%s,',1,in_.nspikes),'%s,%s,%s,%s']),'a',header)
                fwritecell(filename,np.array([np.matlib.repmat('%s,',1,4),np.matlib.repmat('%f,',1,in_.nisos),np.matlib.repmat('%f,',1,in_.nspikes),'%f,%f,%f,%f']),'a',output)
                fwritecell(filename,'%s','a',np.array(['']))
    
    print(np.array(['Output written to ',filename]))