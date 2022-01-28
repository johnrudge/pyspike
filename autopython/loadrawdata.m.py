import numpy as np
import numpy.matlib
    
def loadrawdata(): 
    # main routine for loading in the data about the different isotopic systems, such as standard compositions,
# atomic masses, and raw spike compositions
    
    filename = 'maininput.csv'
    fid = open(filename)
    fgetl(fid)
    
    j = 1
    data = []
    while 1:

        tline = fgetl(fid)
        # bail out when file is finished
        if not ischar(tline) :
            break
        # split line based on commas
#	splitted=regexp(tline,',','split');  # This doesn't work on MATLAB's before 2007b...
        splitted = csvsplit(tline)
        if not len(splitted[0])==0 :
            el = regexprep(splitted[0],'"','')
            element[j] = el
            j = j + 1
            i = 1
        numbers = str2double(np.array([splitted[np.arange(2,end()+1)]]))
        getattr[data,[el]].isonum[i] = numbers(1)
        getattr[data,[el]].mass[i] = numbers(2)
        getattr[data,[el]].standard[i] = numbers(3)
        getattr[data,[el]].rawspike[:,i] = numbers(np.arange(4,end()+1))
        getattr[data,[el]].isoname[i] = np.array([el,num2str(getattr(data,(el)).isonum(i))])
        getattr[data,[el]].isolabel[i] = np.array(['^{',num2str(getattr(data,(el)).isonum(i)),'}',el])
        i = i + 1

    
    # Renormalise compositions, add a few useful pieces of information
    for j in np.arange(1,len(element)+1).reshape(-1):
        el = element[j]
        getattr(data,(el)).element = el
        getattr(data,(el)).standard = getattr(data,(el)).standard / sum(getattr(data,(el)).standard)
        nisos = len(getattr(data,(el)).isonum)
        missingraws = np.any(np.isnan(getattr(data,(el)).rawspike),2)
        getattr(data,(el)).rawspike = getattr(data,(el)).rawspike(not missingraws ,:)
        getattr(data,(el)).rawspike = getattr(data,(el)).rawspike / np.matlib.repmat(np.sum(getattr(data,(el)).rawspike, 2-1),np.array([1,nisos]))
        getattr[data,[el]].isoindex[1,getattr[data,[el]].isonum] = np.arange(1,nisos+1)
        getattr[data,[el]].isoindex[1,np.arange[1,nisos+1]] = np.arange(1,nisos+1)
        getattr[data,[el]].revisoindex[1,getattr[data,[el]].isonum] = getattr(data,(el)).isonum
        getattr[data,[el]].revisoindex[1,np.arange[1,nisos+1]] = getattr(data,(el)).isonum
        getattr(data,(el)).nisos = len(getattr(data,(el)).isonum)
        getattr(data,(el)).nratios = getattr(data,(el)).nisos - 1
        getattr(data,(el)).nspikes = len(getattr(data,(el)).rawspike(:,1))
        for k in np.arange(1,getattr(data,(el)).nspikes+1).reshape(-1):
            getattr[data,[el]].rawspikelabel[k] = np.array(['spike ',num2str(k)])
    
    fid.close()
    return data