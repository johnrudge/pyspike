import numpy as np
import numpy.matlib
    
def fwritecell(filename = None,format = None,permission = None,data = None): 
    # FWRITECELL writes formatted data from a cell array to a text file.
    
    # fwritecell(filename, format, data)
#     writes data using the C language conversion specifications in
#     variable "format" as with SPRINTF
#     Example: fwritecell('textfile1.txt','#2d #1d #21s #8.5f',C);
    
    # fwritecell(filename, data)
#     writes data using a fixed width format padded with whitespace. Note
#     that the decimal point for floating point numbers may not be aligned
    
    # Original version of "fwritecell" was written by Ameya Deoras (available on MATLAB central).
# This version is slightly modified to allow permission setting.
    
    if len(varargin) < 3:
        data = format
        format = []
    
    #Open file to write to
    fid = open(filename,permission)
    #Determine new line character for PC or Unix platform
    if ispc:
        nl = sprintf('\r\n')
    else:
        nl = sprintf('\n')
    
    if not len(format)==0 :
        for i in np.arange(1,data.shape[1-1]+1).reshape(-1):
            try:
                fid.write(np.array([format,nl]) % (data[i,:]))
            finally:
                pass
    else:
        #Determine which columns are characters
        dtypes = cellfun(ischar,data(1,:))
        #Initialize the output string
        datastr = ''
        #Create a column of whitespace to separate fields
        sep = np.matlib.repmat(' ',data.shape[1-1],1)
        #Loop through columns and convert to text
        for i in np.arange(1,len(dtypes)+1).reshape(-1):
            try:
                if not dtypes(i) :
                    datastr = np.array([datastr,sep,num2str(cell2mat(data(:,i)))])
                else:
                    datastr = np.array([datastr,sep,char(data(:,i))])
            finally:
                pass
        #Add new line character
        datastr = np.array([datastr,np.matlib.repmat(nl,datastr.shape[1-1],1)])
        try:
            fwrite(fid,np.transpose(datastr),'char')
        finally:
            pass
    
    fid.close()