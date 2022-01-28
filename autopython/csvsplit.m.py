import numpy as np
    
def csvsplit(line = None): 
    # This is a replacement for the simple command
#     regexp(tline,',','split')
# which doesn't seem to work on versions of MATLAB before 2007b
# Cumbersome code -- please improve!
    
    splitted2 = regexp(line,'([^,]*),?()|,','match')
    splitted = splitted2
    for m in np.arange(1,len(splitted2)+1).reshape(-1):
        spit = regexp(splitted2[m],'[^,]*','match')
        if len(spit)==0:
            splitted[m] = ''
        else:
            splitted[m] = spit[0]
    
    if splitted2[end()]==',':
        splitted[len[splitted2] + 1] = ''
    
    return splitted