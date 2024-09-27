import numpy as np
import juliet
import os

#target = 'HD 20329'
#target = 'HD 213885'
target = 'TOI-1807'

tim, fl, fle = juliet.utils.get_all_TESS_data(target)

try:
    os.mkdir(os.getcwd() + '/Data/PDCSAP/' + target)
except:
    pass

for i in tim.keys():
    f1 = open(os.getcwd() + '/Data/PDCSAP/' + target.replace(' ', '') + '/' + i + '.dat', 'w')
    f1.write('# Time (BJD)\t\tRelative Flux\t\tError in relative flux\n')
    tim1, fl1, fle1 = tim[i], fl[i], fle[i]
    for j in range(len(tim1)):
        f1.write(str(tim1[j]) + '\t' + str(fl1[j]) + '\t' + str(fle1[j]) + '\n')
    f1.close()