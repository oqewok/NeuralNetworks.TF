import scipy.io
import numpy as np

data = scipy.io.loadmat("E:/data/train/digitStruct.mat")

for i in data:
    if '__' not in i and 'readme' not in i:
        np.savetxt(("output/" + i + ".csv"), data[i], delimiter=',')
