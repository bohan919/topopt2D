# This script generates the 'strain energy field (at first iteration) -- support-free, TO-ed result' pairs

import numpy as np
import top88_BohanAM

import matplotlib.pyplot as plt
from matplotlib import colors
import top88_Bohan

# input: (nelx,nely,volfrac,penal,rmin,ft,bc)
#       ft - 1: sensitivity filter
#            2: density filter
#            3: Heaviside Filter
#       bc - 1: Half-MBB
#            2: Cantilever with vertical downward load applied at the bottom-right corner

from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...
inputs = range(1,5) # aspect ratios of 1:1, 2:1, 3:1, 4:1
def processInput(i):
    for nely in range(50, 151):
        nelx = i*nely
        # nely = 60
        volfrac = 0.4
        rmin = 0.04*nelx  # same setting for rmin as the 88-line topopt code

        filepathSE = "OUTPUT/STRAIN ENERGY/SE_" + str(nelx) + '_' + str(nely)
        filepathxPrint = "OUTPUT/XPRINT/" + str(nelx) + '_' + str(nely)
        filepathxPhys = "OUTPUT/TO_ONLY/TO_" + str(nelx) + '_' + str(nely)

        xPrint_AM, se = top88_BohanAM.main(nelx,nely,volfrac, 3, rmin, 2, 1)
        xPhys = top88_Bohan.main(nelx,nely,volfrac, 3, rmin, 2)

        np.save(filepathSE,se)
        np.save(filepathxPrint, xPrint_AM)
        np.save(filepathxPhys, xPhys)
    return
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
