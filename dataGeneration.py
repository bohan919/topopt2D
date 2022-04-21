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

nelx = 30
nely = 20

filepathSE = "OUTPUT/STRAIN ENERGY/SE_" + str(nelx) + '_' + str(nely)
filepathxPrint = "OUTPUT/XPRINT/" + str(nelx) + '_' + str(nely)
xPrint_AM, se = top88_BohanAM.main(nelx,nely,0.4, 3, 1.5, 2, 1)

np.save(filepathSE,se)
np.save(filepathxPrint, xPrint_AM)

# PULL and PLOT the OUTPUT for SANITY CHECK
nelxCheck = nelx
nelyCheck = nely

seCheck = np.load("OUTPUT/STRAIN ENERGY/SE_" + str(nelxCheck) + '_' + str(nelyCheck)+'.npy')
xPrintCheck = np.load("OUTPUT/XPRINT/" + str(nelxCheck) + '_' + str(nelyCheck)+'.npy')
print(seCheck)

fig, (ax1, ax2) = plt.subplots(2)
ax1.set_title("xPrint")
im1 = ax1.imshow(0 - xPrintCheck, cmap='gray', \
    interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))       
im1.set_array(0-xPrintCheck)                                                   

ax2.set_title("Strain Energy")
im2 = ax2.imshow(seCheck, cmap='viridis', \
    interpolation='none')       
im2.set_array(seCheck)  
plt.draw()
plt.show()