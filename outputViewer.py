import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# PULL and PLOT the OUTPUT for SANITY CHECK
nelxCheck = 180
nelyCheck = 60

seCheck = np.load("OUTPUT/STRAIN ENERGY/SE_" + str(nelxCheck) + '_' + str(nelyCheck)+'.npy')
xPrintCheck = np.load("OUTPUT/XPRINT/" + str(nelxCheck) + '_' + str(nelyCheck)+'.npy')
xPhysCheck = np.load("OUTPUT/TO_ONLY/TO_" + str(nelxCheck) + '_' + str(nelyCheck)+'.npy')
print(seCheck)

fig, (ax1, ax2, ax3) = plt.subplots(3)
ax1.set_title("xPrint")
im1 = ax1.imshow(0 - xPrintCheck, cmap='gray', \
    interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))       
im1.set_array(0-xPrintCheck)               

ax2.set_title("xPhys")
im2 = ax2.imshow(0 - xPhysCheck, cmap='gray', \
    interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))       
im2.set_array(0-xPhysCheck) 

ax3.set_title("Strain Energy")
im3 = ax3.imshow(seCheck, cmap='viridis', \
    interpolation='none')       
im3.set_array(seCheck)  
plt.draw()
plt.show()