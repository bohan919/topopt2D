import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# PULL and PLOT the OUTPUT for SANITY CHECK
nelxCheck = 128
nelyCheck = 128

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