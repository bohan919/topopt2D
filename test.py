import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# import math
# import AMFilter
# import scipy.sparse as sps
import top88_BohanAM
import top88_Bohan

# xPrint_topopt = top88_Bohan.main(200,100,0.4,2, 2, 2)
# np.save('xPrint_200_100.npy', xPrint_topopt) # save
xPrint_AM = top88_BohanAM.main(200,100,0.4,2, 2, 3)
np.save('xPrint_200_100AM_heaviside.npy', xPrint_AM) # save
# xPrint_AM2000 = np.load('xPrint_200_100_AM2000.npy')
# xPrint_AM = np.load('xPrint_200_100AM_heaviside.npy') # load
# xPrint_3D = np.squeeze(xPrint_3D)
# difference = xPrint_AM2000-xPrint_AM

# fig, ax = plt.subplots()
# im = ax.imshow(0 - xPrint_AM2000, cmap='gray', \
#     interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))       # REPLACE xPhys with xPrint
# im.set_array(0-xPrint_AM2000)                                                   # REPLACE xPhys with xPrint
# plt.draw()
# plt.show()

fig, ax = plt.subplots()
im = ax.imshow(0 - xPrint_AM, cmap='gray', \
    interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))       # REPLACE xPhys with xPrint
im.set_array(0-xPrint_AM)                                                   # REPLACE xPhys with xPrint
plt.draw()
plt.show()

# fig, ax = plt.subplots()
# im = plt.imshow(difference, cmap='bwr',\
#     interpolation='none', norm=colors.Normalize(vmin=-1, vmax=1))       # REPLACE xPhys with xPrint
# # im.set_array(difference)   
#                                               # REPLACE xPhys with xPrint
# plt.draw()
# plt.colorbar(im, fraction = 0.024, pad=0.04)  
# plt.show()



# PROFILING
# import cProfile
# import pstats
# from pstats import SortKey

# # cProfile.run('top88_Bohan.main(200,100,0.4,5, 2, 2)','cprofilestats1.txt')
# # cProfile.run('top88_BohanAM.main(200,100,0.4,5, 2, 2)','cprofilestats2.txt')
# p = pstats.Stats('cprofilestats2.txt')
# p.sort_stats('tottime').print_stats(50)
