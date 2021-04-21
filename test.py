# import numpy as np
# import numpy.matlib
# import math
# import AMFilter
# import scipy.sparse as sps
import top88_BohanAM
# import top88_Bohan

# top88_BohanAM.main(60,30,0.4,5, 2, 3)

import cProfile
import pstats
from pstats import SortKey

cProfile.run('top88_BohanAM.main(60,30,0.4,5, 2, 3)','cprofilestats.txt')
p = pstats.Stats('cprofilestats.txt')
p.sort_stats('tottime').print_stats(50)

# x = np.array([[1,0,1,1],[1,1,0,1],[1,1,0,1]])
# dc = np.array([[0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3],[0.4, 0.4, 0.4, 0.4]])
# dv = np.array([[0.1, 0.1, 0.1, 0.1],[0.2, 0.2, 0.2, 0.2],[0.3, 0.3, 0.3, 0.3]])
# baseplate = 'N'
# nelx = 4
# nely = 3

#top88_BohanAM.main(100, 50, 0.4, 3, 5.4, 2)
# nelx = 3
# nelz = 5
# a,b,c = np.meshgrid((nelx, 0, np.arange(nelz+1)))

# print(a)

# KE = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# xPhys = np.array([[2, 2], [2, 2], [2, 2]])
# panel = 3
# E0 = 1
# Emin = 1e-9
# print(KE.ravel(order='F')[np.newaxis].T @ (Emin+xPhys.ravel(order = 'F')[np.newaxis]**panel*(E0-Emin)))

# xprint, Out = AMFilter.AMFilter(x, baseplate,nelx,nely,dc,dv)
# print(xprint)
# print(Out[0])
# print(Out[1])

# nelx = 180
# nely = 60
# nodenrs = np.reshape(np.arange(1,((nelx+1)*(nely+1)+1)), (1+nelx,1+nely))
# nodenrs = nodenrs.T
# edofVec = np.ravel(nodenrs[0:nely,0:nelx], order='F') *2 + 1
# edofVec = edofVec.reshape((nelx*nely,1))
# edofMat = np.matlib.repmat(edofVec,1,8) + np.matlib.repmat(np.concatenate(([0, 1], 2*nely+np.array([2,3,0,1]), [-2, -1])),nelx*nely,1)
# iK = np.reshape(np.kron(edofMat, np.ones((8,1))).T, (64*nelx*nely,1),order='F')
# fixeddofs = np.union1d(np.arange(1,2*(nely+1),2),2*(nelx+1)*(nely+1))
# alldofs = np.arange(1,2*(nely+1)*(nelx+1)+1)
# freedofs = np.setdiff1d(alldofs, fixeddofs)
# rmin = 5.4  
# iH = np.ones((nelx*nely*(int(2*(np.ceil(rmin)-1)+1))**2,1))
# jH = np.ones(np.shape(iH))
# sH = np.zeros(np.shape(iH))
# k = 0
# for i1 in range(1,nelx+1):
#     for j1 in range(1,nely+1):
#         e1 = (i1-1)*nely+j1
#         for i2 in range(max(i1-(np.ceil(rmin)-1),1), min(i1+(np.ceil(rmin)-1),nelx)+1):
#             for j2 in range(max(j1-(np.ceil(rmin)-1),1), min(j1+(np.ceil(rmin)-1),nely)+1):
#                 e2 = (i2-1)*nely + j2
#                 iH[k] = int(e1)
#                 jH[k] = int(e2)
#                 sH[k] = max(0, rmin-math.sqrt((i1-i2)**2+(j1-j2)**2))
#                 k = k + 1

# H = sps.coo_matrix( (np.squeeze(sH), (np.squeeze(iH.astype(int))-1,np.squeeze(jH.astype(int))-1)))
# Hs = np.sum(H, axis = 1)
# nu = 0.3
# A11 = np.array([[12, 3, -6, -3],[3, 12, 3, 0],[-6, 3, 12, -3],[-3, 0, -3, 12]])
# A12 = np.array([[-6, -3, 0, 3],[-3, -6, -3, -6],[0, -3, -6, 3],[3, -6, 3, -6]])
# B11 = np.array([[-4, 3, -2, 9],[3, -4, -9, 4],[-2, -9, -4, -3],[9, 4, -3, -4]])
# B12 = np.array([[2, -3, 4, -9],[-3, 2, 9, -2],[4, 9, 2, 3],[-9, -2, 3, 2]])
# Atop = np.concatenate((A11, A12),axis = 1) 
# Abottom = np.concatenate((A12.T, A11), axis = 1)
# A = np.concatenate((Atop,Abottom), axis = 0)
# Btop = np.concatenate((B11, B12), axis = 1)
# Bbottom = np.concatenate((B12.T, B11), axis = 1)
# B = np.concatenate((Btop, Bbottom), axis = 0)
# KE = 1/(1-nu**2)/24 *(A + nu*B)
# volfrac = 0.4
# x = np.matlib.repmat(volfrac,nely,nelx)
# xPhys = x
# print(xPhys.ravel()[np.newaxis])