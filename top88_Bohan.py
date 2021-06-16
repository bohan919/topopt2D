# TOPOLOGY OPTIMISATION BASED ON top88.mat WITH LANGELAAR'S AMFILTER FOR 2D BY BOHAN PENG - IMPERIAL COLLEGE LONDON 2021
# DISCLAIMER -                                                             #
# The author reserves all rights but does not guaranty that the code is    #
# free from errors. Furthermore, he shall not be liable in any event       #
# caused by the use of the program.                                        #

import numpy as np
from scipy.sparse import csr_matrix
from pypardiso import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
import sys

def main(nelx,nely,volfrac,penal,rmin,ft):
    # MATERIAL PROPERTIES
    E0 = 1
    Emin = 1e-9
    nu = 0.3

    # PREPARE FINITE ELEMENT ANALYSIS
    A11 = np.array([[12, 3, -6, -3],[3, 12, 3, 0],[-6, 3, 12, -3],[-3, 0, -3, 12]])
    A12 = np.array([[-6, -3, 0, 3],[-3, -6, -3, -6],[0, -3, -6, 3],[3, -6, 3, -6]])
    B11 = np.array([[-4, 3, -2, 9],[3, -4, -9, 4],[-2, -9, -4, -3],[9, 4, -3, -4]])
    B12 = np.array([[2, -3, 4, -9],[-3, 2, 9, -2],[4, 9, 2, 3],[-9, -2, 3, 2]])
    Atop = np.concatenate((A11, A12),axis = 1) 
    Abottom = np.concatenate((A12.T, A11), axis = 1)
    A = np.concatenate((Atop,Abottom), axis = 0)
    Btop = np.concatenate((B11, B12), axis = 1)
    Bbottom = np.concatenate((B12.T, B11), axis = 1)
    B = np.concatenate((Btop, Bbottom), axis = 0)
    KE = 1/(1-nu**2)/24 *(A + nu*B)
    nodenrs = np.reshape(np.arange(1,((nelx+1)*(nely+1)+1)), (1+nelx,1+nely))
    nodenrs = nodenrs.T
    edofVec = np.ravel(nodenrs[0:nely,0:nelx], order='F') *2 + 1
    edofVec = edofVec.reshape((nelx*nely,1))
    edofMat = np.matlib.repmat(edofVec,1,8) + np.matlib.repmat(np.concatenate(([0, 1], 2*nely+np.array([2,3,0,1]), [-2, -1])),nelx*nely,1)
    iK = np.reshape(np.kron(edofMat, np.ones((8,1))).T, (64*nelx*nely,1),order='F')
    jK = np.reshape(np.kron(edofMat, np.ones((1,8))).T, (64*nelx*nely,1),order='F')

    # DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
    F = np.zeros((2*(nely+1)*(nelx+1),1))
    F[1,0] = -1
    U = np.zeros((2*(nely+1)*(nelx+1),1))
    fixeddofs = np.union1d(np.arange(1,2*(nely+1),2),2*(nelx+1)*(nely+1))
    alldofs = np.arange(1,2*(nely+1)*(nelx+1)+1)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    # PREPARE FILTER
    iH = np.ones((nelx*nely*(int(2*(np.ceil(rmin)-1)+1))**2,1))
    jH = np.ones(np.shape(iH))
    sH = np.zeros(np.shape(iH))
    k = 0
    for i1 in range(1,nelx+1):
        for j1 in range(1,nely+1):
            e1 = (i1-1)*nely+j1
            for i2 in range(max(i1-(int(np.ceil(rmin))-1),1), min(i1+(int(np.ceil(rmin))-1),nelx)+1):
                for j2 in range(max(j1-(int(np.ceil(rmin))-1),1), min(j1+(int(np.ceil(rmin))-1),nely)+1):
                    e2 = (i2-1)*nely + j2
                    iH[k] = e1
                    jH[k] = e2
                    sH[k] = max(0, rmin-np.sqrt((i1-i2)**2+(j1-j2)**2))
                    k = k + 1
    H = csr_matrix( (np.squeeze(sH), (np.squeeze(iH.astype(int))-1,np.squeeze(jH.astype(int))-1)))
    Hs = np.sum(H, axis = 1)

    # INITIATE ITERATION
    x = np.matlib.repmat(volfrac,nely,nelx)
    xPhys = x
    loop = 0
    change = 1

    # START ITERATION
    while change > 0.01 and loop<=1000:
        loop = loop + 1
        # FE ANALYSIS
        sK = np.reshape(KE.ravel(order='F')[np.newaxis].T @ (Emin+xPhys.ravel(order = 'F')[np.newaxis]**penal*(E0-Emin)),(64*nelx*nely,1),order='F')
        K = csr_matrix( (np.squeeze(sK), (np.squeeze(iK.astype(int))-1,np.squeeze(jK.astype(int))-1)))
        K = (K + K.T) / 2
        U[freedofs-1,0]=spsolve(K[freedofs-1,:][:,freedofs-1],F[freedofs-1,0])   

        #OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        ce =  np.reshape((np.sum( U[edofMat-1,0]@KE*U[edofMat-1,0] , axis = 1)),(nely, nelx),order='F')
        c = np.sum(np.sum( (Emin+xPhys**penal*(E0-Emin))*ce ))
        dc = -penal*(E0-Emin)*xPhys**(penal-1)*ce
        dv = np.ones((nely, nelx))

        # FILTERING/MODIFICAITON OF SENSITIVITIES
        if ft == 1:
            dc = H @ np.ravel((x * dc), order='F')[np.newaxis].T / Hs / np.maximum(0.001, x).ravel(order='F')[np.newaxis].T
            dc = np.reshape(dc, (nely, nelx), order='F')
            dc = np.asarray(dc)
        elif ft == 2:
            dc = H @ (dc.ravel(order='F')[np.newaxis].T / Hs)
            dc = np.reshape(dc, (nely, nelx), order='F')
            dc = np.asarray(dc)
            dv = H @ (dv.ravel(order='F')[np.newaxis].T / Hs)
            dv = np.reshape(dv, (nely, nelx), order='F')
            dv = np.asarray(dv)

        # OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
        l1 = 0
        l2 = 1e9
        move = 0.2
        while (l2-l1)/(l1+l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew_step1 = np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))
            xnew_step2 = np.minimum(1, xnew_step1)
            xnew_step3 = np.maximum(x - move, xnew_step2)
            xnew = np.maximum(0, xnew_step3)
            if ft == 1:
                xPhys = xnew
            elif ft == 2:
                xPhys = np.asarray(H @ xnew.ravel(order='F')[np.newaxis].T) / np.asarray(Hs)
                xPhys = np.reshape(xPhys,(nely,nelx),order='F')
            if np.sum(xPhys) > volfrac*nelx*nely:
                l1 = lmid
            else:
                l2 = lmid
        change = np.max(np.abs(xnew[:]-x[:]))
        x = xnew

        print("it.: {0} , obj.: {1:.4f}, vol.: {3:.3f}, ch.: {2:.3f}".format(\
					loop, c, change, volfrac))
    return xPhys