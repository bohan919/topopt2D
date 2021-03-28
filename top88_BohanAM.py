import numpy as np
# import scipy.sparse as sps
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
# import sys
import AMFilter

def main(nelx,nely,volfrac,penal,rmin,ft):
    # MATERIAL PROPERTIES
    E0 = 1
    Emin = 1e-9
    nu = 0.3

    # PRINT DIRECTION
    baseplate = 'S'

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
    
    # H = sps.coo_matrix( (np.squeeze(sH), (np.squeeze(iH.astype(int))-1,np.squeeze(jH.astype(int))-1)))
    H = csr_matrix( (np.squeeze(sH), (np.squeeze(iH.astype(int))-1,np.squeeze(jH.astype(int))-1)))
    Hs = np.sum(H, axis = 1)

    # INITIATE ITERATION
    x = np.matlib.repmat(volfrac,nely,nelx)
    xPhys = x
    beta = 1
    if ft == 1 or ft == 2:
        xPhys = x
        baseplate = 'S'  # USER DEFINED PRINT DIRECTION
        xPrint, _ = AMFilter.AMFilter(xPhys, baseplate)
    elif ft == 3:
        xTilde = x
        xPhys = 1 - np.exp(-beta * xTilde) + xTilde * np.exp(-beta)
        baseplate = 'S'  # USER DEFINED PRINT DIRECTION
        xPrint, _ = AMFilter.AMFilter(xPhys, baseplate)
    ####### AM CALL 1 #####
    # xPrint,_ = AMFilter.AMFilter(xPhys, baseplate)
    #######################
    loop = 0
    loopbeta = 0
    change = 1

    # Plot to screen
    # Initialize plot and plot the initial design
    plt.ion()  # Ensure that redrawing is possible
    fig, ax = plt.subplots()

    # START ITERATION
    while change > 0.01 and loop<=500:
        loop = loop + 1
        # FE ANALYSIS
        sK = np.reshape(KE.ravel(order='F')[np.newaxis].T @ (Emin+xPrint.ravel(order = 'F')[np.newaxis]**penal*(E0-Emin)),(64*nelx*nely,1),order='F')
        # K = sps.csr_matrix((np.squeeze(sK), (np.squeeze(iK.astype(int)) - 1, np.squeeze(jK.astype(int)) - 1)))
        K = csr_matrix( (np.squeeze(sK), (np.squeeze(iK.astype(int))-1,np.squeeze(jK.astype(int))-1)))
        K = (K + K.T) / 2
        U[freedofs-1,0]=spsolve(K[freedofs-1,:][:,freedofs-1],F[freedofs-1,0])   
        #U[freedofs-1] = np.linalg.lstsq(K[freedofs-1, :][:, freedofs-1],F[freedofs-1].T)[0]

        #OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
        ce =  np.reshape((np.sum( U[edofMat-1,0]@KE*U[edofMat-1,0] , axis = 1)),(nely, nelx),order='F')
        c = np.sum(np.sum( (Emin+xPrint**penal*(E0-Emin))*ce ))   # REPLACE xPhys with xPrint
        dc = -penal*(E0-Emin)*xPrint**(penal-1)*ce                # REPLACE xPhys with xPrint
        dv = np.ones((nely, nelx))

        # TRANSFORM SENSITIVITIES BEFORE FILTERING
        ######### AMFILTER CALL 2 #########
        xPrint, senS = AMFilter.AMFilter(xPhys, baseplate, dc, dv)
        dc = senS[0]
        dv = senS[1]
        ###################################

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
        elif ft == 3:
            dx = beta * np.exp(-beta * xTilde) + np.exp(-beta)
            dc = H @ (dc.ravel(order='F')[np.newaxis].T * dx.ravel(order='F')[np.newaxis].T / Hs)
            dc = np.reshape(dc, (nely, nelx), order='F')
            dc = np.asarray(dc)
            dv = H @ (dv.ravel(order='F')[np.newaxis].T * dx.ravel(order='F')[np.newaxis].T / Hs)
            dv = np.reshape(dv, (nely, nelx), order='F')
            dv = np.asarray(dv)
        # OPTIMALITY CRITERIA UPDATE OF DESIGN VARIABLES AND PHYSICAL DENSITIES
        l1 = 0
        l2 = 1e9
        move = 0.05
        while (l2-l1)/(l1+l2) > 1e-3:
            lmid = 0.5 * (l2 + l1)
            xnew_step1 = np.minimum(x + move, x * np.sqrt(-dc / dv / lmid))
            xnew_step2 = np.minimum(1, xnew_step1)
            xnew_step3 = np.maximum(x - move, xnew_step2)
            xnew = np.maximum(0, xnew_step3)
            # ft==1 -> sens, ft==2 -> dens (Density Filter is preferred)
            if ft == 1:
                xPhys = xnew
            elif ft == 2:
                xPhys = np.asarray(H @ xnew.ravel(order='F')[np.newaxis].T) / np.asarray(Hs)
                xPhys = np.reshape(xPhys, (nely, nelx), order='F')
            elif ft == 3:
                xTilde = np.asarray(H @ xnew.ravel(order='F')[np.newaxis].T) / np.asarray(Hs)
                xTilde = np.reshape(xTilde, (nely, nelx), order='F')
                xPhys = 1 - np.exp(-beta * xTilde) + xTilde * np.exp(-beta)
                
            ######### AMFILTER CALL 1 ######
            xPrint, _ = AMFilter.AMFilter(xPhys, baseplate)
            #################################
            if np.sum(xPrint) > volfrac*nelx*nely: # REPLACE xPhys with xPrint
                l1 = lmid
            else:
                l2 = lmid
        change = np.max(np.abs(xnew[:]-x[:]))
        x = xnew
        if ft == 3 and beta < 512 and (loopbeta >= 50 or change <= 0.01):
            beta = 2 * beta
            loopbeta = 0
            change = 1
            print("Parameter beta increased to {0}. \n".format(beta))
        # Write iteration history to screen (req. Python 2.6 or newer)
        print("it.: {0} , ch.: {1:.3f}, obj.: {2:.4f}".format(\
					loop, change, c))

    # PLOT THE RESULT
    im = ax.imshow(0 - xPrint, cmap='gray', \
        interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))       # REPLACE xPhys with xPrint
    im.set_array(0-xPrint)                                                   # REPLACE xPhys with xPrint
    plt.pause(0.000001)
    plt.draw()
    

    # Make sure the plot stays and that the shell remains	
    input("Press any key...")
