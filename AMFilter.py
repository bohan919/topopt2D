# TRANSLATED PYTHON FUNCTION of AMFilter.m by LANGELAAR
# Developed by BOHAN PENG - IMPERIAL COLLEGE LONDON 2021 
# DISCLAIMER -                                                             #
# The author reserves all rights but does not guaranty that the code is    #
# free from errors. Furthermore, he shall not be liable in any event       #
# caused by the use of the program.                                        #

import numpy as np
import numpy.matlib
import scipy.sparse as sps
from copy import deepcopy

def AMFilter(x, baseplate, *args):
    #   Possible uses:
    #   xi = AMfilter(x, baseplate)   idem, with baseplate orientation specified
    #   [xi, df1dx, df2dx,...] = AMfilter(x, baseplate, df1dxi, df2dxi, ...)
    #       This includes also the transformation of design sensitivities
    # where
    #   x : blueprint design (2D array), 0 <= x(i,j) <= 1
    #   xi: printed design (2D array)
    #   baseplate: character indicating baseplate orientation: 'N','E','S','W'
    #              default orientation is 'S'
    #              for 'X', the filter is inactive and just returns the input.
    #   df1dx, df1dxi etc.:  design sensitivity (2D arrays)

    #INTERNAL SETTINGS
    P = 40
    ep = 1e-4 
    xi_0 = 0.5 # parameters for smooth max/min functions

    # INPUT CHECKS
    if baseplate=='X':
    # bypass option: filter does not modify the blueprint design
        xi = x
        varargout = args
        return xi, varargout
    baseplateUpper = baseplate.upper()
    orientation = "SWNE"
    nRot = orientation.find(baseplateUpper)
    nSens = max(0, len(args))

    # Orientation
    x = np.rot90(x, nRot)
    xi = np.zeros(np.shape(x))
    lstArgs = list(args)
    i = 0
    for arg in lstArgs:
        arg = np.rot90(arg, nRot)   
        lstArgs[i] = arg
        i = i+1
    args = tuple(lstArgs)
    nely, nelx = np.shape(x)

    #AM Filter
    Ns = 3
    Q = P + np.log(Ns)/np.log(xi_0)         
    SHIFT = 100*np.finfo(float).tiny **(1/P)
    BACKSHIFT = 0.95*Ns**(1/Q)*SHIFT**(P/Q)
    Xi = np.zeros(np.shape(x))
    keep = np.zeros(np.shape(x))
    sq = np.zeros(np.shape(x))

    # Baseline: identity
    xi[nely-1,:] = x[nely-1,:]
    for i in range(nely-2, -1, -1):
        # compute maxima of current base row
        cbr = np.pad(xi[i+1,:],(1,1),'constant') + SHIFT
        keep[i,:] = cbr[0:nelx]**P + cbr[1:nelx+1]**P + cbr[2:]**P
        Xi[i,:] = keep[i,:]**(1/Q) - BACKSHIFT
        sq[i,:] = np.sqrt( (x[i,:]-Xi[i,:])**2 + ep )
        # set row above to supported value using smooth minimum
        xi[i,:] = 0.5*( (x[i,:]+Xi[i,:]) - sq[i,:] + np.sqrt(ep) )
    
    # SENSITIVITIES
    if nSens != 0:
        dfxi = ()
        for arg in args:
            dfxi = dfxi + (deepcopy(np.reshape(arg,(nely, nelx))),)

        dfx = ()
        for arg in args:
            dfx = dfx + (deepcopy(np.reshape(arg,(nely, nelx))),)
        varLambda = np.zeros((nSens, nelx))

        # from top to base layer:
        for i in range(nely-1):
            # smin sensitivity terms
            dsmindx = 0.5*( 1-(x[i,:]-Xi[i,:])/sq[i,:] )
            dsmindXi = 1-dsmindx
            # smax sensitivity terms
            cbr = np.pad(xi[i+1,:],(1,1),'constant') + SHIFT
            dmx = np.zeros((Ns,nelx))
            for j in range(Ns):
                dmx[j,:] = (P/Q)*keep[i,:]**(1/Q-1)*cbr[0+j:nelx+j:1]**(P-1)
            # rearrange data for quick multiplication
            qj = np.matlib.repmat([[-1],[0],[1]],nelx,1)
            qi = np.matlib.repmat(np.arange(nelx)+1,3,1)
            qi = np.ravel(qi, order='F')
            qi = np.reshape(qi, (3*nelx,1))

            qj = qj + qi
            qs = np.ravel(dmx, order='F')[np.newaxis]
            qsX, qsY = np.shape(qs)
            qs = np.reshape(qs, (qsX*qsY,1))
            dsmaxdxi = sps.csr_matrix( (np.squeeze(qs[1:len(qs)-1]), (np.squeeze(qi[1:len(qi)-1])-1,np.squeeze(qj[1:len(qj)-1])-1) ),dtype=np.float )
            dsmaxdxi.eliminate_zeros()
            for k in range(nSens):
                dfx[k][i,:] = dsmindx*( dfxi[k][i,:]+varLambda[k,:] )
                varLambda[k,:] = ( (dfxi[k][i,:]+varLambda[k,:])*dsmindXi ) @ dsmaxdxi
        
        # base layer
        i = nely
        for k in range(nSens):
            dfx[k][i-1,:] = dfxi[k][i-1,:] + varLambda[k,:]
    
    # ORIENTATION
    xi = np.rot90(xi,-nRot)
    varargout = ()
    for s in range(nSens):
        varargout = varargout + (np.rot90(dfx[s],-nRot) ,)

    return xi, varargout