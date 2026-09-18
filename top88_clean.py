import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import cg, spilu, LinearOperator
from profiler_utils import profile_optimisation

# Import MMA functions
from MMA import mmasub

def get_element_stiffness():
    E = 1.0; nu = 0.3
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, 
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    return E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])

def precompute_filter(nelx, nely, rmin):
    iH = np.ones(nelx * nely * (2 * (int(np.ceil(rmin)) - 1) + 1)**2)
    jH, sH = np.ones(iH.shape), np.zeros(iH.shape)
    k = 0
    for i1 in range(nelx):
        for j1 in range(nely):
            e1 = i1 * nely + j1
            for i in range(max(i1 - (int(np.ceil(rmin)) - 1), 0), min(i1 + int(np.ceil(rmin)), nelx)):
                for j in range(max(j1 - (int(np.ceil(rmin)) - 1), 0), min(j1 + int(np.ceil(rmin)), nely)):
                    e2 = i * nely + j
                    iH[k], jH[k] = e1, e2
                    sH[k] = max(0.0, rmin - np.sqrt((i1 - i)**2 + (j1 - j)**2))
                    k += 1
    H = coo_matrix((sH[:k], (iH[:k], jH[:k])), shape=(nelx*nely, nelx*nely)).tocsr()
    return H, H.sum(axis=1).A1

@profile_optimisation("Improved Python Top88", print_stats=True)
def optimise_python(nelx=20, nely=10, volfrac=0.4, penal=3.0, rmin=1.5, ft=2, opt_algo='mma', max_loop=1500):
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE, E0, Emin = get_element_stiffness(), 1.0, 1e-9
    H, Hs = precompute_filter(nelx, nely, rmin)

    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el, n1, n2 = ely + elx * nely, (nely + 1) * elx + ely, (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3]
    iK, jK = np.kron(edofMat, np.ones((8, 1))).flatten(), np.kron(edofMat, np.ones((1, 8))).flatten()
    K = coo_matrix((np.ones(len(iK)), (iK, jK)), shape=(ndof, ndof)).tocsc()

    F, U = np.zeros((ndof, 1)), np.zeros((ndof, 1))
    F[1, 0] = -1.0
    fixeddofs = np.union1d(np.arange(0, 2*(nely+1), 2), [2*(nelx+1)*(nely+1)-1])
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)

    # Initialise Arrays
    x = np.full((nely, nelx), volfrac)
    xPhys = x.copy()
    
    # Conditional Initialisation for MMA Variables
    if opt_algo == 'mma':
        nele = nelx * nely
        m, n = 1, nele
        xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
        xold1 = x.flatten('F')[:, np.newaxis]
        xold2 = x.flatten('F')[:, np.newaxis]
        low, upp = np.ones((n, 1)), np.ones((n, 1))
        a0, a, c_MMA, d = 1.0, np.zeros((m, 1)), 10000 * np.ones((m, 1)), np.zeros((m, 1))
        mma_move = 0.2

    loop = 0
    change = 1.0

    while change > 0.01 and loop < max_loop:
        loop += 1
        
        # 1. Update K.data directly
        sK = ((KE.flatten()[np.newaxis]).T * (Emin + xPhys.flatten('F')**penal * (E0 - Emin))).flatten(order='F')
        K.data = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc().data

        # 2. Linear Solve
        K_free = K[freedofs, :][:, freedofs]
        ilu = spilu(K_free, drop_tol=1e-4)
        M_op = LinearOperator(shape=K_free.shape, matvec=ilu.solve, dtype=K_free.dtype)
        U_free, _ = cg(K_free, F[freedofs, 0], M=M_op, rtol=1e-5)
        U[freedofs, 0] = U_free

        # 3. Objective and Sensitivity
        ce = np.reshape(np.sum((U[edofMat, 0] @ KE) * U[edofMat, 0], axis=1), (nely, nelx), order='F')
        c = np.sum((Emin + xPhys**penal * (E0 - Emin)) * ce)
        dc = -penal * xPhys**(penal-1) * (E0 - Emin) * ce
        dv = np.ones((nely, nelx))

        # 4. Filters
        if ft == 1:
            dc = H @ (x.flatten('F') * dc.flatten('F')) / Hs / np.maximum(0.001, x.flatten('F'))
            dc = np.reshape(dc, (nely, nelx), order='F')
        elif ft == 2:
            dc = np.reshape(H @ (dc.flatten('F') / Hs), (nely, nelx), order='F')
            dv = np.reshape(H @ (dv.flatten('F') / Hs), (nely, nelx), order='F')

        # 5. Algorithm Branching
        if opt_algo == 'oc':
            l1, l2, oc_move = 0.0, 1e9, 0.2
            while (l2 - l1) > 1e-4:
                lmid = 0.5 * (l2 + l1)
                xnew = np.clip(x * np.sqrt(-dc / dv / lmid), x - oc_move, x + oc_move)
                xnew = np.clip(xnew, 0.001, 1.0)
                
                if ft == 1:
                    xPhys_new = xnew
                elif ft == 2:
                    xPhys_new = np.reshape((H @ xnew.flatten('F')) / Hs, (nely, nelx), order='F')
                    
                if np.mean(xPhys_new) - volfrac > 0:
                    l1 = lmid
                else:
                    l2 = lmid
                    
            xPhys = xPhys_new
            
        elif opt_algo == 'mma':
            # Format parameters for mmasub
            xval = x.flatten('F')[:, np.newaxis]
            df0dx = dc.flatten('F')[:, np.newaxis]
            fval = np.sum(xPhys) / (volfrac * nele) - 1.0
            dfdx = (dv.flatten('F') / (volfrac * nele))[np.newaxis, :] 

            # MMA Update
            xmma, _, _, _, _, _, _, _, _, low, upp = mmasub(
                m, n, loop, xval, xmin, xmax, xold1, xold2, 
                c, df0dx, fval, dfdx, low, upp, a0, a, c_MMA, d, mma_move
            )

            # Post-process MMA step
            xnew = np.reshape(xmma, (nely, nelx), order='F')
            if ft == 1:
                xPhys = xnew
            elif ft == 2:
                xPhys = np.reshape((H @ xnew.flatten('F')) / Hs, (nely, nelx), order='F')

            xold2 = xold1.copy()
            xold1 = x.flatten('F')[:, np.newaxis]

        change = np.max(np.abs(xnew - x))
        x = xnew

        print(f"It.: {loop:03d} , obj.: {c:.4f}, vol.: {np.mean(xPhys):.3f}, ch.: {change:.3f}")
        
    return x

if __name__ == "__main__":
    # Test MMA with Density Filter (ft=2)
    optimise_python(ft=2, opt_algo='mma', max_loop=50)