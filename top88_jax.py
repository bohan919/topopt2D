import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.scipy.sparse.linalg import cg
from profiler_utils import profile_optimisation
from MMA import mmasub

# Enforce double precision for linear solver stability
jax.config.update("jax_enable_x64", True)

def get_element_stiffness_jax():
    E = 1.0; nu = 0.3
    k = np.array([1/2-nu/6, 1/8+nu/8, -1/4-nu/12, -1/8+3*nu/8, 
                  -1/4+nu/12, -1/8-nu/8, nu/6, 1/8-3*nu/8])
    KE = E / (1 - nu**2) * np.array([
        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]
    ])
    return jnp.array(KE)

def setup_mesh_and_filter(nelx, nely, rmin):
    ndof = 2 * (nelx + 1) * (nely + 1)
    
    edofMat = np.zeros((nelx * nely, 8), dtype=int)
    for elx in range(nelx):
        for ely in range(nely):
            el, n1, n2 = ely + elx * nely, (nely + 1) * elx + ely, (nely + 1) * (elx + 1) + ely
            edofMat[el, :] = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n2+2, 2*n2+3, 2*n1+2, 2*n1+3]
            
    iK = np.kron(edofMat, np.ones((8, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 8))).flatten()
    k_indices = jnp.column_stack((iK, jK))
    
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
                    
    H_bcoo = BCOO((jnp.array(sH[:k]), jnp.column_stack((iH[:k], jH[:k]))), shape=(nelx*nely, nelx*nely))
    Hs = jnp.array(H_bcoo.sum(axis=1).todense()).flatten()
    return ndof, k_indices, H_bcoo, Hs

# Static argument 'ft' ensures JAX recompiles a specific graph for each filter type
def compliance_objective(x, KE, k_indices, F, free_mask, ndof, penal, E0, Emin, H_bcoo, Hs, ft):
    # If using the density filter (ft=2), apply it inside the differentiable function
    xPhys = jnp.where(ft == 2, (H_bcoo @ x) / Hs, x)
        
    E = Emin + xPhys**penal * (E0 - Emin)
    sK = (KE.flatten() * E[:, None]).flatten()
    K = BCOO((sK, k_indices), shape=(ndof, ndof))
    
    # Conjugate Gradient linear operator enforcing boundary conditions
    def A_op(u):
        Ku = K @ (u * free_mask)
        return Ku * free_mask + u * (1.0 - free_mask)
    
    # Solve iterative FEA (JAX calculates adjoint gradients through this solve)
    u, _ = cg(A_op, F, rtol=1e-5, maxiter=3000)
    return jnp.dot(F, u)

# Compile the value and gradient pipeline
compute_loss_and_grad = jax.jit(jax.value_and_grad(compliance_objective), static_argnames=['ft'])

@jax.jit
def apply_sensitivity_filter(dc, x, H_bcoo, Hs):
    return (H_bcoo @ (x * dc)) / Hs / jnp.maximum(0.001, x)

@profile_optimisation("JAX Top88 (OC + MMA)", print_stats=True)
def optimise_jax(nelx=60, nely=20, volfrac=0.5, penal=3.0, rmin=1.5, ft=2, opt_algo='mma', max_loop=150):
    KE, E0, Emin = get_element_stiffness_jax(), 1.0, 1e-9
    ndof, k_indices, H_bcoo, Hs = setup_mesh_and_filter(nelx, nely, rmin)
    
    F = jnp.zeros(ndof).at[1].set(-1.0)
    fixeddofs = np.union1d(np.arange(0, 2*(nely+1), 2), [2*(nelx+1)*(nely+1)-1])
    free_mask = jnp.ones(ndof).at[fixeddofs].set(0.0)
    
    # Design variables (Flat 1D array for JAX)
    x = jnp.full(nelx * nely, volfrac)
    
    if opt_algo == 'mma':
        nele = nelx * nely
        m, n = 1, nele
        xmin, xmax = np.zeros((n, 1)), np.ones((n, 1))
        xold1 = np.asarray(x).reshape(-1, 1, order='F')
        xold2 = np.asarray(x).reshape(-1, 1, order='F')
        low, upp = np.ones((n, 1)), np.ones((n, 1))
        a0, a, c_MMA, d = 1.0, np.zeros((m, 1)), 10000 * np.ones((m, 1)), np.zeros((m, 1))
        mma_move = 0.2
        
    loop = 0
    change = 1.0
    
    while change > 0.01 and loop < max_loop:
        loop += 1
        
        # 1. JAX Forward Pass and Automatic Differentiation
        c_val, raw_dc = compute_loss_and_grad(x, KE, k_indices, F, free_mask, ndof, penal, E0, Emin, H_bcoo, Hs, ft=ft)
        
        # 2. Extract and format gradients based on the selected filter
        if ft == 1:
            dc = np.asarray(apply_sensitivity_filter(raw_dc, x, H_bcoo, Hs))
            dv = np.ones(nelx * nely)
            xPhys_current = np.asarray(x)
        elif ft == 2:
            dc = np.asarray(raw_dc) # JAX applied the density chain rule automatically
            dv = np.asarray(H_bcoo @ (jnp.ones(nelx * nely) / Hs))
            xPhys_current = np.asarray((H_bcoo @ x) / Hs)

        # Move constants to host NumPy for the optimisers
        c_numpy = float(c_val)
        x_numpy = np.asarray(x)
        
        # 3. Optimisation Branches
        if opt_algo == 'oc':
            l1, l2, oc_move = 0.0, 1e9, 0.2
            while (l2 - l1) > 1e-4:
                lmid = 0.5 * (l2 + l1)
                xnew = np.clip(x_numpy * np.sqrt(-dc / dv / lmid), x_numpy - oc_move, x_numpy + oc_move)
                xnew = np.clip(xnew, 0.001, 1.0)
                
                xPhys_new = xnew if ft == 1 else np.asarray(H_bcoo @ jnp.array(xnew)) / np.asarray(Hs)
                    
                if np.mean(xPhys_new) - volfrac > 0:
                    l1 = lmid
                else:
                    l2 = lmid
                    
            xPhys_current = xPhys_new

        elif opt_algo == 'mma':
            # Reshape inputs to Fortran column-major order for mmasub
            xval = x_numpy.reshape(-1, 1, order='F')
            df0dx = dc.reshape(-1, 1, order='F')
            fval = np.sum(xPhys_current) / (volfrac * nele) - 1.0
            dfdx = (dv / (volfrac * nele)).reshape(1, -1, order='F')
            
            xmma, _, _, _, _, _, _, _, _, low, upp = mmasub(
                m, n, loop, xval, xmin, xmax, xold1, xold2, 
                c_numpy, df0dx, fval, dfdx, low, upp, a0, a, c_MMA, d, mma_move
            )
            
            xnew = xmma.flatten(order='F')
            xPhys_current = xnew if ft == 1 else np.asarray(H_bcoo @ jnp.array(xnew)) / np.asarray(Hs)
                
            xold2 = xold1.copy()
            xold1 = xval.copy()

        change = np.max(np.abs(xnew - x_numpy))
        x = jnp.array(xnew)
        
        print(f"It.: {loop:03d} , obj.: {c_numpy:.4f}, vol.: {np.mean(xPhys_current):.3f}, ch.: {change:.3f}")

    return xPhys_current

if __name__ == "__main__":
    # Ensure MMA.py is in the directory to run
    optimise_jax(nelx=20, nely=10, volfrac=0.4, ft=2, opt_algo='mma', max_loop=1500)