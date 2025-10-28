import numpy as np 

def corrmat(rhoiip1):
    # This is for Ng x Ng matrices, that is b1, c2, b3, cgg
    # rhoiip1 = 1-np.array(epsiip1)**2/2.
    n = len(rhoiip1)
    mat = 0. * np.eye(n+1)
    for i in range(0, n):
        mat[i, i+1] = rhoiip1[i]  # Make 1st diagonal
        for j in range(i+2, n+1):
            mat[i, j] = rhoiip1[i] * rhoiip1[j-1] # Multiply correlations for the rest
    mat = np.eye(n+1) + mat + mat.T
    return mat

def corrmatgs(rhoiip1Ng, rhojjp1Ns):
    # rhoiip1Ng = 1-np.array(epsiip1Ng)**2/2.
    # rhojjp1Ns = 1-np.array(epsjjp1Ns)**2/2.
    Ng = len(rhoiip1Ng)+1
    Ns = len(rhojjp1Ns)+1
    bigmat = np.zeros([Ng, Ns, Ng, Ns])
    corrNg = corrmat(rhoiip1Ng)
    corrNs = corrmat(rhojjp1Ns)
    for j in range(Ns):
        bigmat[:, j, :, j] = corrNg  # Use previous function for same lens or same source
    for i in range(Ng):
        bigmat[i, :, i, :] = corrNs
    for i in range(Ng):
        for j in range(Ns):
            for kg in range(i+1, Ng):
                for ks in range(j+1, Ns):
                    # Now we fill the rest of the matrix
                    bigmat[i, j, kg, ks] = bigmat[i, j, kg, j] * bigmat[i, j, i, ks]
                    bigmat[kg, j, i, ks] = 1 * bigmat[i, j, kg, ks]
                    bigmat[i, ks, kg, j] = 1 * bigmat[i, j, kg, ks]
                    bigmat[kg, ks, i, j] = 1 * bigmat[i, j, kg, ks]
    return bigmat.reshape(Ng*Ns, Ng*Ns)

def corrmatss(rhojjp1Ns):
    # rhoiip1Ng = 1-np.array(epsiip1Ng)**2/2.
    # rhojjp1Ns = 1-np.array(epsjjp1Ns)**2/2.
    Ns = len(rhojjp1Ns)+1
    bigmat = np.zeros([Ns, Ns, Ns, Ns])
    corrNs = corrmat(rhojjp1Ns)
    for j in range(Ns):
        bigmat[:, j, :, j] = corrNs  # Use previous function for same lens or same source
        bigmat[j, :, j, :] = corrNs
    for i in range(Ns):
        for j in range(Ns):
            for kg in range(i+1, Ns):
                for ks in range(j+1, Ns):
                    # Now we fill the rest of the matrix.
                    bigmat[i, j, kg, ks] = bigmat[i, j, kg, j] * bigmat[i, j, i, ks]
                    bigmat[kg, j, i, ks] = 1 * bigmat[i, j, kg, ks]
                    bigmat[i, ks, kg, j] = 1 * bigmat[i, j, kg, ks]
                    bigmat[kg, ks, i, j] = 1 * bigmat[i, j, kg, ks]
    def idx(i, j, n):
        return int((2 * (n+1) - 3 - i) * i / 2 + j)
    matred = np.zeros((int(Ns*(Ns+1)/2), int(Ns*(Ns+1)/2)))
    for i in range(Ns):
        for j in range(i, Ns):
            for ki in range(Ns):
                for kj in range(ki, Ns):
                    matred[idx(i,j,Ns), idx(ki,kj,Ns)] = 1 * bigmat[i, j, ki, kj]
    return matred

def corrmat_ct_gs_gg(rho_ct, rhoiip1Ng, rhojjp1Ns):
    Ng = len(rhoiip1Ng)+1
    Ns = len(rhojjp1Ns)+1
    bigmat = np.zeros([Ng, Ns, Ng, Ns, Ng])
    corrNg = corrmat(rhoiip1Ng)
    corrNs = corrmat(rhojjp1Ns)
    for j in range(Ns):
        for l in range(Ng): 
            bigmat[:, j, :, j, l] = corrNg * rho_ct
    for i in range(Ng):
        for l in range(Ng): 
            bigmat[i, :, i, :, l] = corrNs
    for i in range(Ng):
        for j in range(Ns):
            for kg in range(i+1, Ng):
                for ks in range(j+1, Ns):
                    # Now we fill the rest of the matrix
                    bigmat[i, j, kg, ks] = bigmat[i, j, kg, j] * bigmat[i, j, i, ks]
                    bigmat[kg, j, i, ks] = 1 * bigmat[i, j, kg, ks]
                    bigmat[i, ks, kg, j] = 1 * bigmat[i, j, kg, ks]
                    bigmat[kg, ks, i, j] = 1 * bigmat[i, j, kg, ks]
    return bigmat.reshape(Ng*Ns, Ng*Ns)
