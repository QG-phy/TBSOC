# -*- coding: utf-8 -*-
import numpy as np


def interpolationkpath(khp,npoints):
    """Interpolate k-path for given high- symmetry points."""
    N=len(khp)
    dim=len(khp[0])
    kpath=np.zeros([(N-1)*npoints,dim])
    for i in range(N-1):
        for j in range(dim):
            temp = np.linspace(khp[i][j],khp[i+1][j],npoints)
            kpath[i*npoints : (i+1)*npoints,j]=temp
    return kpath


def hr2hk(hr, Rlatt, kpath, num_wann):
    """
    Transform Hr to Hk using vectorized operations.
    
    Args:
        hr: Hopping matrices (nrpts, num_wann, num_wann) (or 2*num_wann for spinor)
        Rlatt: Lattice vectors (nrpts, 3)
        kpath: k-points (nk, 3)
        num_wann: Number of wannier functions (dimension of matrix)
        
    Returns:
        Hk: Hamiltonian at k-points (nk, num_wann, num_wann)
    """
    kpath = np.reshape(kpath, [-1, 3])
    
    # Calculate phase factors: exp(-i * 2pi * k . R)
    # kpath shape: (nk, 3), Rlatt shape: (nr, 3) -> dot product shape: (nk, nr)
    phase = np.exp(-1j * 2 * np.pi * np.dot(kpath, Rlatt.T))
    
    # Sum over R: Hk(k) = sum_R phase(k, R) * hr(R)
    # phase: (nk, nr), hr: (nr, n, n) -> result: (nk, n, n)
    Hk = np.einsum('kr,rij->kij', phase, hr)
    
    return Hk
