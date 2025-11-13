import numpy as np
from tbsoc.lib.cal_tools import interpolationkpath, hr2hk

def test_interpolationkpath():
    """
    Test the interpolationkpath function.
    """
    # Define a simple path from Gamma (0,0,0) to X (0.5, 0, 0)
    high_symm_points = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0]
    ])
    npoints = 3
    
    kpath = interpolationkpath(high_symm_points, npoints)
    
    # Expected shape is ( (N-1)*npoints, dim ) = ( (2-1)*3, 3 ) = (3, 3)
    assert kpath.shape == (3, 3)
    
    # The first point should be the start point
    assert np.allclose(kpath[0], [0.0, 0.0, 0.0])
    
    # The last point should be the end point
    assert np.allclose(kpath[-1], [0.5, 0.0, 0.0])
    
    # The intermediate point should be halfway
    assert np.allclose(kpath[1], [0.25, 0.0, 0.0])

def test_hr2hk_simple_model():
    """
    Test the hr2hk function with a simple 1D tight-binding model.
    """
    num_wann = 1
    # On-site energy of 2.0, nearest-neighbor hopping of 1.0
    hr = np.array([
        [[2.0]],  # H(R=0)
        [[1.0]]   # H(R=1)
    ])
    Rlatt = np.array([
        [0, 0, 0],
        [1, 0, 0]
    ])
    # K-path from Gamma to X
    kpath = np.array([
        [0.0, 0.0, 0.0],  # Gamma point
        [0.5, 0.0, 0.0]   # X point
    ])
    
    Hk = hr2hk(hr, Rlatt, kpath, num_wann)
    Hk = np.array(Hk) # Convert list of arrays to a single array
    
    # Check shape
    assert Hk.shape == (2, 1, 1)
    
    # At Gamma (k=0), E = 2.0 + 1.0 * exp(0) = 3.0
    assert np.isclose(Hk[0, 0, 0], 3.0)
    
    # At X (k=0.5), E = 2.0 + 1.0 * exp(-i*2*pi*0.5) = 2.0 - 1.0 = 1.0
    assert np.isclose(Hk[1, 0, 0], 1.0)

def test_hr2hk_2d_model():
    """
    Test hr2hk with a 2-orbital model.
    """
    num_wann = 2
    # Simple 2-orbital model, diagonal on-site, off-diagonal hopping
    hr = np.array([
        [[1.0, 0.0], [0.0, 2.0]],  # H(R=0)
        [[0.0, 0.5], [0.5, 0.0]]   # H(R=1)
    ])
    Rlatt = np.array([
        [0, 0, 0],
        [1, 0, 0]
    ])
    kpath = np.array([
        [0.0, 0.0, 0.0]  # Gamma point
    ])

    Hk = hr2hk(hr, Rlatt, kpath, num_wann)
    Hk = np.array(Hk)

    # At Gamma, Hk = H(R=0) + H(R=1)
    expected_Hk_gamma = np.array([[1.0, 0.5], [0.5, 2.0]])
    
    assert Hk.shape == (1, 2, 2)
    assert np.allclose(Hk[0], expected_Hk_gamma)
