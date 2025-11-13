import pytest
import numpy as np
from tbsoc.lib.soc_mat import creat_basis_lm, get_matrix_lmbasis, trans_lm_spatial

def test_creat_basis_lm():
    """
    Test the creat_basis_lm function.
    """
    # Test for 's' orbital
    s_basis = creat_basis_lm('s')
    assert s_basis == [[0, 0, 1], [0, 0, -1]]

    # Test for 'p' orbital
    p_basis = creat_basis_lm('p')
    expected_p_basis = [
        [1, -1, 1], [1, -1, -1],
        [1, 0, 1], [1, 0, -1],
        [1, 1, 1], [1, 1, -1]
    ]
    assert p_basis == expected_p_basis

    # Test for 'd' orbital
    d_basis = creat_basis_lm('d')
    expected_d_basis = [
        [2, -2, 1], [2, -2, -1],
        [2, -1, 1], [2, -1, -1],
        [2, 0, 1], [2, 0, -1],
        [2, 1, 1], [2, 1, -1],
        [2, 2, 1], [2, 2, -1]
    ]
    assert d_basis == expected_d_basis

    # Test for invalid orbital
    with pytest.raises(AssertionError, match="The orb parameter must be one of the s, p ,d values in the format of str"):
        creat_basis_lm('f')

    with pytest.raises(AssertionError):
        creat_basis_lm('g')


def test_get_matrix_lmbasis_p_orbital():
    """
    Test the get_matrix_lmbasis function for p orbitals.
    """
    p_basis = creat_basis_lm('p')
    ldot_s_matrix = get_matrix_lmbasis(p_basis)

    # Check shape
    assert ldot_s_matrix.shape == (6, 6)

    # Check if it's Hermitian
    assert np.allclose(ldot_s_matrix, ldot_s_matrix.conj().T)

    # Check some specific, known values of the L.S matrix for p-orbitals
    # L.S = 0.5 * (L+S- + L-S+ + LzSz)
    # Check <l=1, m=1, s=1/2 | L.S | l=1, m=1, s=1/2> = 0.5 * m * s = 0.5 * 1 * 0.5 = 0.25
    # In our basis: |1, 1, 1> is the 5th element (index 4)
    # Spin is +1/-1, so LzSz is m*s/2. Total is 0.5 * (m*s)
    assert np.isclose(ldot_s_matrix[4, 4], 0.5)

    # Check <l=1, m=0, s=1/2 | L.S | l=1, m=1, s=-1/2> = 0.5 * <0,1/2|L-S+|1,-1/2>
    # = 0.5 * sqrt(l(l+1)-m(m-1)) * sqrt(s(s+1)-ms(ms-1)) where m=1, ms=-1/2
    # = 0.5 * sqrt(2) * 1 = 0.707
    # In our basis: |1, 0, 1> is index 2; |1, 1, -1> is index 5
    assert np.isclose(ldot_s_matrix[2, 5], 0.5 * np.sqrt(2))


def test_trans_lm_spatial_p_orbital():
    """
    Test the trans_lm_spatial function for p orbitals.
    """
    p_basis = creat_basis_lm('p')
    ldot_s_matrix = get_matrix_lmbasis(p_basis)
    msoc_spatial = trans_lm_spatial('p', ldot_s_matrix)

    # Check shape
    assert msoc_spatial.shape == (6, 6)

    # Check if it's Hermitian
    assert np.allclose(msoc_spatial, msoc_spatial.conj().T)
