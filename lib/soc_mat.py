# -*- coding: utf-8 -*-
import numpy as np
from lm_expand import MapLmSp, MapLpSm, MapLzSz
from const import Mtrans
from printmat import printsocmat


def creat_basis_lm(orb):
    """Creat |lm,s> stated for orbital."""
    if orb=='p':
        basis = []
        for m in [-1,0,1]:
            for spin in [1,-1]:
                basis.append([1,m,spin])
    if orb=='d':
        basis = []
        for m in [-2,-1,0,1,2]:
            for spin in [1,-1]:
                basis.append([2,m,spin])
    if orb=='s':
        print('we do not consider the soc in s orbital')
    if orb=='f':
        print('for now, soc for f orbital is not added.')
        exit()
    return basis


def get_matrix_lmbasis(basis):
    """Creat Hsoc matrix  in |lm,s> basis."""
    ndim = len(basis)
    MatLpSm = np.zeros([ndim,ndim])
    MatLmSp = np.zeros([ndim,ndim])
    MatLzSz = np.zeros([ndim,ndim])
    LdotS = np.zeros([ndim,ndim])
    for i in range(len(basis)):
        raw = i
        cof,bas = MapLpSm(basis[i])
        if bas in basis:
            col = basis.index(bas)
            MatLpSm[raw,col] = cof
            
        cof,bas = MapLmSp(basis[i])
        if bas in basis:
            col = basis.index(bas)
            MatLmSp[raw,col] = cof
        
        cof,bas = MapLzSz(basis[i])
        if bas in basis:
            col = basis.index(bas)
            MatLzSz[raw,col] = cof
    LdotS = np.mat(0.5*(MatLpSm + MatLmSp + MatLzSz))
    return LdotS


def trans_lm_spatial(orb,Msoc):
    """Transform Hsoc matrix  in |lm,s> basis to px py pz or dxy,dyz ... basis."""
    trans = np.mat(np.kron(Mtrans[orb],np.eye(2))).T
    Msoc_spatial = np.dot(np.dot(trans.H,Msoc),trans)
    return Msoc_spatial


def soc_order(orb,Msoc):
    """Transfer the spin basis form up down up down ... to up up ... down down..."""
    print('transfer the spin basis form up down up down ... to up up ... down down...')
    if orb=='p':
        norbs=3
    elif orb=='d':
        norbs=5
    else:
        print('can not recognize the orbital !')
        exit()
    Mattmp = np.zeros([2*norbs,2*norbs],dtype=complex)
    Mattmp[0    :  norbs,    0:  norbs] = Msoc[0:2*norbs:2,0:2*norbs:2]
    Mattmp[norbs:2*norbs,norbs:2*norbs] = Msoc[1:2*norbs:2,1:2*norbs:2]
    Mattmp[0    :  norbs,norbs:2*norbs] = Msoc[0:2*norbs:2,1:2*norbs:2]
    Mattmp[norbs:2*norbs,    0:  norbs] = Msoc[1:2*norbs:2,0:2*norbs:2]
    return Mattmp


def get_mat_soc_orb(orb):
    """Get soc matrix."""
    if orb != 'p' and orb != 'd':
        print("orbtial is wrong, can be eitgher 'p' or 'd' ")
        exit()
    basis = creat_basis_lm(orb)
    LdotS = get_matrix_lmbasis(basis)
    LdoS_spatial_udud = trans_lm_spatial(orb,LdotS)
    print("generating Hsoc mat for " + orb + "orbital in atomic orbital basis")
    printsocmat(orb,LdoS_spatial_udud)
    LdoS_spatial_uudd = soc_order(orb,LdoS_spatial_udud)
    return LdoS_spatial_uudd


def get_Hsoc(lambdas,orbitals,orb_type,orb_num,Msoc):
    """Get Hsoc."""
    if len(lambdas) != len(np.unique(orb_type)):
        print("Number of parameters for soc strength is wrong!")
        exit()

    num_wan = 2*np.sum(orb_num)
    tot_orb = np.sum(orb_num)
    Hsoc = np.zeros([num_wan,num_wan],dtype=complex)
    for i in range(len(orbitals)):
        istn1 = np.sum(orb_num[0:i])
        istn2 = istn1 + tot_orb
        
        #isto1 = 2*np.sum(orb_num[0:i])
        #isto2 = isto1 + orb_num[i]
             
        Hsoc[istn1:istn1 + orb_num[i],istn1:istn1 + orb_num[i]] = \
            lambdas[orb_type[i]] * Msoc[orbitals[i]][0:orb_num[i],0:orb_num[i]]
    
        Hsoc[istn2:istn2 + orb_num[i],istn2:istn2 + orb_num[i]] = \
            lambdas[orb_type[i]] * Msoc[orbitals[i]][orb_num[i]:2*orb_num[i],orb_num[i]:2*orb_num[i]]
    
        Hsoc[istn1:istn1 + orb_num[i],istn2:istn2 + orb_num[i]] = \
            lambdas[orb_type[i]] * Msoc[orbitals[i]][0:orb_num[i],orb_num[i]:2*orb_num[i]]
    
        Hsoc[istn2:istn2 + orb_num[i],istn1:istn1 + orb_num[i]] = \
            lambdas[orb_type[i]] * Msoc[orbitals[i]][orb_num[i]:2*orb_num[i],0:orb_num[i]]
    return Hsoc


