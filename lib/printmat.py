# -*- coding: utf-8 -*-


def printsocmat(orb,Msoc):
    """Print the soc matrix."""
    if orb=='p':
        norbs=3
    elif orb=='d':
        norbs=5
    else:
        print('can not recognize the orbital !')
        exit()
    for i in range(2*norbs):
        for j in range(2*norbs):
            if Msoc[i,j].real==0 and Msoc[i,j].imag==0:
                print('%7.2f ' %0 , end='')
            elif Msoc[i,j].real==0:
                print ('%7.2fI' %(2*Msoc[i,j].imag),end='')
            elif Msoc[i,j].imag==0:
                print ('%7.2f ' %(2*Msoc[i,j].real),end='')
            else:
                print ('%7.2f + %7.2fI' %(2*Msoc[i,j].real,2*Msoc[i,j].imag),end='')
        print ('')
        print ('')