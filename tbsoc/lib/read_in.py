# -*- coding: utf-8 -*-
import re
import numpy as np
from .const import orb_l_dict, orb_num_dict
from .cal_tools import interpolationkpath


def read_hr(Filename='wannier90_hr.dat'):
    """Read wannier90_hr.dat."""
    print('reading wannier90_hr.dat ...')
    f=open(Filename,'r')
    data=f.readlines()
    #read hopping matrix
    num_wann = int(data[1])
    nrpts = int(data[2])
    r_hop= np.zeros([num_wann,num_wann], dtype=complex)
    #hop=[]
    #skip n lines of degeneracy of each Wigner-Seitz grid point
    skiplines = int(np.ceil(nrpts / 15.0))
    istart = 3 + skiplines
    deg=[]
    for i in range(3,istart):
        deg.append(np.array([int(j) for j in data[i].split()]))
    deg=np.concatenate(deg,0)
    
    icount=0
    ii=-1
    Rlatt = []
    hopps = []
    for i in range(istart,len(data)):
        line=data[i].split()
        m = int(line[3]) - 1
        n = int(line[4]) - 1
        r_hop[m,n] = complex(round(float(line[5]),6),round(float(line[6]),6))
        icount+=1
        if(icount % (num_wann*num_wann) == 0):
            ii+=1
            R = np.array([float(x) for x in line[0:3]])
            #hop.append(np.asarray([R,r_hop]))
            #r_hop= np.zeros([num_wann,num_wann], dtype=complex)
            
            Rlatt.append(R)
            hopps.append(r_hop)
            #hop.append(np.asarray([R,r_hop]))
            r_hop= np.zeros([num_wann,num_wann], dtype=complex)
    Rlatt=np.asarray(Rlatt)
    hopps=np.asarray(hopps)
    deg = np.reshape(deg,[nrpts,1,1])
    hopps=hopps/deg
    #deg2 = np.ones(nrpts)

    hop_spinor = np.zeros([nrpts,2*num_wann,2*num_wann],dtype=complex)
    for i in range(nrpts):
        hop_spinor[i] = np.kron(np.eye(2), hopps[i])
        if (Rlatt[i]==0).all():
            indR0 = i
    print('successfully reading wannier90_hr.dat ...')
    return hop_spinor,Rlatt,indR0


def read_poscar_wan_in(poscarfile = 'POSCAR',waninfile='wannier90.win'):
    """Read poscar and wannier90.win."""
    print('reading POSCAR ...')
    f=open(poscarfile)
    pos=f.readlines()
    f.close()
    natom_spec=np.array([int(i) for i in pos[6].split()])
    spec = pos[5].split()
    atoms=[]
    for i in range(len(spec)):
        atoms+=([spec[i]]*int(natom_spec[i]))
    ntot=np.sum(natom_spec)
    A1=(np.asarray(pos[2].split())).astype(float)[0:3]
    A2=(np.asarray(pos[3].split())).astype(float)[0:3]
    A3=(np.asarray(pos[4].split())).astype(float)[0:3]
    Lattice=np.array([A1,A2,A3])
    # Parse coordinates
    start_line = 7
    if pos[7].strip().lower().startswith('s'): # Selective dynamics
        start_line = 8
    
    mode = pos[start_line].strip().lower()
    coord_start = start_line + 1
    
    coords = []
    for i in range(ntot):
        line_parts = pos[coord_start + i].split()
        coords.append([float(x) for x in line_parts[0:3]])
    coords = np.array(coords)
    
    # Convert to Cartesian if Direct
    if mode.startswith('d'):
        coords = np.dot(coords, Lattice)

    print('successfully reading POSCAR ...')

    print('reading wannier90.win ...')
    f=open(waninfile)
    wan=f.readlines()
    f.close()
    icproj=0
    projind=np.zeros([2],dtype=int)
    for i in range(len(wan)):    
        if re.search('Projections',wan[i]) or re.search('projections',wan[i]) :
            projind[icproj]=i
            icproj+=1
            #print(i)
        if icproj==2:
            break
    atom_proj={}
    proj_type=0
    
    # 1. Identify projected species
    for iatom in spec:
        for i in range(projind[0]+1,projind[1]):       
            if re.search(iatom,wan[i]):
                # Use regex to specifically find l=<number> to avoid matching axis coordinates
                l_matches = re.findall(r"l\s*=\s*(\d+)", wan[i])
                if l_matches:
                    atom_proj[iatom] = l_matches
                    proj_type += len(l_matches)
                else:
                    print(f"Warning: No 'l=' definitions found in projection line for {iatom}: {wan[i].strip()}")
                    # print('The style to set orb in wannier90.win can not be recognized') 
                    # print( 'only be read in the style with l=1,2,3 ...')
                break # Found projection for this species

    # 2. Filter atoms and coords to keep only projected ones
    filtered_atoms = []
    filtered_coords = []
    
    print(f"Filtering POSCAR atoms based on projections: Keeping {list(atom_proj.keys())}")
    
    for k, atom in enumerate(atoms):
        if atom in atom_proj:
            filtered_atoms.append(atom)
            filtered_coords.append(coords[k])
            
    atoms = filtered_atoms
    coords = np.array(filtered_coords)
    
    # 3. Update spec to only include projected species (preserving order)
    spec = [s for s in spec if s in atom_proj]
    


    print('successfully reading wannier90.win ...')
    print('The projections is :')

    typeindex={}
    ic=-1
    orb_labels=[]
    for i in spec:
        typeindex[i]={}
        for j in range(len(atom_proj[i])):
            ic+=1
            l_val = atom_proj[i][j]
            typeindex[i][l_val]=ic
            # Construct label e.g. Ga:s, As:p
            orb_name = orb_l_dict.get(l_val, f"l={l_val}")
            # If mapped to list/tuple (e.g. s->'s', p->'p'), assume it's string.
            # wait, line 140 says orbitals+=orb_l_dict[l]. So orb_l_dict[l] is a list of orbitals?
            # '0' -> ['s'], '1' -> ['pz','px','py'] ?? No, usually SOC fitting parameters are per l-shell.
            # Let's check orb_l_dict usage.
            # Line 140: orbitals+=orb_l_dict[l]
            # Line 142: print(orb_l_dict[l])
            # If orb_l_dict['1'] is ['p'] or 'p'?
            # Let's assume we want just the shell name. 
            # Looking at previous logs: "generating Hsoc mat for porbital"
            # It seems we want "Ga:p" or "Ga:l=1".
            orb_labels.append(f"{i}:{orb_name if isinstance(orb_name, str) else orb_name[0] if isinstance(orb_name, list) and len(orb_name)>0 else l_val}")

    orbitals=[]
    orb_type=[]
    for a in atoms:
        print(a,'\t', end='')
        for l in atom_proj[a]:
            orbitals+=orb_l_dict[l]
            orb_type.append(typeindex[a][l])
            print(orb_l_dict[l],'\t',end='')
        print('')
    
    orb_num=[]
    for orb in orbitals:
        orb_num.append(orb_num_dict[orb])
    orb_num=np.asarray(orb_num)

    return Lattice, atoms, coords, atom_proj, orbitals, orb_num, orb_type, orb_labels


def read_KPOINTS(Latt,kpofile='KPOINTS'):
    """Read KPOINTS."""
    print('reading KPOINTS ...')
    f=open(kpofile)
    kpo=f.readlines()
    f.close()
    
    KPOINT_IN_LINE = int(kpo[1].split()[0])
    kpo
    ksymbol=[]
    khsym=[]
    for i in range(4,len(kpo)):
        #print (kpo[i])
        if re.findall(r"\-?\d+\.?\d*",kpo[i]) != []:
            kpoints = np.array([float(ikp) for ikp in re.findall(r"\-?\d+\.?\d*",kpo[i])[:3]])
            parts = kpo[i].strip().split()
            if len(parts) >= 4:
                symbol = parts[3].lstrip('#')
            else:
                symbol = 'UNK'
            khsym.append(kpoints)
            ksymbol.append(symbol)
            #print (re.findall("\d+\.?\d*",kpo[i]))
            #print(re.findall("[A-Z]+[a-z]*",kpo[i]))
            
    khsym = np.asarray(khsym)        
    khsym = np.reshape(khsym,[-1,2,3])
    ksymbol = np.reshape(ksymbol,[-1,2])
    nkslice=ksymbol.shape[0]
    if nkslice != khsym.shape[0]:
        print ('Err the number of slice of k-points!')
    for i in range(nkslice):
        print(ksymbol[i,0],": ",end='')
        print("[%8.3f,%8.3f,%8.3f]" %(khsym[i,0,0],khsym[i,0,1],khsym[i,0,2]),end='')
        print(" ===> ",ksymbol[i,1],": ",end="")    
        print("[%8.3f,%8.3f,%8.3f]" %(khsym[i,1,0],khsym[i,1,1],khsym[i,1,2]))
    kpath=[]
    xpath=[]
    xsymm=[0]
    B=2*np.pi*np.linalg.inv(Latt).T
    temp=0
    for i in range(nkslice):
        kslice = interpolationkpath(khsym[i],KPOINT_IN_LINE)
        xslice = [temp]
        for i in range(1, len(kslice)):
            ka = np.dot(kslice[i], B)
            kb = np.dot(kslice[i-1], B)
            delta = np.sqrt(sum([(kb[j] - ka[j]) ** 2 for j in range(0, 3)]))
            temp = xslice[i - 1] + delta 
            #print (delta)
            xslice.append(temp)
        #print (temp)
        xsymm.append(temp)
        xpath.append(xslice)    
        kpath.append(kslice)
    kpath = np.reshape(kpath,[-1,3])
    xpath = np.reshape(xpath,[-1])
    xsymm = np.asarray(xsymm)
    ksymbol = ksymbol.tolist()
    plot_sbol=[ksymbol[0][0]]
    for i in range(1,nkslice):
        if ksymbol[i][0] == ksymbol[i-1][1]:
            plot_sbol += [ksymbol[i][0]]
        else:
            plot_sbol += [ksymbol[i-1][1] + '|' + ksymbol[i][0]]
    plot_sbol += [ksymbol[nkslice-1][1]]
    print('successfully reading KPOINTS ...')
    #np.savetxt('xpath.txt',xpath)
    return kpath,xpath,xsymm,plot_sbol


def read_EIGENVAL(FILENAME='EIGENVAL'):
    """Read EIGENVAL."""
    print('reading EIGENVAL ...')
    k_bands = []
    kb_temp = []
    kb_count= 0
    k_list2 = []
    f     = open(FILENAME, 'r')
    data  = f.readlines()
    Nhse  = 0
    EFERMI= 0
    # Read the number of bands
    NBND = int(re.findall('[0-9]+', data[5])[2])
    
    for i in range(7+(NBND+2)*Nhse, len(data)):
        temp = re.findall(r'[0-9\-\.\+E]+', data[i])
        if not temp:
            continue
        if len(temp) == 4:
            kt2 = (np.array([float(i) for i in temp[0:3]])).tolist()
            k_list2.append(kt2)
        else:
            kb_temp.append(float(temp[1]) - EFERMI)
            kb_count += 1
            if kb_count == NBND:
                k_bands.append(sorted(kb_temp))
                kb_temp = []
                kb_count = 0
    k_bands = np.array(k_bands)
    k_list2 = np.asarray(k_list2)
    print('successfully reading EIGENVAL ...')
    return k_bands, k_list2


