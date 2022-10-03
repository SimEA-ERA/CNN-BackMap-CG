from __future__ import print_function

atTypeCIS = 0
atTypeTRN = 1
atTypeVIN = 2

atNameCIS = 'CIS'
atNameTRN = 'TRN'
atNameVIN = 'VIN'

allAtomTypes = [0,1,2]

def atName(atType):
    if atType == atTypeCIS:
        return atNameCIS
    elif atType == atTypeTRN:
        return atNameTRN
    elif atType == atTypeVIN:
        return atNameVIN
    else:
        raise Exception('Bad atom name')

def load_chemistry(filename, polymer_length):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()

    molcount = 0
    chemistry = []
    chemistry_names = []
    for line in lines:
        line = line.strip()
        if len(line) > 0:
            mers = line.split(',')
            if polymer_length != None and len(mers) != polymer_length:
                raise Exception("Expected %d mer" % polymer_length)
            mol = []
            mol_names = []
            for mer in mers:
                parts = mer.split('-')
                nm = parts[0]
                mol_names.append(nm)
                if nm == 'C':
                    mol.append(atTypeCIS)
                elif nm == 'T':
                    mol.append(atTypeTRN)
                elif nm == 'V' or nm == 'VE5':
                    mol.append(atTypeVIN)
                else:
                    raise Exception('Invalid monomer name: ' + nm)
            chemistry.append(mol)
            chemistry_names.append(mol_names)            
            molcount += 1
    print("Read", molcount, "molecules")
    return chemistry, chemistry_names

def normalized_bond_type(atTypeA, atTypeB):
    if atTypeA <= atTypeB:
        return (atTypeA, atTypeB)
    else:
        return (atTypeB, atTypeA)

def normalized_bond(atoms, atAindex, atBindex):
    #print(atoms)
    atA = atoms[atAindex]
    atB = atoms[atBindex]
    if atA.atType <= atB.atType:
        return (atAindex, atBindex)        
    else:
        return (atBindex, atAindex)

def normalized_angle_type(atTypeA, atTypeB, atTypeC):
    if atTypeA <= atTypeC:
        return (atTypeA, atTypeB, atTypeC)
    else:
        return (atTypeC, atTypeB, atTypeA)

def normalized_angle(atoms, atAindex, atBindex, atCindex):
	atA = atoms[atAindex]
	atC = atoms[atCindex]
	if atA.atType <= atC.atType:
		return (atAindex, atBindex, atCindex)
	else:
		return (atCindex, atBindex, atAindex)

def normalized_dihedral_type(atTypeA, atTypeB, atTypeC, atTypeD):
    if atTypeA > atTypeD:
        return (atTypeD, atTypeC, atTypeB, atTypeA)
    elif atTypeD > atTypeA:
        return (atTypeA, atTypeB, atTypeC, atTypeD)
    else:
        if atTypeC >= atTypeB:
            return (atTypeA, atTypeB, atTypeC, atTypeD)
        else:
            return (atTypeD, atTypeC, atTypeB, atTypeA)

def normalized_dihedral(atoms, atAindex, atBindex, atCindex, atDindex):
	atA = atoms[atAindex]
	atD = atoms[atDindex]
	if atA.atType > atD.atType:
		return (atDindex, atCindex, atBindex, atAindex)
	elif atD.atType > atA.atType:
		return (atAindex, atBindex, atCindex, atDindex)
	else:
		atB = atoms[atBindex]
		atC = atoms[atCindex]
		if atC.atType >= atB.atType:
			return (atAindex, atBindex, atCindex, atDindex)
		else:
			return (atDindex, atCindex, atBindex, atAindex)

def generate_possible_bonds(chemistry):
    bonds = set()
    for mol in chemistry:
        nmon = len(mol)
        for imon in range(nmon-1):
            bonds.add(normalized_bond_type(mol[imon], mol[imon+1]))
    bond_dict = {}
    id = 0
    for b in sorted(list(bonds)):
        bond_dict[b] = id
        id += 1
    return bond_dict

def generate_bond_names(bond_types, start_from=0):
    result = {}
    for btype in bond_types.keys():
        id = bond_types[btype] + start_from
        name = atName(btype[0]) + '-' + atName(btype[1])
        result[id] = name
    return result


def generate_possible_angles(chemistry):
    angles = set()
    for mol in chemistry:
        nmon = len(mol)
        for imon in range(nmon - 2):
            angles.add(normalized_angle_type(mol[imon], mol[imon + 1], mol[imon + 2]))
    angle_dict = {}
    id = 0
    for a in sorted(list(angles)):
        angle_dict[a] = id
        id += 1
    return angle_dict

def generate_angle_names(angle_types, start_from=0):
    result = {}
    for atype in angle_types.keys():
        id = angle_types[atype] + start_from
        name = atName(atype[0]) + '-' + atName(atype[1]) + '-' + atName(atype[2])
        result[id] = name
    return result


def generate_possible_dihedrals(chemistry):
    dihedrals = set()
    imol = 0
    for mol in chemistry:
        nmon = len(mol)
        for imon in range(nmon - 3):
            dtype = normalized_dihedral_type(mol[imon], mol[imon + 1], mol[imon + 2], mol[imon + 3])
            #if dtype == (atTypeCIS, atTypeTRN, atTypeCIS, atTypeCIS):
            #    print "Dtype", dtype, "at mol", imol
            dihedrals.add( dtype )
        imol += 1
    dihedral_dict = {}
    id = 0
    for d in sorted(list(dihedrals)):
        dihedral_dict[d] = id
        id += 1
    return dihedral_dict

def generate_dihedral_names(dihedral_types, start_from=0):
    result = {}
    for dtype in dihedral_types.keys():
        id = dihedral_types[dtype] + start_from
        name = atName(dtype[0]) + '-' + atName(dtype[1]) + '-' + atName(dtype[2]) + '-' + atName(dtype[3])
        result[id] = name
    return result

def generate_mol_names(chemistry):
    result = {}
    for i in range(len(chemistry)):
        result[i] = 'PB' + str(i+1)
    return result

def generate_atom_names():
    result = {}
    result[atTypeCIS] = atNameCIS
    result[atTypeTRN] = atNameTRN
    result[atTypeVIN] = atNameVIN
    return result

def print_dict(dct):
    for k in sorted(dct.keys()):
        print(k, ':', dct[k])

def save_dict(dct, file):
    for k in sorted(dct.keys()):
        file.write('%s : %s\n' %(k, dct[k]))

def generateAllPossibleBonds():
    bonds = set()
    for a in allAtomTypes:
        for b in allAtomTypes:
                bonds.add(normalized_bond_type(a,b))
    bond_dict = {}
    id = 0
    for d in sorted(list(bonds)):
        bond_dict[d] = id
        id += 1
    return bond_dict


def generateAllPossibleAngles():
    angles = set()
    for a in allAtomTypes:
        for b in allAtomTypes:
            for c in allAtomTypes:
                    angles.add(normalized_angle_type(a,b,c))
    angle_dict = {}
    id = 0
    for d in sorted(list(angles)):
        angle_dict[d] = id
        id += 1
    return angle_dict

def generateAllPossibleDihedrals():
    dihedrals = set()
    for a in allAtomTypes:
        for b in allAtomTypes:
            for c in allAtomTypes:
                for d in allAtomTypes:
                    dihedrals.add(normalized_dihedral_type(a,b,c,d))
    dihedral_dict = {}
    id = 0
    for d in sorted(list(dihedrals)):
        dihedral_dict[d] = id
        id += 1
    return dihedral_dict



if __name__ == '__main__':
    chemistry = load_chemistry('chem_tcv_252550.txt', None)
    #print chemistry

    bond_types = generate_possible_bonds(chemistry)
    angle_types = generate_possible_angles(chemistry)
    dihedral_types = generate_possible_dihedrals(chemistry)

    #print bond_types
    #print angle_types
    #print dihedral_types

    print(len(bond_types), "bond types")
    print(len(angle_types), "angle types")
    print(len(dihedral_types), "dihedral types")
    print()

    print_dict(generate_bond_names(bond_types))
    print()
    #all_bond_types = generateAllPossibleBonds()
    #print_dict(generate_bond_names(all_bond_types))
    #print()

    print_dict(generate_angle_names(angle_types))
    print()
    #all_angle_types = generateAllPossibleAngles()
    #print_dict(generate_angle_names(all_angle_types))
    #print()

    print_dict(generate_dihedral_names(dihedral_types))
    print()
    #all_dihedral_types = generateAllPossibleDihedrals()
    #print_dict(generate_dihedral_names(all_dihedral_types))
    #print()

    #print generate_mol_names(chemistry)
    #print generate_atom_names()






