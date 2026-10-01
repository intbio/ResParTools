"""Файлы силового поля GROMACS: hdb, atp, r2b, удаление лишних водородов."""
import os
import sys
import time
import re
import glob
import json
from pathlib import Path
from collections import Counter
from collections import defaultdict, deque
import numpy as np

import itertools
import threading
import subprocess

from decimal import Decimal as D

from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D
from typing import Callable

import datetime
import functools
import hashlib
import inspect
import platform
import traceback
import warnings
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from .log import logged


# Дубликат: ниже в модуле есть второе определение check_duplicate_atom_names, оно и действует
# (в Python работает последнее определение). Этот вариант закомментирован, чтобы не путаться.
# def check_duplicate_atom_names(mol):
#     """Проверяет уникальность имен атомов"""
#     name_to_indices = {}
#     for atom in mol.GetAtoms():
#         if atom.HasProp("AtomName"):
#             name = atom.GetProp("AtomName")
#             idx = atom.GetIdx()
#             if name not in name_to_indices:
#                 name_to_indices[name] = []
#             name_to_indices[name].append(idx)
#
#     duplicates = {name: indices for name, indices in name_to_indices.items() if len(indices) > 1}
#
#     if duplicates:
#         print("⚠️ Обнаружены дубликаты имен:")
#         for name, indices in duplicates.items():
#             print(f"  {name}: индексы {indices}")
#         return False
#
#     print("✅ Все имена атомов уникальны")
#     return True

def remove_extra_H(top, extra_pattern_H = 'HW'):
    atom_to_remove = [atom for atom in top.atoms if extra_pattern_H in atom.name]
    print(f"Найденные атомы для удаления: {atom_to_remove}")
    # Удаление атомов из списка атомов
    for atom in atom_to_remove[::-1]:
        top.atoms.remove(atom)

    # Ручное удаление связей, углов и диэдральных углов
    bonds_to_remove = [
        bond for bond in top.bonds if any(atom in atom_to_remove for atom in (bond.atom1, bond.atom2))
    ]
    angles_to_remove = [
        angle for angle in top.angles if any(atom in atom_to_remove for atom in (angle.atom1, angle.atom2, angle.atom3))
    ]
    dihedrals_to_remove = [
        dihedral for dihedral in top.dihedrals if any(atom in atom_to_remove for atom in (dihedral.atom1, dihedral.atom2, dihedral.atom3, dihedral.atom4))
    ]

    # Удаление связей
    for bond in bonds_to_remove:
        top.bonds.remove(bond)

    # Удаление углов
    for angle in angles_to_remove:
        top.angles.remove(angle)

    # Удаление диэдральных углов
    for dihedral in dihedrals_to_remove:
        top.dihedrals.remove(dihedral)

# Сохранение измененного файла топологии
# top.write(f"Acpype_data/{acpype_name}.acpype/{acpype_name}_GMX_cleaned.itp")
@logged
def hdb_generator(mol_chem, resname='MOD', resid=1, segid='A'):
    """
    Принимает на вход:
     - mol путь к файлу или rdkit.Chem объект
    Модифицирует имена атомов в молекуле по правилам аминокислот:
    - N, H, CA, C, O имеют стандартные имена
    - Боковые атомы получают буквенные метки по греческому алфавиту
    - Протоны наследуют имя родительского атома и получают числовой суффикс
    """
    def find_common_prefix(strings):
        if not strings:
            return ""

        first = strings[0]
        for i in range(len(first), 0, -1):
            prefix = first[:i]
            if all(s.startswith(prefix) for s in strings[1:]):
                return prefix
        return ""

    hdb = ['1	1	H	N	-C	CA	\n',
           '1	5	HA	CA	N	CB	C\n']
    
    for atom in mol_chem.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        current_idx = atom.GetIdx()
        current_atom_name = atom.GetProp('AtomName')
        protons = []
        visited_atoms = {current_atom_name: current_idx}
        for n_atom in atom.GetNeighbors():
            atom_name = n_atom.GetProp('AtomName')
            atom_idx = n_atom.GetIdx()
            if n_atom.GetSymbol() != "H":
                visited_atoms[atom_name] = atom_idx
            elif n_atom.GetProp('AtomName') not in ['H','HA', 'HN']:
                protons.append(atom_name)
        if not protons:  # У атома нет протонов идем к слежующему атому
            continue
        # Если H на атоме с валентностью 2
        if len(visited_atoms) == 2:  # Будет только 1 тяжелый сосед
            atom_2 = mol_chem.GetAtomWithIdx(min(visited_atoms.values()))
            # # Берем индекс соседа и находим его соседа с наименьшим индексом
            atom_3 = min(filter(lambda n: n.GetSymbol() != "H" and
                                n.GetIdx() not in visited_atoms.values(), atom_2.GetNeighbors()),
                         key=lambda n: n.GetIdx(), default=None)
            if atom_3:
                atom_3_name = atom_3.GetProp('AtomName')
                atom_3_idx = atom_3.GetIdx()
                visited_atoms[atom_3_name] = atom_3_idx

        hyb = atom.GetHybridization()
        n_hydrogens = len(protons)
        n_heavy_neighbors = int(atom.GetDegree()) - n_hydrogens
        if hyb == Chem.HybridizationType.SP3:
            if n_hydrogens == 1 and n_heavy_neighbors == 3:
                geom_n = 5  # sp3 углерод с 1 протоном
            elif n_hydrogens == 2 and n_heavy_neighbors == 2:
                geom_n = 6  # sp3 углерод с 2 протоном
            elif n_hydrogens == 3 and n_heavy_neighbors == 1:
                geom_n = 4  # sp3 углерод с 3 протоном
            elif n_hydrogens == 1 and n_heavy_neighbors == 1:
                geom_n = 2  # sp3 углерод с 1 протоном
            else:
                print(f"Не учтенный вариант {current_atom_name}:",
                     f"{hyb}, {n_hydrogens}, {n_heavy_neighbors}", sep='\n')
                geom_n = '-'
        elif hyb == Chem.HybridizationType.SP2:
            if n_hydrogens == 1 and n_heavy_neighbors == 2:
                geom_n = 1
            elif n_hydrogens == 2 and n_heavy_neighbors == 1:
                geom_n = 3
            elif n_hydrogens == 1 and n_heavy_neighbors == 1:
                geom_n = 2
            else:
                print(f"Не учтенный вариант {current_atom_name}:",
                     f"{hyb}, {n_hydrogens}, {n_heavy_neighbors}", sep='\n')
                geom_n = '-'
        else:
            if protons:
                print("Существуют не описанные протоны:")
                print(atom.GetProp("AtomName"), hyb)
                print(protons)
                geom_n = '-'
            
        if protons: # ['H'] ['HB1', 'HB2'] ['HC1', 'HC2', 'HC3']
            # print(protons)
            n_H = str(len(protons))
            H_name = find_common_prefix(protons)
            if H_name not in ['H', 'HA', 'HN']:
                hdb.append("{}\t{}\t{}\t{}\n".format(n_H, geom_n, H_name, "\t".join(visited_atoms)))
                # ['CB', 'CA', 'CG'] ['CK', 'CI'] ['CA', 'N', 'C', 'CB']
    # print(heavy_atoms)
    hdb = [f'{resname}\t{len(hdb)}\n'] + hdb
    return hdb
    #     print(neighbors, sep = '\n')
    # print(heavy_atoms)

@logged
def check_atomtypes(top, path_to_atp = '', param_folder = ''):
    exist_types = []
    with open(path_to_atp, 'r') as atomtypes:
        file = atomtypes.readlines()
    for line in file:
        exist_types.append(line.split()[0])

    atom_types = {str(atom.atom_type) : atom.mass for atom in top}
    add_atom_types = []       
    for atom_type, atom_mass in atom_types.items():
        if atom_type not in exist_types:
            add_atom_types.append('%-2s%24.5f\n' % (atom_type, atom_mass))
    
    if param_folder:
        os.makedirs(param_folder, exist_ok=True)
    save_path = f'{param_folder}/atomtypes.atp'
    
    if add_atom_types:
        add_atom_types.extend(file)
        # print(f'In {path_to_atp}\nAdd line(-s):\n{add_atom_types}')
        
        with open (save_path, 'w') as atomtypes:
            atomtypes.writelines(add_atom_types)
        print(f'Добавлены недостающие типы атомов в {save_path}')
    else:
        with open (save_path, 'w') as atomtypes:
            atomtypes.writelines(file)
        print(f'Все используемые типы атомов указаны в {path_to_atp}')

        
@logged
def make_r2b(path_to_r2b = '', reference_aa = '', add_aa_name = '', param_folder = '', out=False):
    with open (path_to_r2b, 'r') as r2b:
        r2b_list = r2b.readlines()
    for i, r2b_line in enumerate(r2b_list):
        rtp_str = r2b_line.upper()
        if reference_aa.upper() == rtp_str.split()[0]:
            print(f'Replese old srt:\n{rtp_str}')
            add_rtp_line = rtp_str.replace('-', add_aa_name.upper(),1)
            print(f'To new str:\n{add_rtp_line}')
            r2b_list[i] = add_rtp_line

    save_path = f'{param_folder}/aminoacids.r2b'
    with open(save_path, 'w') as f:
        f.writelines(r2b_list)
    print(f'Save in  {save_path}')
