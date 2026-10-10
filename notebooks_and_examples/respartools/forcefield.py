"""Файлы силового поля GROMACS: rtp (read_rtp, write_rtp), itp (read_itp), acpype (run_acpype), hdb, atp, r2b."""
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
import shutil

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


# -----------------------------------------------------------------------------
# rtp: один парсер и один писатель на весь пакет
# -----------------------------------------------------------------------------
RTP_SECTIONS = ('atoms', 'bonds', 'angles', 'dihedrals', 'impropers', 'exclusions', 'cmap')
RTP_BONDEDTYPES = [1, 1, 9, 4, 1, 3, 1, 0]   # amber14sb: bonds angles dihedrals impropers all_dihedrals nrexcl HH14 RemoveDih


def read_rtp(path):
    """
    Файл .rtp GROMACS. Комментарии (после ;) отбрасываются.
    Возвращает (bondedtypes - список чисел или None,
                {остаток: {'atoms': [(имя, тип, заряд, группа)], 'bonds': [[поля]], 'angles': ..., ...}})
    В секциях кроме atoms строка хранится списком полей как есть (имена атомов, затем параметры).
    """
    bondedtypes, residues, current, section = None, {}, None, None
    with open(path) as file:
        for line in file:
            text = line.split(';')[0].strip()
            if not text:
                continue
            header = re.fullmatch(r'\[\s*(\S+)\s*\]', text)
            if header:
                name = header.group(1)
                if name == 'bondedtypes':
                    current, section = None, 'bondedtypes'
                elif name in RTP_SECTIONS:
                    if current is None:
                        raise ValueError(f'{path}: секция [ {name} ] вне остатка')
                    section = name
                    current.setdefault(name, [])
                else:
                    current, section = residues.setdefault(name, {}), None
                continue
            parts = text.split()
            if section == 'bondedtypes':
                bondedtypes = [int(x) for x in parts]
            elif section == 'atoms':
                current['atoms'].append((parts[0], parts[1], float(parts[2]), int(parts[3])))
            elif section is not None:
                current[section].append(parts)
    return bondedtypes, residues


def write_rtp(path, residues, bondedtypes=RTP_BONDEDTYPES):
    """
    Пишет .rtp: заголовок [ bondedtypes ] и остатки {имя: {'atoms': [(имя, тип, заряд, группа)], ...}}
    (формат read_rtp). Заряды пишутся без округления.
    """
    lines = ['[ bondedtypes ]\n',
             '; bonds  angles  dihedrals  impropers all_dihedrals nrexcl HH14 RemoveDih\n',
             ' ' + ' '.join(f'{x:6d}' for x in bondedtypes) + '\n']
    for resname, res in residues.items():
        lines.append(f'\n[ {resname} ]\n')
        for section in RTP_SECTIONS:
            if section not in res:
                continue
            lines.append(f' [ {section} ]\n')
            if section == 'atoms':
                lines += [f'{n:>6} {t:>6} {q!r:>12} {g:5d}\n' for n, t, q, g in res['atoms']]
            else:
                lines += [''.join(f'{x:>6} ' for x in row).rstrip() + '\n' for row in res[section]]
    with open(path, 'w') as file:
        file.writelines(lines)
    return path


# -----------------------------------------------------------------------------
# acpype (шаг 2) и чтение itp
# -----------------------------------------------------------------------------
ITP_SECTIONS = ('atoms', 'bonds', 'pairs', 'angles', 'dihedrals')


def read_itp(path):
    """
    Секции молекулы из .itp GROMACS (acpype): {'atoms': [[поля]], 'bonds': ..., 'dihedrals': ...}.
    Строка - список полей без комментария; номера атомов - как в файле (с 1). Секций
    [ dihedrals ] в acpype две (собственные функция 9, несобственные функция 4) - они идут
    в один список, функция - в 5-м поле.
    """
    sections, current = {}, None
    with open(path) as file:
        for line in file:
            text = line.split(';')[0].strip()
            if not text:
                continue
            header = re.fullmatch(r'\[\s*(\S+)\s*\]', text)
            if header:
                current = header.group(1) if header.group(1) in ITP_SECTIONS else None
                if current:
                    sections.setdefault(current, [])
                continue
            if current:
                sections[current].append(text.split())
    return sections


def _acpype_bin(acpype_bin=None):
    """acpype: переданный путь, из PATH или из окружения topmol2 рядом с текущим."""
    if acpype_bin:
        return acpype_bin
    found = shutil.which('acpype')
    if found:
        return found
    envs = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
    candidate = os.path.join(envs, 'topmol2', 'bin', 'acpype')
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError('Нет acpype: окружение topmol2 (acpype, ambertools) или передайте acpype_bin.')


@logged
def run_acpype(mol, workdir, atom_type='amber', charge_type='gas', acpype_bin=None, random_seed=42):
    """
    Шаг 2: топология acpype (типы атомов и параметры связей, углов, диэдралов) для остатка.
    acpype не принимает атомы с оборванной валентностью, поэтому, если в mol ещё нет
    водородов шага 4.1 (свойство атома AddedH), они добавляются здесь
    (add_protons_and_renumber_H) - только для acpype, в mol их нет. Вход acpype - MOL-файл
    прямо из RDKit (порядки связей и 3D-координаты, без obabel), заряд - формальный заряд
    молекулы (не руками). Порядок атомов itp сверяется с молекулой по элементам (по массе).

    Записывает в атомы mol 'AcpypeType', в молекулу 'AcpypeItp' (путь к itp; атомы itp с
    номером больше числа атомов остатка - добавленные водороды). Заряды acpype (charge_type)
    не используются: заряды - шаг 5/6. Меняет и возвращает mol.
    """
    name = mol.GetProp('ResName') if mol.HasProp('ResName') else \
        mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName().strip()
    os.makedirs(workdir, exist_ok=True)
    has_added = any(a.HasProp('AddedH') and a.GetBoolProp('AddedH') for a in mol.GetAtoms())
    if has_added:
        work = Chem.Mol(mol)
    else:
        info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
        work = add_protons_and_renumber_H(mol, resname=name, resid=info.GetResidueNumber() if info else 1,
                                          segid=info.GetChainId() if info else 'A')
    if not work.GetNumConformers() or not work.GetConformer().Is3D():
        work = set_mol_coords(work, coords='3D', random_seed=random_seed)
    mol_path = os.path.join(workdir, f'{name}.mol')
    Chem.MolToMolFile(work, mol_path)
    net = Chem.GetFormalCharge(mol)
    cmd = [_acpype_bin(acpype_bin), '-i', f'{name}.mol', '-c', charge_type, '-a', atom_type,
           '-n', str(net), '-b', name, '-o', 'gmx']
    run = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    with open(os.path.join(workdir, 'acpype.log'), 'w') as log:
        log.write(run.stdout + run.stderr)
    itp = os.path.join(workdir, f'{name}.acpype', f'{name}_GMX.itp')
    if run.returncode != 0 or not os.path.exists(itp):
        tail = (run.stdout + run.stderr).strip().splitlines()[-3:]
        raise RuntimeError(f'acpype не построил топологию ({os.path.join(workdir, "acpype.log")}): '
                           + ' | '.join(tail))
    atoms = read_itp(itp)['atoms']
    table = Chem.GetPeriodicTable()
    if len(atoms) != work.GetNumAtoms():
        raise ValueError(f'В itp {len(atoms)} атомов, во входе acpype {work.GetNumAtoms()}')
    wrong = [(a.GetIdx(), a.GetSymbol(), row[7]) for a, row in zip(work.GetAtoms(), atoms)
             if abs(table.GetAtomicWeight(a.GetSymbol()) - float(row[7])) > 0.1]
    if wrong:
        raise ValueError(f'Порядок атомов itp не совпадает с молекулой (индекс, элемент, масса в itp): {wrong[:5]}')
    for atom, row in zip(mol.GetAtoms(), atoms):
        atom.SetProp('AcpypeType', row[1])
    mol.SetProp('AcpypeItp', os.path.abspath(itp))
    print(f'acpype ({atom_type}, заряд {net:+d}): {len(atoms)} атомов'
          + ('' if has_added else f' (из них {len(atoms) - mol.GetNumAtoms()} водорода только для acpype)')
          + f', типы записаны в AcpypeType; {itp}')
    return mol


from .residue import set_mol_coords, add_protons_and_renumber_H  # noqa: E402
