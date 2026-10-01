"""Заряды: ограничения из rtp amber14sb, работа со списками зарядов."""
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
from .utils import _PACKAGE_ROOT
        
DEFAULT_RTP = os.path.join(_PACKAGE_ROOT,
                           'amber14sb_parmbsc1_cufix.ff', 'aminoacids.rtp')
BACKBONE_CONSTRAINT_ATOMS = ('N', 'H', 'CA', 'HA', 'C', 'O')


def read_rtp_charges(resname, rtp_path=DEFAULT_RTP):
    """
    Заряды атомов остатка из секции [ atoms ] файла .rtp (GROMACS).
    Возвращает {имя атома: заряд} в порядке файла.
    """
    charges, in_residue, in_atoms = {}, False, False
    with open(rtp_path) as file:
        for line in file:
            text = line.split(';')[0].strip()
            if not text:
                continue
            header = re.fullmatch(r'\[\s*(\S+)\s*\]', text)
            if header:
                name = header.group(1)
                if in_residue and name not in ('atoms',):
                    if name in ('bonds', 'impropers', 'dihedrals', 'angles', 'exclusions', 'cmap'):
                        in_atoms = False
                        continue
                    break  # следующий остаток
                if name == resname:
                    in_residue = True
                elif in_residue and name == 'atoms':
                    in_atoms = True
                continue
            if in_residue and in_atoms:
                parts = text.split()
                charges[parts[0]] = float(parts[2])
    if not charges:
        raise ValueError(f'Остаток {resname} не найден в {rtp_path}')
    return charges


@logged
def charge_constraints_from_rtp(mol, rtp_residue, atom_names=BACKBONE_CONSTRAINT_ATOMS,
                                rtp_path=DEFAULT_RTP):
    """
    Ограничения зарядов для Espaloma (charge(mol, constraints=...)): атомы с именами
    atom_names получают заряды одноимённых атомов остатка rtp_residue из силового поля.
    По умолчанию - атомы остова N H CA HA C O (как во всех ноутбуках лизинов): остов
    модифицированного остатка сохраняет заряды amber14sb, боковая цепь считается Espaloma.

    Аргументы:
        mol (Chem.Mol) - остаток с именами атомов (свойство 'AtomName', шаг 4)
        rtp_residue (str) - остаток в .rtp, чьи заряды берутся (например 'CYS', 'LYS')
        atom_names (list) - имена атомов с фиксированным зарядом
        rtp_path (str) - файл .rtp; по умолчанию aminoacids.rtp amber14sb из репозитория
    Возвращает:
        dict {индекс атома: заряд}
    """
    rtp = read_rtp_charges(rtp_residue, rtp_path)
    index = defaultdict(list)
    for atom in mol.GetAtoms():
        if atom.HasProp('AtomName'):
            index[atom.GetProp('AtomName').strip()].append(atom.GetIdx())
    not_in_rtp = [n for n in atom_names if n not in rtp]
    if not_in_rtp:
        raise ValueError(f'В {rtp_residue} ({rtp_path}) нет атомов {not_in_rtp}; есть: {list(rtp)}')
    problems = {n: index.get(n, []) for n in atom_names if len(index.get(n, [])) != 1}
    if problems:
        raise ValueError(f'Имена атомов должны встречаться в молекуле ровно один раз '
                         f'(имя: найденные индексы): {problems}')
    constraints = {index[n][0]: rtp[n] for n in atom_names}
    print(f'Заряды из {rtp_residue} (имя, индекс, заряд): '
          + ', '.join(f'{n} {index[n][0]} {rtp[n]:+.4f}' for n in atom_names))
    return constraints



## Работа с листом зарядов

@logged
def make_substructure_charge_list(pol_name, charge_array, match_dict, ):
    monomer_charge = [0] * (len(match_dict['substructure'][pol_name].GetAtoms()))
    # charge_array = charge_array.round(5)
    for monomer_index, polymer_index in enumerate(match_dict['mon_pol_matches'][pol_name].values()):
        monomer_charge[monomer_index] = charge_array[polymer_index]
    monomer_charge = np.array(monomer_charge).round(4)
    return monomer_charge

def group_charges(mol, charges):
    """
    Заряды групп: тяжёлый атом вместе со своими водородами (CH2, NH3 и т.п.). По ним удобно
    сравнивать схемы зарядов: AM1-BCC и RESP по-разному делят заряд внутри группы.
    Возвращает {имя тяжёлого атома (или индекс): заряд группы} в порядке атомов.
    """
    if len(charges) != mol.GetNumAtoms():
        raise ValueError(f'Зарядов {len(charges)}, атомов {mol.GetNumAtoms()}')
    groups = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        q = float(charges[atom.GetIdx()]) + sum(float(charges[n.GetIdx()]) for n in atom.GetNeighbors()
                                                 if n.GetAtomicNum() == 1)
        name = atom.GetProp('AtomName') if atom.HasProp('AtomName') else str(atom.GetIdx())
        groups[name] = q
    return groups


@logged
def compare_charges(mol, charge_sets, top=10):
    """
    Сравнивает наборы зарядов одного остатка ({'Espaloma': q1, 'RESP': q2, ...}) по группам
    (group_charges): печатает суммы, среднее и наибольшие расхождения групп между первым
    и остальными наборами. Возвращает {набор: {группа: заряд}}.
    """
    names = list(charge_sets)
    tables = {n: group_charges(mol, q) for n, q in charge_sets.items()}
    ref = names[0]
    for n in names:
        print(f'{n}: сумма {sum(float(x) for x in charge_sets[n]):+.4f}')
    for n in names[1:]:
        diff = {g: tables[n][g] - tables[ref][g] for g in tables[ref]}
        mean = sum(abs(d) for d in diff.values()) / len(diff)
        worst = sorted(diff, key=lambda g: -abs(diff[g]))[:top]
        print(f'{n} - {ref}: среднее |разница групп| {mean:.3f} e; наибольшие: '
              + ', '.join(f'{g} {tables[ref][g]:+.2f} -> {tables[n][g]:+.2f}' for g in worst))
    return tables
