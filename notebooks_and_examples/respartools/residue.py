"""Остаток: перенумерация, имена атомов, остов, координаты и стереохимия, параметры остатка, протонирование."""
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

from .log import log_note, logged
from .utils import increment_name, print_green, print_red


@logged
def renumber_residue_atoms(mol, ref_base_name=None, residue_type='protein',
                           path_to_ref_mol=None, only_heavy_mapping=True, timeout=3,
                           strict_parent=True):
    """
    Перенумеровывает атомы остатка: основание (атомы родительского остатка) - в порядке
    шаблона, остальные атомы - обходом молекулярного графа от основания.

    1. В остатке ищется родительский остаток (find_ref_residue); его атомы получают номера
       в том порядке, в котором стоят в шаблоне (amber: N H CA HA CB HB1 HB2 ... C O).
    2. Новые атомы вставляются сразу за каноническим тяжёлым атомом, к которому пришиты
       (и за его водородами): для метки на Cys - N H CA HA CB HB1 HB2 SG [метка] C O,
       для N-метилирования - N H [метил] CA ...
    3. Внутри вставки - обход в ширину, каждый водород сразу за своим тяжёлым атомом.
       Порядок ветвей задаётся каноническим рангом RDKit без учёта стереохимии, поэтому
       результат не зависит от исходного порядка атомов (например, от того, как был записан
       SMILES) и от R/S стереоцентров.
    Если индексы уже в таком порядке, возвращается копия молекулы без изменений.

    Аргументы:
        mol (Chem.Mol) - остаток (с H или без H)
        ref_base_name (str) - родительский остаток (см. find_ref_residue), обязателен
        residue_type (str) - тип остатка из REF_TEMPLATES ('protein', ...)
        path_to_ref_mol (str) - папка с шаблонами или путь к шаблону
        only_heavy_mapping (bool) - MCS только по тяжелым атомам
        timeout (int) - таймаут поиска MCS, с
        strict_parent (bool) - см. find_ref_residue
    Возвращает:
        Chem.Mol - перенумерованная молекула. Свойства: 'RefResidue' - имя шаблона,
        'OldIndices' - JSON-список: на позиции нового индекса стоит старый индекс атома,
        'ParentAtoms' - JSON-список новых индексов атомов, сопоставленных с шаблоном
        родительского остатка (для подсветки: draw_molecule(highlight_atoms=...)).
    """
    _, match_data, _ = find_ref_residue({'residue': mol}, path_to_ref_mol=path_to_ref_mol,
                                     ref_base_name=ref_base_name, residue_type=residue_type,
                                     only_heavy_mapping=only_heavy_mapping, timeout=timeout,
                                     strict_parent=strict_parent)
    ref_name, mapping = next(iter(match_data['mon_pol_matches'].items()))

    # канонический ранг атомов: не зависит от исходной нумерации и от стереохимии
    # (стереохимия, прочитанная из случайной 3D-структуры, меняла порядок при каждом запуске)
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=False))
    placed = set(mapping)

    def by_rank(atoms):
        return sorted(atoms, key=lambda atom: rank[atom.GetIdx()])

    def new_block(anchor_idx):
        """Новые атомы, пришитые к каноническому атому anchor_idx: обход в ширину."""
        block = []
        anchor = mol.GetAtomWithIdx(anchor_idx)
        # лишние водороды самого канонического атома (например, у изменённой NZ)
        for h in by_rank(anchor.GetNeighbors()):
            if h.GetAtomicNum() == 1 and h.GetIdx() not in placed:
                placed.add(h.GetIdx())
                block.append(h.GetIdx())
        queue = [a.GetIdx() for a in by_rank(anchor.GetNeighbors())
                 if a.GetAtomicNum() > 1 and a.GetIdx() not in placed]
        placed.update(queue)
        while queue:
            idx = queue.pop(0)
            block.append(idx)
            atom = mol.GetAtomWithIdx(idx)
            for nbr in by_rank(atom.GetNeighbors()):
                if nbr.GetIdx() in placed:
                    continue
                placed.add(nbr.GetIdx())
                if nbr.GetAtomicNum() == 1:
                    block.append(nbr.GetIdx())
                else:
                    queue.append(nbr.GetIdx())
        return block

    # канонические атомы в порядке шаблона; вставка новых атомов - перед следующим
    # каноническим тяжёлым атомом, т.е. после водородов своего якоря
    canonical = sorted(mapping, key=mapping.get)
    new_order, anchor = [], None
    for idx in canonical:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1:
            if anchor is not None:
                new_order += new_block(anchor)
            anchor = idx
        new_order.append(idx)
    if anchor is not None:
        new_order += new_block(anchor)

    leftover = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in placed]
    if leftover:
        print_red(f'⚠ Атомы {leftover} не связаны с основанием остатка и поставлены в конец.')
        new_order += leftover

    if new_order == list(range(mol.GetNumAtoms())):
        print_green('Индексы атомов уже канонические, перенумерация не требуется.')
        new_mol = Chem.Mol(mol)
    else:
        moved = [(old, new) for new, old in enumerate(new_order) if old != new]
        print(f'Перенумеровано атомов: {len(moved)} из {mol.GetNumAtoms()} '
              f'(канонических: {len(canonical)})')
        new_mol = Chem.RenumberAtoms(mol, new_order)

    new_mol.SetProp('RefResidue', ref_name)
    new_mol.SetProp('OldIndices', json.dumps(new_order))
    new_index = {old: new for new, old in enumerate(new_order)}
    new_mol.SetProp('ParentAtoms', json.dumps(sorted(new_index[idx] for idx in canonical)))
    return new_mol

@logged
def generate_atom_names_by_ref_aa(mod_aa_mol, ref_aa_mol, dict_match, output_path):
    """
    Переименовывает атомы в модифицированной аминокислоте на основе референсной.
    
    Аргументы:
        mod_aa_mol: RDKit объект Chem.Mol для модифицированной молекулы.
        ref_aa_mol: RDKit объект Chem.Mol для референсной молекулы.
        dict_match: Словарь сопоставления индексов атомов (модифицированные -> референсные).
        output_path: Путь для сохранения модифицированного PDB файла.
    """
    # Получаем текущие имена атомов модифицированной молекулы
    all_mod_atom_names = [str(atom.GetProp("AtomName")) if atom.HasProp("AtomName") else "" for atom in mod_aa_mol.GetAtoms()]
    all_ref_atom_names = [str(atom.GetProp("AtomName")) if atom.HasProp("AtomName") else "" for atom in ref_aa_mol.GetAtoms()]
    print(all_mod_atom_names)

    for mod_indx, ref_indx in dict_match.items():
        mod_atom_name = all_mod_atom_names[mod_indx]
        ref_atom_name = all_ref_atom_names[ref_indx]

        # Если имя из референсной молекулы уже есть в модифицированной
        if ref_atom_name in all_mod_atom_names:
            new_name = ref_atom_name
            counter = 1
            while new_name in all_mod_atom_names:
                # Изменяем только числовую часть имени
                match = re.match(r"(\D+)(\d*)", ref_atom_name)
                if match:
                    prefix = match.group(1)  # Буквенная часть
                    suffix = match.group(2)  # Числовая часть (может быть пустой)
                    new_name = f"{prefix}{int(suffix) + counter if suffix else counter}"
                else:
                    # Если имя не содержит числовой части, добавляем ее
                    new_name = f"{ref_atom_name}{counter}"
                counter += 1

            print(f'Переименовываем {mod_indx}:{ref_atom_name} -> {mod_indx}:{new_name}')
            index_rename_mod_atom = all_mod_atom_names.index(ref_atom_name)
            all_mod_atom_names[index_rename_mod_atom] = new_name
            all_mod_atom_names[mod_indx] = ref_atom_name
        else:
            # Если имени нет в модифицированной молекуле, обновляем напрямую
            all_mod_atom_names[mod_indx] = ref_atom_name


        print(f'({mod_indx}:{mod_atom_name}) -> ({ref_indx}:{ref_atom_name})')

    # Проверяем, что все имена уникальны
    name_counts = Counter(all_mod_atom_names)
    duplicates = [name for name, count in name_counts.items() if count > 1]

    if duplicates:
        print("Обнаружены повторяющиеся имена атомов в модифицированной молекуле:")
        for name in duplicates:
            print(f"Имя: {name}, количество повторений: {name_counts[name]}")
        raise ValueError("Есть повторяющиеся имена атомов. Проверьте логи.")
    else:
        print("Все имена атомов уникальны.")

    return all_mod_atom_names        


def rdkit_pdb_modification_old(rdkit_mol, resname='MOD', resid=1, segid='A'):
    """
    Модифицирует имена атомов в молекуле по правилам аминокислот:
    - N, H, CA, C, O имеют стандартные имена
    - Боковые атомы получают буквенные метки по греческому алфавиту
    - Протоны наследуют имя родительского атома и получают числовой суффикс
    """
    greek_alphabet = {0: 'B', 1: 'G', 2: 'D', 3: 'E', 4: 'Z', 5: 'H', 
                      6: 'T', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 
                      12: 'X', 13: 'O', 14: 'P', 15: 'R', 16: 'S'}

    # Найдём индекс CA (альфа-углерода)
    submatch = rdkit_mol.GetSubstructMatch(Chem.MolFromSmarts('[H][N]C([H])C=O'))
    if not submatch:
        raise ValueError("Не удалось найти структуру аминокислоты [H][N]CC=O")
    idx_H, idx_N, idx_CA, idx_HA ,idx_C, idx_O = submatch[0], submatch[1], submatch[2], submatch[3], submatch[4], submatch[5] 

    # Префиксы по индексам атомов
    atom_names = {idx_H: {'atom_symbol': 'H','atom_letter':''}, idx_N: {'atom_symbol': 'N','atom_letter':''}, 
                  idx_CA: {'atom_symbol': 'C','atom_letter':'A'}, idx_HA: {'atom_symbol': 'H','atom_letter':'A'},
                  idx_C: {'atom_symbol': 'C','atom_letter':''}, idx_O: {'atom_symbol': 'O','atom_letter':''}}
    # Старт нумерации с атомов, следующих за CA
    # visited = set(atom_names.keys())
    # atom_names = {}
    greek_counter = 0
    hydrogen_counts = {}

    queue = [idx_CA]
    while queue:
        current_idx = queue.pop(0)
        current_atom = rdkit_mol.GetAtomWithIdx(current_idx)
        neighbors = defaultdict(int)

        for atom in current_atom.GetNeighbors():
            if atom.GetIdx() not in atom_names:
                neighbors[atom.GetAtomicNum()] += 1

        
        count_hidrogen = neighbors[1] 
        count_heavy = sum(neighbors.values()) - neighbors.get(1, 0) 

        hidrogen_index = 1 if count_hidrogen > 1 else None
        heavy_index = 1 if count_heavy > 1 else None
            
        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in atom_names.keys():
                continue
                
            if neighbor.GetAtomicNum() != 1:
                # Тяжёлый атом
                symbol = neighbor.GetSymbol()
                if heavy_index and count_heavy > 1:
                    greek_index = greek_alphabet.get(greek_counter, f"Y{greek_counter}") + str(count_heavy)
                    atom_names[nbr_idx] = {'atom_symbol': symbol,'atom_letter': greek_index}
                    count_heavy -= 1
                    queue.append(nbr_idx)
                elif heavy_index and count_heavy == 1:
                    greek_index = greek_alphabet.get(greek_counter, f"Y{greek_counter}") + str(count_heavy)
                    atom_names[nbr_idx] = {'atom_symbol': symbol,'atom_letter': greek_index}
                    greek_counter += 1
                    queue.append(nbr_idx)
                else:
                    greek_index = greek_alphabet.get(greek_counter, f"Y{greek_counter}")
                    atom_names[nbr_idx] = {'atom_symbol': symbol,'atom_letter': greek_index}
                    greek_counter += 1
                    queue.append(nbr_idx)
            else :
                # Это водород — имя зависит от родителя
                parent_letter = atom_names[current_idx]['atom_letter']
                if hidrogen_index and count_hidrogen > 1:
                    greek_index = parent_letter + str(count_hidrogen)
                    atom_names[nbr_idx] = {'atom_symbol': 'H','atom_letter': greek_index}
                    count_hidrogen -= 1
                elif hidrogen_index and count_hidrogen == 1:
                    greek_index = parent_letter + str(count_hidrogen)
                    atom_names[nbr_idx] = {'atom_symbol': 'H','atom_letter': greek_index}
                    count_hidrogen -= 1
                else:
                    atom_names[nbr_idx] = {'atom_symbol': 'H','atom_letter': parent_letter} #f"H{parent_name}{hydrogen_counts[parent_name]}"

    # Объединяем всё
    for atom in rdkit_mol.GetAtoms():
        idx = atom.GetIdx()
        name = atom_names[idx]['atom_symbol']+atom_names[idx]['atom_letter']
        name = name[:4].ljust(4)  # PDB формат требует длину 4 символа

        info = Chem.AtomPDBResidueInfo()
        info.SetName(name)
        info.SetResidueName(resname)
        info.SetResidueNumber(resid)
        info.SetChainId(segid)
        atom.SetProp("AtomName", name.strip())
        atom.SetMonomerInfo(info)

    return rdkit_mol  

def find_backbone_match(rdkit_mol, C_terminal=False):
    """
    Находит атомы остова аминокислотного остатка по шаблону H-N-CA(H)-C(=O)
    (тот же, что в rdkit_pdb_modification_old). Водороды должны быть явными атомами.

    Аргументы:
        rdkit_mol (Chem.Mol) - остаток
        C_terminal (bool) - C-концевой остаток: у C два кислорода (OC1 - двойная связь, OC2 - одинарная)
    Возвращает:
        (CA_idx, backbone) - backbone = [N, CA, C, O] или [N, CA, C, OC1, OC2] при C_terminal
    Если подходящих остовов нет или их больше одного, вызывает ValueError: выбор был бы угадыванием.
    """
    smarts = '[H][N][C]([H])C(=O)[O]' if C_terminal else '[H][N][C]([H])C=O'
    matches = rdkit_mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=False)
    # совпадения, отличающиеся только выбором H при N и CA, - это один и тот же остов
    backbones = sorted({(m[1], m[2], m[4], m[5], m[6]) if C_terminal else (m[1], m[2], m[4], m[5])
                        for m in matches})
    if not backbones:
        raise ValueError(f"Остов аминокислоты ({smarts}) не найден: проверьте, что водороды явные "
                         f"и что C_terminal={C_terminal} соответствует остатку.")
    if len(backbones) > 1:
        raise ValueError(f"Найдено {len(backbones)} возможных остовов (индексы N, CA, C, O...): "
                         f"{backbones}. Остов неоднозначен.")
    backbone = list(backbones[0])
    return backbone[1], backbone


GREEK_LEVELS = 'BGDEZHTIKLMN'  # буква уровня боковой цепи: 1 связь от CA - B, 2 - G, 3 - D, ...
NAMING_METHODS = ('greek', 'index')


class _NamingError(ValueError):
    """Буквенный способ неприменим к остатку (слишком много уровней, длинные имена)."""


def _greek_heavy_names(mol, CA_idx, heavy_atoms):
    """
    Буквенные имена тяжёлых атомов боковой цепи: элемент + буква уровня (число связей от CA)
    + номер, если на уровне несколько атомов (1..n в порядке индексов).
    heavy_atoms - атомы, которые нужно назвать; путь от CA идёт только через них.
    Возвращает ({индекс: имя}, [индексы, не связанные с CA через heavy_atoms]).
    """
    depth = {CA_idx: 0}
    queue = deque([CA_idx])
    while queue:
        idx = queue.popleft()
        for nbr in sorted(a.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors()):
            if nbr in heavy_atoms and nbr not in depth:
                depth[nbr] = depth[idx] + 1
                queue.append(nbr)
    levels = defaultdict(list)
    for idx, d in depth.items():
        if idx != CA_idx:
            levels[d].append(idx)
    if levels and max(levels) > len(GREEK_LEVELS):
        raise _NamingError(f'боковая цепь длиннее {len(GREEK_LEVELS)} уровней '
                           f'(самый дальний атом в {max(levels)} связях от CA)')
    names = {}
    for d in sorted(levels):
        atoms = sorted(levels[d])
        for k, idx in enumerate(atoms, 1):
            element = mol.GetAtomWithIdx(idx).GetSymbol().upper()
            names[idx] = f'{element}{GREEK_LEVELS[d - 1]}' + (str(k) if len(atoms) > 1 else '')
    unreached = sorted(set(heavy_atoms) - set(depth))
    return names, unreached


def _hydrogen_names(mol, heavy_names, hydrogens):
    """
    Имена водородов при названных тяжёлых атомах: H + имя атома без элемента, при нескольких
    водородах - ещё номер: N -> H (H1 H2 H3), CA -> HA, CB -> HB1 HB2, SG -> HG, CD1 -> HD11 HD12.
    hydrogens - водороды, которые нужно назвать.
    """
    names = {}
    for idx, name in heavy_names.items():
        atom = mol.GetAtomWithIdx(idx)
        suffix = name[len(atom.GetSymbol()):]
        hs = sorted(h.GetIdx() for h in atom.GetNeighbors()
                    if h.GetAtomicNum() == 1 and h.GetIdx() in hydrogens)
        for k, h_idx in enumerate(hs, 1):
            names[h_idx] = f'H{suffix}' + (str(k) if len(hs) > 1 else '')
    return names


def _pdb_atom_name_field(name, element):
    """Поле имени атома PDB (колонки 13-16): однобуквенный элемент начинается с 14-й колонки."""
    if len(name) < 4 and len(element) == 1:
        return f' {name:<3}'
    return f'{name:<4}'


@logged
def rdkit_pdb_modification(rdkit_mol, resname="MOD", resid=1, segid="A",
                           force_numeric=False, C_terminal=False, naming='greek', parent_atoms=None):
    """
    Назначает атомам остатка PDB-имена и параметры остатка (имя, номер, цепь).
    Меняет и возвращает переданную молекулу (save_aa_chem_to_pdb передаёт сюда копию).
    Имена строятся только по графу молекулы и индексам атомов; старые имена не используются,
    поэтому повторный вызов на той же молекуле даёт те же имена.

    Остов (find_backbone_match): N, CA, C, O (C-концевой остаток: OC1, OC2).

    naming='greek' - буквенный способ (по умолчанию), как в amber14sb.
        Тяжёлые атомы боковой цепи делятся на уровни по числу связей от CA:
        1 - B, 2 - G, 3 - D, 4 - E, 5 - Z, 6 - H, 7 - T, 8 - I, 9 - K, 10 - L, 11 - M, 12 - N.
        Имя = элемент + буква уровня; номер добавляется, только если на уровне несколько
        тяжёлых атомов (1..n в порядке индексов): CB, SG; CD1, CD2.
        Водороды: H + имя атома без элемента, при нескольких водородах - ещё номер:
        при N - H (H1 H2 H3), при CA - HA, HB1 HB2, HG, HD11 HD12.
        Если уровней больше 12 или имя длиннее 4 символов, буквенный способ неприменим:
        функция сообщает об этом и называет атомы способом 'index'.
    naming='index' - атомы родительского остатка (parent_atoms) называются буквенным способом,
        остальные - элемент + индекс атома: C8, N15, H122 (индекс из шага 3.1 виден в имени).
        Без parent_atoms родительскими считаются только атомы остова и их водороды.

    Аргументы:
        rdkit_mol (Chem.Mol) - остаток с явными водородами
        resname, resid, segid - имя (до 3 символов), номер и цепь остатка
        force_numeric (bool) - устаревший параметр, равносилен naming='index'
        C_terminal (bool) - C-концевой остаток (OC1, OC2)
        naming (str) - 'greek' или 'index'
        parent_atoms (list) - индексы атомов родительского остатка (свойство 'ParentAtoms'
            после renumber_residue_atoms); используется при naming='index'
    Возвращает:
        Chem.Mol - та же молекула; имена в свойстве атомов 'AtomName' и в PDB residue info.
    """
    if force_numeric:
        print_red("⚠ force_numeric устарел: используется naming='index'.")
        naming = 'index'
    if naming not in NAMING_METHODS:
        raise ValueError(f"naming='{naming}': допустимые способы {NAMING_METHODS}")

    old_resnames = {atom.GetPDBResidueInfo().GetResidueName().strip()
                    for atom in rdkit_mol.GetAtoms() if atom.GetPDBResidueInfo() is not None}
    old_resnames -= {'', 'UNL'}
    if old_resnames:
        print(f"Атомы уже названы (остаток {', '.join(sorted(old_resnames))}): "
              "имена назначаются заново, старые имена не используются.")

    CA_idx, backbone = find_backbone_match(rdkit_mol, C_terminal)
    backbone_names = ["N", "CA", "C", "OC1", "OC2"] if C_terminal else ["N", "CA", "C", "O"]
    backbone_heavy = dict(zip(backbone, backbone_names))

    heavy_all = {a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetAtomicNum() > 1}
    hydrogens_all = {a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetAtomicNum() == 1}
    side_all = heavy_all - set(backbone)

    def greek_names(heavy_side, hydrogens):
        heavy_names, unreached = _greek_heavy_names(rdkit_mol, CA_idx, heavy_side)
        heavy_names.update(backbone_heavy)
        names = dict(heavy_names)
        names.update(_hydrogen_names(rdkit_mol, heavy_names, hydrogens))
        long_names = sorted(name for name in names.values() if len(name) > 4)
        if long_names:
            raise _NamingError(f'имена длиннее 4 символов: {long_names}')
        return names, unreached

    method = naming
    if naming == 'greek':
        try:
            names, unreached = greek_names(side_all, hydrogens_all)
        except _NamingError as e:
            print_red(f"⚠ Буквенный способ неприменим: {e}. Атомы названы способом 'index'"
                      + ("." if parent_atoms is not None else
                         " (parent_atoms не переданы: буквенные имена только у остова)."))
            method = 'index'
        else:
            if unreached:
                raise ValueError(f'Атомы {unreached} не связаны с CA через боковую цепь: '
                                 'это не один остаток.')

    if method == 'index':
        if parent_atoms is None:
            parent = set(backbone) | {h.GetIdx() for idx in backbone
                                      for h in rdkit_mol.GetAtomWithIdx(idx).GetNeighbors()
                                      if h.GetAtomicNum() == 1}
        else:
            parent = {int(i) for i in parent_atoms}
            bad = sorted(i for i in parent if not 0 <= i < rdkit_mol.GetNumAtoms())
            if bad:
                raise ValueError(f'parent_atoms: индексы вне молекулы ({rdkit_mol.GetNumAtoms()} атомов): {bad}')
            missing = sorted(set(backbone) - parent)
            if missing:
                raise ValueError(f'parent_atoms не содержат атомы остова {missing}: '
                                 'список не от этой молекулы или от другой нумерации.')
        try:
            names, unreached = greek_names(side_all & parent, hydrogens_all & parent)
        except _NamingError as e:
            raise ValueError(f'Родительский остаток нельзя назвать буквенным способом: {e}')
        if unreached:
            raise ValueError(f'Атомы родительского остатка {unreached} не связаны с CA '
                             'через другие атомы родительского остатка.')
        for atom in rdkit_mol.GetAtoms():
            if atom.GetIdx() not in names:
                names[atom.GetIdx()] = f'{atom.GetSymbol().upper()}{atom.GetIdx()}'
        long_names = sorted(name for name in names.values() if len(name) > 4)
        if long_names:
            raise ValueError(f'Имена длиннее 4 символов (поле PDB): {long_names}')

    counts = Counter(names.values())
    duplicates = {name: sorted(i for i, n in names.items() if n == name)
                  for name, c in counts.items() if c > 1}
    if duplicates:
        raise ValueError(f'Повторяющиеся имена атомов (имя: индексы): {duplicates}')

    for atom in rdkit_mol.GetAtoms():
        name = names[atom.GetIdx()]
        info = Chem.AtomPDBResidueInfo()
        info.SetName(_pdb_atom_name_field(name, atom.GetSymbol()))
        info.SetResidueName(resname)
        info.SetResidueNumber(resid)
        info.SetChainId(segid)
        info.SetIsHeteroAtom(False)
        atom.SetProp("AtomName", name)
        atom.SetMonomerInfo(info)

    if method == 'greek':
        print(f"Имена атомов (буквенный способ): {' '.join(names[i] for i in sorted(names))}")
    else:
        rule_named = [i for i in sorted(names) if names[i] != f'{rdkit_mol.GetAtomWithIdx(i).GetSymbol().upper()}{i}']
        print(f"Имена атомов (способ 'index'): родительский остаток - "
              f"{' '.join(names[i] for i in rule_named)}; "
              f"остальные {rdkit_mol.GetNumAtoms() - len(rule_named)} атомов - элемент + индекс.")
    log_note('имена атомов', method=method, names={i: names[i] for i in sorted(names)})
    return rdkit_mol


COORD_TYPES = ('2D', '3D')


def _ca_volume(conf, N_idx, CA_idx, C_idx, CB_idx):
    """
    Ориентированный объём у CA: (N-CA)·((C-CA)×(CB-CA)). У L-аминокислот он положительный
    (проверено по всем шаблонам molecules/aminoacids_template/*_H.pdb), у D - отрицательный.
    В отличие от R/S, знак не зависит от приоритетов CIP (L-цистеин - это R).
    """
    p = [np.array(conf.GetAtomPosition(i)) for i in (N_idx, CA_idx, C_idx, CB_idx)]
    return float(np.dot(p[0] - p[1], np.cross(p[2] - p[1], p[3] - p[1])))


def _embed_3d(mol, random_seed):
    if AllChem.EmbedMolecule(mol, randomSeed=random_seed, enforceChirality=True) != 0:
        raise ValueError('Не удалось построить 3D-структуру с заданной стереохимией '
                         '(EmbedMolecule). Проверьте параметр stereo.')


def set_mol_coords(rdkit_mol, coords='2D', stereo=None, ca_config='L', stereo_default='R',
                   random_seed=42):
    """
    Задаёт координаты молекулы: 2D-раскладку или 3D-структуру с заданной стереохимией.
    Меняет переданную молекулу (функции сохранения передают сюда копию).

    coords='2D' (по умолчанию) - плоская раскладка RDKit. Она однозначна, не зависит от
        случайных чисел, и по ней RDKit при чтении PDB не придумывает стереохимию.
        Стереохимия в 2D PDB не сохраняется, параметры stereo и ca_config не используются.
    coords='3D' - 3D-структура (EmbedMolecule + оптимизация UFF) с заданной стереохимией:
        1. stereo={индекс: 'R' или 'S'} - явно заданные стереоцентры (по правилам CIP);
        2. CA аминокислоты (если в молекуле найден остов N-CA-C=O с явными H) -
           ca_config: 'L' (по умолчанию, как в белках), 'D' или None (как остальные центры).
           L/D задаётся по геометрии, а не через R/S: L-цистеин по CIP - R, остальные L - S;
        3. остальные стереоцентры: если конфигурация уже задана в молекуле (например, @ в
           SMILES), она сохраняется; иначе ставится stereo_default ('R' или 'S') и функция
           сообщает, какие центры так заданы;
        4. random_seed фиксирует построение: одна и та же молекула даёт одни и те же координаты.
        После оптимизации конфигурации проверяются по 3D-координатам.

    Аргументы:
        rdkit_mol (Chem.Mol) - молекула (для 3D - с явными водородами)
        coords (str) - '2D' или '3D'
        stereo (dict) - {индекс атома: 'R'/'S'}
        ca_config (str) - 'L', 'D' или None
        stereo_default (str) - 'R' или 'S' для незаданных стереоцентров
        random_seed (int) - затравка случайных чисел для 3D
    Возвращает:
        Chem.Mol - та же молекула с одним конформером.
    """
    if coords not in COORD_TYPES:
        raise ValueError(f"coords='{coords}': допустимые значения {COORD_TYPES}")
    if coords == '2D':
        if stereo:
            print_red("⚠ stereo задаётся только при coords='3D': в 2D стереохимия не сохраняется.")
        rdkit_mol.RemoveAllConformers()
        rdDepictor.Compute2DCoords(rdkit_mol)
        return rdkit_mol

    from rdkit.Chem import rdCIPLabeler
    stereo = {int(k): str(v).upper() for k, v in (stereo or {}).items()}
    if stereo_default not in ('R', 'S'):
        raise ValueError(f"stereo_default='{stereo_default}': допустимо 'R' или 'S'")
    if ca_config not in ('L', 'D', None):
        raise ValueError(f"ca_config='{ca_config}': допустимо 'L', 'D' или None")
    if any(v not in ('R', 'S') for v in stereo.values()):
        raise ValueError(f"stereo: конфигурации задаются как 'R' или 'S': {stereo}")

    centers = [idx for idx, _ in Chem.FindMolChiralCenters(rdkit_mol, includeUnassigned=True,
                                                           useLegacyImplementation=False)]
    bad = sorted(set(stereo) - set(centers))
    if bad:
        raise ValueError(f'stereo: атомы {bad} не стереоцентры (стереоцентры: {centers})')

    # CA и его соседи по остову
    ca = None
    if ca_config is not None:
        try:
            CA_idx, backbone = find_backbone_match(rdkit_mol)
        except ValueError as e:
            print(f'Остов аминокислоты не найден ({e}); CA задаётся как остальные стереоцентры.')
        else:
            N_idx, C_idx = backbone[0], backbone[2]
            cb = [a.GetIdx() for a in rdkit_mol.GetAtomWithIdx(CA_idx).GetNeighbors()
                  if a.GetAtomicNum() > 1 and a.GetIdx() not in (N_idx, C_idx)]
            if CA_idx in stereo:
                print(f'CA ({CA_idx}) задан в stereo: {stereo[CA_idx]}, ca_config не используется.')
            elif len(cb) == 1 and CA_idx in centers:
                ca = (N_idx, CA_idx, C_idx, cb[0])
    ca_idx = ca[1] if ca else None

    # R/S: явно заданные, уже заданные в молекуле, по умолчанию
    target = dict(stereo)
    by_default = []
    for idx in centers:
        if idx in target or idx == ca_idx:
            continue
        tag = rdkit_mol.GetAtomWithIdx(idx).GetChiralTag()
        if tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            continue
        target[idx] = stereo_default
        by_default.append(idx)
    for idx in target:
        rdkit_mol.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    # CA: тег подбирается по ориентации в 3D
    if ca is not None:
        ca_atom = rdkit_mol.GetAtomWithIdx(ca_idx)
        was_set = ca_atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                             Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        if not was_set:
            ca_atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        probe = Chem.Mol(rdkit_mol)
        _embed_3d(probe, random_seed)
        if (_ca_volume(probe.GetConformer(), *ca) > 0) != (ca_config == 'L'):
            ca_atom.InvertChirality()
            if was_set:
                print_red(f'⚠ CA ({ca_idx}) в исходной молекуле не {ca_config}; '
                          f'задаётся {ca_config} (ca_config).')

    # подгонка R/S: метка CIP центра может зависеть от соседних центров, поэтому несколько проходов
    for _ in range(3):
        rdCIPLabeler.AssignCIPLabels(rdkit_mol)
        wrong = [idx for idx, cip in target.items()
                 if rdkit_mol.GetAtomWithIdx(idx).HasProp('_CIPCode')
                 and rdkit_mol.GetAtomWithIdx(idx).GetProp('_CIPCode') != cip]
        if not wrong:
            break
        for idx in wrong:
            rdkit_mol.GetAtomWithIdx(idx).InvertChirality()
    no_label = [idx for idx in target if not rdkit_mol.GetAtomWithIdx(idx).HasProp('_CIPCode')]
    if no_label:
        print_red(f'⚠ Для центров {no_label} метка R/S не определяется (CIP): '
                  'их конфигурация выбрана произвольно, но воспроизводимо.')

    rdkit_mol.RemoveAllConformers()
    _embed_3d(rdkit_mol, random_seed)
    AllChem.UFFOptimizeMolecule(rdkit_mol)

    # проверка по 3D-координатам
    check = Chem.Mol(rdkit_mol)
    Chem.AssignStereochemistryFrom3D(check)
    rdCIPLabeler.AssignCIPLabels(check)

    def cip(idx):
        atom = check.GetAtomWithIdx(idx)
        return atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else '?'

    got = {idx: cip(idx) for idx in target}
    mismatch = {idx: (target[idx], got[idx]) for idx in target
                if idx not in no_label and got[idx] != target[idx]}
    if ca is not None and (_ca_volume(rdkit_mol.GetConformer(), *ca) > 0) != (ca_config == 'L'):
        mismatch[ca_idx] = (ca_config, 'L' if ca_config == 'D' else 'D')
    if mismatch:
        raise ValueError(f'3D-структура не совпала с заданной стереохимией (атом: (задано, получено)): {mismatch}')

    parts = []
    if ca is not None:
        parts.append(f'CA ({ca_idx}) - {ca_config} ({cip(ca_idx)} по CIP)')
    if stereo:
        parts.append('заданы: ' + ', '.join(f'{i} - {c}' for i, c in sorted(stereo.items())))
    kept = sorted(set(centers) - set(target) - {ca_idx})
    if kept:
        parts.append('из молекулы: ' + ', '.join(f'{i} - {cip(i)}' for i in kept))
    if by_default:
        parts.append(f'по умолчанию {stereo_default}: {by_default}')
    print('3D-структура, стереоцентры: ' + ('; '.join(parts) if parts else 'нет'))
    return rdkit_mol

        
def check_PDB_residue_info(atom):
    """
    Проверяет, что параметры остатка были правильно установлены для атома.
    """
    monomer_info = atom.GetMonomerInfo()
    # assert monomer_info.IsValid(), "Информация об остатке не установлена!"
    assert monomer_info.GetName().strip(), "Имя атома не установлено!"
    assert monomer_info.GetResidueName().strip(), "Имя остатка не установлено!"
    assert monomer_info.GetResidueNumber() >= 1, "Номер остатка не установлен!"
    assert monomer_info.GetChainId().strip(), "ID сегмента не установлен!"
    print(f"Параметры остатка для атома {monomer_info.GetName()} установлены корректно.", 
          monomer_info.GetName(), 
          monomer_info.GetResidueName(),
        monomer_info.GetResidueNumber(),
        monomer_info.GetChainId(),sep='\n')

def check_duplicate_atom_names(mol):
    """
    Проверяет наличие атомов с одинаковыми именами в молекуле и возвращает словарь.
    """
    # Получаем список всех имен атомов
    atom_names = [atom.GetProp('AtomName') for atom in mol.GetAtoms()]

    # Подсчитываем частоту появления каждого имени
    # Формируем словарь с именами атомов и их индексами в молекуле
    atom_indices = {}
    for i, atom in enumerate(mol.GetAtoms()):
        name = atom.GetProp('AtomName')
        if name in atom_indices:
            atom_indices[name].append(i)
        else:
            atom_indices[name] = [i]

    # Отбираем только те атомы, которые имеют одинаковые имена
    duplicates = {name: indices for name, indices in atom_indices.items() if len(indices) > 1}

    if duplicates:
        print("Обнаружены атомы с одинаковыми именами:")
        for name, indices in duplicates.items():
            print(f"Имя: '{name}', Индексы атомов: {indices}")
    else:
        print("Дубликаты имен атомов не найдены.")

    return duplicates

def set_PDB_residue_info(atom, atom_name, resname='MOD', resid=1, segid='A'):
    
    info = Chem.AtomPDBResidueInfo()
    info.SetName(_pdb_atom_name_field(atom_name, atom.GetSymbol()))  # колонки 13-16 PDB
    info.SetResidueName(resname)            # Устанавливаем имя остатка
    info.SetResidueNumber(resid)            # Устанавливаем номер остатка
    info.SetChainId(segid)                  # Устанавливаем ID сегмента
    atom.SetMonomerInfo(info)
    # return info
    

@logged
def modifie_residue_info(modified_mol,  index_map, resname='MOD', resid=1, segid='A'):
    """
    Переименовывает атомы по словарю {старое имя: новое имя} и задаёт всем атомам параметры
    остатка (имя, номер, цепь). Если имя атома, не упомянутого в словаре, совпадает с одним
    из новых имён, атом получает следующее свободное имя (increment_name).
    Меняет и возвращает переданную молекулу.
    """
    # ref_mol_atom_dict = {atom.GetIdx(): atom.GetProp('AtomName') for atom in reference_mol.GetAtoms() if atom.GetPDBResidueInfo().GetResidueNumber()==2 }
    
    for atom in modified_mol.GetAtoms():
        atom_name = atom.GetProp('AtomName') 
        
        if atom_name in index_map.keys():
            atom_name = index_map[atom_name]
            atom.SetProp("AtomName", atom_name)
            print(f'Имя атома {atom.GetSymbol()} с индексом {atom.GetIdx()} заменено на имя {atom_name} ')
            set_PDB_residue_info(atom, atom_name, resname, resid, segid) 
        else:
            ref_names = set(index_map.values()) 
            new_name = atom_name
            if atom_name in ref_names:
                while new_name in ref_names:
                    new_name = increment_name(new_name)
                print(f'Имя атома {atom_name} с индексом {atom.GetIdx()} заменено на имя {new_name} ')
                atom_name = new_name
                atom.SetProp("AtomName", atom_name)
            set_PDB_residue_info(atom, atom_name, resname, resid, segid)
    return modified_mol

@logged
def add_names_from_residue(modified_chem, index_map, resname='MOD', resid=1, segid='A'):

    
    mod_mol = list(modified_chem.values())[0] # костыль работы с словарем молекулы
    # ref_mol = list(reference_chem.values())[0] # костыль работы с словарем молекулы
   

    # старые ноутбуки передают сюда целиком словарь данных сопоставления (match_data_dict)
    if isinstance(index_map, dict) and 'mon_pol_matches' in index_map:
        key_map = next(iter(index_map['mon_pol_matches'].keys()))
        index_map = index_map['mon_pol_matches'][key_map]
    # Присваиваем новые имена атомам в модифицированной молекуле
    # Перезадаем параметры модифицированного остатка  
    mod_residue = modifie_residue_info(mod_mol, index_map, resname, resid, segid)
    
    # Проверка наличия дублирующих имен атомов
    has_duplicates = check_duplicate_atom_names(mod_residue)

    return mod_residue


# =============================================================================
# ИМПОРТИРОВАННЫЕ ФУНКЦИИ
# Перенесены из ячеек ноутбуков <PTM>/1_charge_calculation*.ipynb (Шаг 4.1),
# где они были определены локально. Код перенесён без изменений.
# =============================================================================

# Источник: Lysine_3M/1_charge_calculation.ipynb (идентичная копия в AF_546_*, Lysine_*;
# в Lysine_Cro вариант с префиксом имени водорода 'HW' вместо 'HW1')
@logged
def add_protons_and_renumber_H(modifie_residue, resname='MOD', resid=1, segid='A', h_name='HW1'):
    """
    Шаг 4.1: достраивает водороды у атомов с неполной валентностью. acpype не строит
    топологию для таких атомов, поэтому у вырезанного из тримера остатка появляются протоны,
    которых нет в белке: на N (вместо связи с предыдущим остатком) и на C (вместо связи
    со следующим). В 3_edd_topology они удаляются из топологии.

    1. Атомы остатка сохраняют свои индексы, имена и параметры остатка: новые водороды
       добавляются в конец (индексы и заряды шага 5 совпадают с шагом 4).
    2. Новые водороды получают свободные имена h_name, h_name+1, ... (HW1, HW2, ...),
       параметры остатка resname, resid, segid и свойство атома 'AddedH' = True.
    3. Список индексов новых водородов - в свойстве молекулы 'AddedH' (JSON).
    Атомы с формальным зарядом (например, O- сульфогруппы) не протонируются: RDKit
    считает их валентность полной.

    Аргументы:
        modifie_residue (Chem.Mol) - остаток после шага 4 (у всех атомов есть 'AtomName')
        resname, resid, segid - параметры остатка для новых водородов
        h_name (str) - имя первого нового водорода (в Lysine_Cro использовалось 'HW')
    Возвращает:
        Chem.Mol - новая молекула с 2D-координатами (исходная не меняется).
    """
    n_old = modifie_residue.GetNumAtoms()
    unnamed = [a.GetIdx() for a in modifie_residue.GetAtoms() if not a.HasProp('AtomName')]
    if unnamed:
        raise ValueError(f'У атомов {unnamed} нет имён (AtomName): на вход нужен остаток после шага 4.')
    names = [a.GetProp('AtomName') for a in modifie_residue.GetAtoms()]

    H_modifie_residue = Chem.AddHs(modifie_residue, addCoords=True)
    added = list(range(n_old, H_modifie_residue.GetNumAtoms()))

    report = []
    for idx in added:
        atom = H_modifie_residue.GetAtomWithIdx(idx)
        new_name = h_name
        while new_name in names:
            new_name = increment_name(new_name)
        names.append(new_name)
        set_PDB_residue_info(atom, new_name, resname, resid, segid)
        atom.SetProp('AtomName', new_name)
        atom.SetBoolProp('AddedH', True)
        heavy = atom.GetNeighbors()[0]
        report.append(f"{new_name} - {heavy.GetProp('AtomName')} ({heavy.GetIdx()})")
    H_modifie_residue.SetProp('AddedH', json.dumps(added))
    AllChem.Compute2DCoords(H_modifie_residue)

    if added:
        print(f"Добавлены водороды (имя - атом): {', '.join(report)}")
    else:
        print_green('Все атомы со стандартной валентностью, водороды не добавлены.')
    log_note('добавленные водороды', added={i: r for i, r in zip(added, report)})
    return H_modifie_residue


# Импорт из других модулей пакета - в конце файла: функции этого модуля уже
# определены, поэтому взаимные ссылки модулей друг на друга не мешают импорту.
from .matching import find_ref_residue  # noqa: E402
