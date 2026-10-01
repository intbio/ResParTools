"""Сопоставление подструктур (MCS), шаблоны родительских остатков, find_ref_residue."""
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
from .utils import _PACKAGE_ROOT, animate, print_red


# def modify_atoms_for_charge(mol):
#     """
#     Модифицирует атомные номера на основе заряда.
#     Формальный заряд добавляется к атомному номеру.
#     """
#     for atom in mol.GetAtoms():
#         charge = atom.GetFormalCharge()
#         atom.SetIntProp("OriginalAtomicNum", atom.GetAtomicNum())  # Сохраняем оригинальный номер
#         atom.SetAtomicNum(atom.GetAtomicNum() + charge * 100)  # Смещаем атомный номер

# def restore_original_atomic_numbers(mol):
#     """
#     Восстанавливает исходные атомные номера после модификации.
#     """
#     for atom in mol.GetAtoms():
#         if atom.HasProp("OriginalAtomicNum"):
#             atom.SetAtomicNum(atom.GetIntProp("OriginalAtomicNum"))
#             atom.ClearProp("OriginalAtomicNum")

def match_chem(mol_chem_1, mol_chem_2, compare_any_bond=False, match_residue_number=None):
    """
    Находит максимальную общую подструктуру (MCS) между двумя молекулами,
    ограничивая поиск атомами, принадлежащими заданному остатку во второй молекуле.
    
    Args:
        mol_chem_1 (Chem.Mol): Первая молекула (мономер).
        mol_chem_2 (Chem.Mol): Вторая молекула (полимер).
        compare_any_bond (bool): Игнорировать порядок связей.
        match_residue_number (int, optional): Номер остатка для фильтрации атомов второй молекулы.
    
    Returns:
        Tuple[Chem.Mol, Dict[int, int]]: Общая подструктура и соответствие атомов.
    """
    
    class CompareElements(rdFMCS.MCSAtomCompare):
        def __call__(self, p, mol1, atom1, mol2, atom2):
            a1 = mol1.GetAtomWithIdx(atom1)
            a2 = mol2.GetAtomWithIdx(atom2)

            # Проверки атомов
            if a1.GetAtomicNum() != a2.GetAtomicNum():
                return False
            if p.MatchValences and a1.GetTotalValence() != a2.GetTotalValence():
                return False
            if p.MatchChiralTag and not self.CheckAtomChirality(p, mol1, atom1, mol2, atom2):
                print(a1.GetProp('AtomName'), a2.GetProp('AtomName'))
                return False
            if p.MatchFormalCharge and not self.CheckAtomCharge(p, mol1, atom1, mol2, atom2):
                print(a1.GetProp('AtomName'), a2.GetProp('AtomName'))
                return False
            if p.RingMatchesRingOnly:
                return self.CheckAtomRingMatch(p, mol1, atom1, mol2, atom2)
            return True
        
    # Функция для фильтрации атомов по номеру остатка
    def filter_atoms_by_residue_number(mol, residue_number):
        """Возвращает новую молекулу, содержащую только атомы с заданным номером остатка."""
        indices_to_keep = []
        for atom in mol.GetAtoms():
            monomer_info = atom.GetMonomerInfo()
            if monomer_info and monomer_info.GetResidueNumber() == residue_number:
                indices_to_keep.append(atom.GetIdx())
        return Chem.PathToSubmol(mol, indices_to_keep)

    # Применяем фильтр к второй молекуле, если задан номер остатка
    if match_residue_number is not None:
        print(f'Matched only with residue namber {match_residue_number}')
        filtered_indices = []  # Список индексов атомов, которые принадлежат нужному остатку
        for atom in mol_chem_2.GetAtoms():
            monomer_info = atom.GetMonomerInfo()
            if monomer_info and monomer_info.GetResidueNumber() == match_residue_number:
                filtered_indices.append(atom.GetIdx())

        # Создаем отфильтрованную молекулу для поиска MCS
        filtered_mol_chem_2 = Chem.PathToSubmol(mol_chem_2, filtered_indices)
        # print(f"Filtered molecule contains {filtered_mol_chem_2.GetNumAtoms()} atoms.")
    else:
        filtered_mol_chem_2 = mol_chem_2

    # Настройки поиска MCS
    params = rdFMCS.MCSParameters()
    params.AtomCompareParameters.MatchValences = True
    params.AtomTyper = CompareElements()
    params.BondCompare = rdFMCS.BondCompare.CompareAny if compare_any_bond else rdFMCS.BondCompare.CompareOrder

    # Поиск MCS
    res = rdFMCS.FindMCS([mol_chem_1, filtered_mol_chem_2], params)
    # print(f'FindMCS result: {res}')
    substructure = Chem.MolFromSmarts(res.smartsString)

    # Сопоставление атомов    
    match_mol1 = mol_chem_1.GetSubstructMatch(substructure)
    match_polymer = mol_chem_2.GetSubstructMatch(substructure)
    # print(match_mol1, match_polymer, sep='\n')
    dict_matches = {}
    # if type_match_dict == 'Indexes':
    dict_matches['Indexes'] = dict(zip(match_mol1, match_polymer))
    # elif type_match_dict == 'Names':
    if mol_chem_1.HasProp('AtomNames') and mol_chem_2.HasProp('AtomNames'):
        mol1_atoms_name = [ mol_chem_1.GetAtomWithIdx(atom_idx).GetProp('AtomName') for atom_idx in match_mol1 ]
        mol2_atoms_name = [ mol_chem_2.GetAtomWithIdx(atom_idx).GetProp('AtomName') for atom_idx in match_polymer ]
        dict_matches['Names'] = dict(zip(mol1_atoms_name, mol2_atoms_name))
    # else:
    #     raise ValueError("Неверное значение type_match_dict. Допустимые значения: 'Indexes' и 'Names'.")
        

    # Копирование зарядов из исходной молекулы
    for idx in match_polymer:
        atom = mol_chem_2.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() != 0:
            substructure.GetAtomWithIdx(match_polymer.index(idx)).SetFormalCharge(atom.GetFormalCharge())

    return substructure, dict_matches

def remove_hydrogens_preserve_indices(mol):
    """
    Убирает все протоны из молекулы, сохраняя индексы тяжёлых атомов.
    
    Параметры:
        mol (Chem.Mol): исходная молекула с H.
    
    Возвращает:
        mol_noH (Chem.Mol): молекула без H.
        heavy_idx_map (dict): словарь {новый индекс тяжелого атома: исходный индекс в mol}
    """
    mol_rw = Chem.RWMol(mol)  # Создаём RWMol для редактирования

    # Запоминаем исходный индекс каждого атома до удаления H
    for atom in mol_rw.GetAtoms():
        atom.SetIntProp('_orig_idx', atom.GetIdx())

    # Проходим по атомам и собираем индексы протонов
    H_indices = [atom.GetIdx() for atom in mol_rw.GetAtoms() if atom.GetAtomicNum() == 1]

    # Переносим удаляемые H в счётчик явных H соседа, чтобы сохранить валентность
    # (иначе NH2 мономера и NH полимера получат разную валентность при MatchValences)
    for idx in H_indices:
        for neighbor in mol_rw.GetAtomWithIdx(idx).GetNeighbors():
            if neighbor.GetAtomicNum() != 1:
                neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)

    # Удаляем протонные атомы (любые индексы)
    for idx in sorted(H_indices, reverse=True):
        mol_rw.RemoveAtom(idx)

    mol_noH = mol_rw.GetMol()
    mol_noH.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol_noH)  # RemoveAtom сбрасывает информацию о кольцах

    # Словарь соответствия новых индексов → оригинальные
    heavy_idx_map = {}
    for atom in mol_noH.GetAtoms():
        heavy_idx_map[atom.GetIdx()] = atom.GetIntProp('_orig_idx')
        atom.ClearProp('_orig_idx')

    # Возвращаем молекулу и mapping
    return mol_noH, heavy_idx_map


def match_chem_v3(mol_chem_1, mol_chem_2,
                  compare_any_bond=False,
                  match_residue_number=None,
                  match_type='heavy',  # 'heavy' или 'all'
                  timeout=3):
    """
    Находит максимальную общую подструктуру между мономером и полимером.
    
    Args:
        mol_chem_1 (Chem.Mol) - мономер
        mol_chem_2 (Chem.Mol) - полимер
        compare_any_bond (bool) - сравнивать любые связи
        match_residue_number (int) - номер остатка для фильтрации
        match_type (str) - 'heavy' (только тяжелые атомы) или 'all' (все атомы)
        timeout (int) - таймаут поиска MCS
    
    Returns:
        substructure (Chem.Mol) - подструктура с протонами
        dict_matches (dict) - соответствие индексов и имен атомов
    """
    # --- 1. Предупреждение для match_type='all' ---
    if match_type == 'all':
        print("ВНИМАНИЕ: Сопоставление по всем атомам (включая водороды)")
        print("Это может занять очень много времени для больших молекул!")
    
    # --- 2. Фильтрация по номеру остатка ---
    def filter_atoms_by_residue_number(mol, residue_number):
        """Возвращает новую молекулу, содержащую только атомы с заданным номером остатка"""
        indices_to_keep = []
        for atom in mol.GetAtoms():
            monomer_info = atom.GetMonomerInfo()
            if monomer_info and monomer_info.GetResidueNumber() == residue_number:
                indices_to_keep.append(atom.GetIdx())
        
        if not indices_to_keep:
            print(f"Предупреждение: Остаток с номером {residue_number} не найден")
            return mol
        
        print(f"Фильтрация по остатку {residue_number}: оставлено {len(indices_to_keep)} атомов")
        return Chem.PathToSubmol(mol, indices_to_keep)
    
    if match_residue_number is not None:
        print(f'Поиск MCS только для остатка с номером {match_residue_number}')
        filtered_mol_chem_2 = filter_atoms_by_residue_number(mol_chem_2, match_residue_number)
    else:
        filtered_mol_chem_2 = mol_chem_2
    
    # --- 3. Подготовка молекул в зависимости от match_type ---
    if match_type == 'heavy':
        print("Сопоставление по тяжелым атомам (водороды исключены)")
        mol_1_processed = Chem.RemoveHs(mol_chem_1)
        mol_2_processed = Chem.RemoveHs(filtered_mol_chem_2)
    else:  # 'all'
        print("Сопоставление по всем атомам (включая водороды)")
        mol_1_processed = mol_chem_1
        mol_2_processed = filtered_mol_chem_2
    
    # --- 4. Настройка параметров MCS ---
    mcs_params = rdFMCS.MCSParameters()
    mcs_params.Timeout = timeout
    mcs_params.Maximize = True
    mcs_params.CompareAnyBond = compare_any_bond
    
    if match_type == 'heavy':
        mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElementsAndCharge
        mcs_params.MatchValences = False
    else:
        mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
        mcs_params.MatchValences = True
    
    mcs_params.BondCompare = rdFMCS.BondCompare.CompareOrder
    mcs_params.RingMatchesRingOnly = False
    mcs_params.CompleteRingsOnly = False
    
    # --- 5. Поиск MCS ---
    print("Поиск максимальной общей подструктуры...")
    mcs_result = rdFMCS.FindMCS([mol_1_processed, mol_2_processed], mcs_params)
    
    if not mcs_result or mcs_result.numAtoms == 0:
        raise ValueError("Общая подструктура не найдена!")
    
    print(f"Найдена общая подструктура из {mcs_result.numAtoms} атомов")
    
    # --- 6. Получение сопоставлений ---
    query = Chem.MolFromSmarts(mcs_result.smartsString)
    matches_1 = mol_1_processed.GetSubstructMatches(query, uniquify=False)
    matches_2 = mol_2_processed.GetSubstructMatches(query, uniquify=False)
    
    # --- 7. Выбор лучшего сопоставления (максимум атомов) ---
    # best_match_1 = None
    # best_match_2 = None
    max_size = 0
    
    for match_1 in matches_1:
        for match_2 in matches_2:
            size = len(match_1)  # количество сопоставленных атомов
            if size > max_size:
                max_size = size
                best_match_1 = match_1
                best_match_2 = match_2
    
    if best_match_1 is None:
        best_match_1 = matches_1[0]
        best_match_2 = matches_2[0]
    
    print(f"Выбрано сопоставление с {max_size} атомами")
    
    # --- 8. Создание словаря соответствий ---
    dict_matches = {}
    
    # Сохраняем индексы (без сортировки!)
    dict_matches['Indexes'] = dict(zip(best_match_1, best_match_2))
    
    # Добавляем имена атомов, если они есть
    if mol_chem_1.HasProp('AtomNames') and mol_chem_2.HasProp('AtomNames'):
        try:
            mol1_atoms_name = [mol_chem_1.GetAtomWithIdx(idx).GetProp('AtomName') 
                               for idx in best_match_1]
            mol2_atoms_name = [mol_chem_2.GetAtomWithIdx(idx).GetProp('AtomName') 
                               for idx in best_match_2]
            dict_matches['Names'] = dict(zip(mol1_atoms_name, mol2_atoms_name))
        except Exception as e:
            print(f"Предупреждение: Не удалось получить имена атомов - {e}")
    
    # --- 9. Построение подструктуры с протонами ---
    # Берем атомы из исходной молекулы-полимера (с водородами)
    # Используем best_match_2 как есть, без сортировки
    
    # Добавляем водороды, связанные с сопоставленными атомами
    all_indices = list(best_match_2)  # копируем список
    
    for idx in best_match_2:
        atom = mol_chem_2.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # Водород
                if neighbor.GetIdx() not in all_indices:
                    all_indices.append(neighbor.GetIdx())
    
    # Не сортируем индексы, сохраняем порядок
    substructure = Chem.PathToSubmol(mol_chem_2, all_indices)
    
    # Заряды копируются автоматически через PathToSubmol
    # Дополнительно проверяем и восстанавливаем если нужно
    for i, old_idx in enumerate(all_indices):
        old_atom = mol_chem_2.GetAtomWithIdx(old_idx)
        new_atom = substructure.GetAtomWithIdx(i)
        if old_atom.GetFormalCharge() != 0:
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
    
    return substructure, dict_matches

@logged
def match_chem_v6(
    mol_chem_1, mol_chem_2,
    only_heavy_mapping=True,
    compare_any_bond=False,
    match_residue_number=None,
    timeout=30
):
    """
    Сопоставление двух молекул с использованием RDKit MCS.
    Возвращает подструктуры с H и без H, а также словарь индексов и имен.

    Args:
        mol_chem_1 (Chem.Mol) - мономер (из его атомов строится подструктура)
        mol_chem_2 (Chem.Mol) - полимер
        only_heavy_mapping (bool) - MCS только по тяжелым атомам, H сопоставляются потом
        compare_any_bond (bool) - игнорировать порядок связей
        match_residue_number (int) - искать только среди атомов этого остатка полимера
        timeout (int) - таймаут поиска MCS, с

    Returns:
        substructure_dict (dict) - {"H": подструктура с H, "noH": подструктура без H}
        dict_matches (dict) - {"Indexes": {индекс в mol_chem_1: индекс в mol_chem_2},
                               "Names": {имя в mol_chem_1: имя в mol_chem_2}} (если есть имена)
        Атом i подструктуры "H" соответствует i-й паре в "Indexes".
        Если сопоставление не найдено, возвращает (None, None).
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    # === 1. Фильтрация по остаткам ===
    residue_index_map = None  # индекс в подмолекуле остатка -> индекс в mol_chem_2
    mol2_for_mcs = mol_chem_2

    if match_residue_number is not None:
        residue_atom_indices = [
            atom.GetIdx()
            for atom in mol_chem_2.GetAtoms()
            if atom.GetMonomerInfo() is not None
            and atom.GetMonomerInfo().GetResidueNumber() == match_residue_number
        ]

        if not residue_atom_indices:
            warnings.warn("⚠ Указанный residue не найден в молекуле.")
            return None, None

        bond_indices = []
        for bond in mol_chem_2.GetBonds():
            if bond.GetBeginAtomIdx() in residue_atom_indices and bond.GetEndAtomIdx() in residue_atom_indices:
                bond_indices.append(bond.GetIdx())

        atom_map = {}  # заполняется RDKit: индекс в mol_chem_2 -> индекс в подмолекуле
        residue_submol = Chem.PathToSubmol(mol_chem_2, bond_indices, useQuery=False, atomMap=atom_map)
        residue_submol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(residue_submol)

        # mapping submol_idx -> polymer_idx
        residue_index_map = {new_idx: old_idx for old_idx, new_idx in atom_map.items()}
        mol2_for_mcs = residue_submol

    # === 2. Heavy mapping: удаляем H для поиска MCS ===
    if only_heavy_mapping:
        mol1_work, mol1_map = remove_hydrogens_preserve_indices(mol_chem_1)
        mol2_work, mol2_map = remove_hydrogens_preserve_indices(mol2_for_mcs)
    else:
        warnings.warn("⚠ MCS с явными протонами может быть очень медленным.")
        mol1_work = mol_chem_1
        mol2_work = mol2_for_mcs
        mol1_map = {atom.GetIdx(): atom.GetIdx() for atom in mol1_work.GetAtoms()}
        mol2_map = {atom.GetIdx(): atom.GetIdx() for atom in mol2_work.GetAtoms()}

    # индекс в mol2_work -> индекс в исходном mol_chem_2 (с учетом фильтрации по остатку)
    if residue_index_map is not None:
        mol2_map = {i: residue_index_map[j] for i, j in mol2_map.items()}

    # === 3. Настройки MCS ===
    params = rdFMCS.MCSParameters()
    params.Timeout = timeout
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.AtomCompareParameters.MatchValences = True
    params.AtomCompareParameters.MatchFormalCharge = True
    params.AtomCompareParameters.RingMatchesRingOnly = True
    params.BondTyper = rdFMCS.BondCompare.CompareAny if compare_any_bond else rdFMCS.BondCompare.CompareOrder
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = False

    # === 4. Анимация поиска MCS ===
    done = False
    t = threading.Thread(target=animate, args=(lambda: done, "Поиск подструктуры между молекулами"))
    t.start()
    try:
        res = rdFMCS.FindMCS([mol1_work, mol2_work], params)
    finally:
        done = True
        t.join()

    if res.canceled:
        print_red("⚠ MCS остановлен по timeout, подструктура может быть неполной")

    if res.numAtoms == 0:
        warnings.warn("⚠ Общая подструктура не найдена.")
        return None, None

    queryMol = res.queryMol

    # === 5. Получаем сопоставления ===
    # queryMol не хранит валентность и заряд, поэтому проверяем их сами
    def atoms_compatible(a1, a2):
        return (a1.GetTotalValence() == a2.GetTotalValence()
                and a1.GetFormalCharge() == a2.GetFormalCharge())

    matches1 = mol1_work.GetSubstructMatches(queryMol, uniquify=False, maxMatches=1000)
    matches2 = mol2_work.GetSubstructMatches(queryMol, uniquify=False, maxMatches=1000)

    if not matches1 or not matches2:
        warnings.warn("⚠ Сопоставление подструктур не найдено.")
        return None, None

    # Для V6 убираем скоринг, берем первую пару, совместимую по валентности и заряду
    match1 = matches1[0]
    match2 = next(
        (m2 for m2 in matches2
         if all(atoms_compatible(mol1_work.GetAtomWithIdx(i1), mol2_work.GetAtomWithIdx(i2))
                for i1, i2 in zip(match1, m2))),
        None
    )
    if match2 is None:
        print_red(f"⚠ Ни одно из {len(matches2)} сопоставлений не совпадает по валентности "
                  "и заряду атомов, взято первое. Проверьте имена атомов в результате.")
        match2 = matches2[0]

    # индексы переводим в исходные mol_chem_1 / mol_chem_2
    mapping = {mol1_map[i1]: mol2_map[i2] for i1, i2 in zip(match1, match2)}

    # === 6. Восстанавливаем H ===
    if only_heavy_mapping:
        full_mapping = mapping.copy()
        for idx1, idx2 in mapping.items():
            atom1 = mol_chem_1.GetAtomWithIdx(idx1)
            atom2 = mol_chem_2.GetAtomWithIdx(idx2)

            H1 = sorted([n.GetIdx() for n in atom1.GetNeighbors() if n.GetAtomicNum() == 1])
            H2 = sorted([n.GetIdx() for n in atom2.GetNeighbors() if n.GetAtomicNum() == 1])

            for h1, h2 in zip(H1, H2):
                full_mapping[h1] = h2
    else:
        full_mapping = mapping

    # Порядок: по возрастанию индекса в mol_chem_1 — такой же, как у атомов подструктуры
    full_mapping = dict(sorted(full_mapping.items()))

    # === 7. Строим подструктуры ===
    # Удаляем из копии мономера несопоставленные атомы: оставшиеся атомы
    # сохраняют исходный порядок, свойства (AtomName, PDB info), заряды и координаты
    keep = set(full_mapping.keys())
    substructure_rw = Chem.RWMol(mol_chem_1)
    for idx in sorted((a.GetIdx() for a in mol_chem_1.GetAtoms() if a.GetIdx() not in keep), reverse=True):
        substructure_rw.RemoveAtom(idx)
    substructure_H = substructure_rw.GetMol()
    substructure_H.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(substructure_H)

    substructure_no_H, _ = remove_hydrogens_preserve_indices(substructure_H)

    substructure_dict = {"H": substructure_H, "noH": substructure_no_H}

    # === 8. Словарь соответствий ===
    dict_matches = {"Indexes": full_mapping}

    if mol_chem_1.HasProp('AtomNames') and mol_chem_2.HasProp('AtomNames'):
        name_mapping = {}
        for i1, i2 in full_mapping.items():
            name1 = mol_chem_1.GetAtomWithIdx(i1).GetProp('AtomName')
            name2 = mol_chem_2.GetAtomWithIdx(i2).GetProp('AtomName')
            name_mapping[name1] = name2
        dict_matches['Names'] = name_mapping

    return substructure_dict, dict_matches
    
class _MatchData(dict):
    """
    Словарь результатов match_mon_to_pol. Старые имена ключей из ранних ноутбуков
    ('substructure_noH') перенаправляются на новые, не дублируясь в словаре.
    """
    _OLD_KEYS = {'substructure_noH': 'substructure_no_H'}

    def __missing__(self, key):
        if key in self._OLD_KEYS:
            return self[self._OLD_KEYS[key]]
        raise KeyError(key)


@logged
def match_mon_to_pol(monomer_dict, polymer_dict, compare_any_bond = False,
                     match_residue_number = None, monomer_key = True, timeout = 3,
                     only_heavy_mapping = True):
    """
    Аргументы:
        monomer_dict - словарь мономера (состоит из одной пары имя: rdkit.Chem)
        polymer_dict - словарь полимера (состоит из одной пар имя: rdkit.Chem)
        only_heavy_mapping - MCS только по тяжелым атомам (H сопоставляются после)
    Возвращает:
        Словарь match_data_dict со следующими подсловарями:
        - substructure - сожердит общие подструктуры мономера и полимера (с H), для каждого из полимеров
        - substructure_no_H - те же подструктуры без H
        - mon_pol_matches - содержит словари соответствия номеров атомов в общей подструктуре
        с номерами атомов в полимере
        Пары, для которых сопоставление не найдено, пропускаются.
    """
    def add_match(key, mon_val, pol_val):
        sub, dict_mon_pol_matches = match_chem_v6(mon_val, pol_val,
                                                  only_heavy_mapping = only_heavy_mapping,
                                                  compare_any_bond = compare_any_bond,
                                                  match_residue_number = match_residue_number,
                                                  timeout=timeout
                                                  )
        if sub is None:
            print_red(f'Сопоставление для {key} не найдено, пропускаем.')
            return

        match_data_dict['substructure'][key] = sub['H']
        match_data_dict['substructure_no_H'][key] = sub['noH']
        match_data_dict['N_match_atoms'][key] = sub['H'].GetNumAtoms()
        match_data_dict['mon_pol_matches'][key] = dict_mon_pol_matches['Indexes']
        match_data_dict['mon_pol_atom_names'][key] = dict_mon_pol_matches.get('Names')

    try:
        match_data_dict = _MatchData({'substructure': {},
                           'substructure_no_H': {},
                           'mon_pol_matches': {},
                           'N_match_atoms': {},
                           'mon_pol_atom_names':{}})

        if (len(monomer_dict) >= 1 and len(polymer_dict) == 1) and monomer_key is True:

            pol_key, pol_val = next(iter(polymer_dict.items()))

            for mon_key, mon_val in monomer_dict.items():
                add_match(mon_key, mon_val, pol_val)

        elif len(monomer_dict) == 1 and len(polymer_dict) > 1 or monomer_key is False:
            mon_key, mon_val = next(iter(monomer_dict.items()))

            for pol_key, pol_val in polymer_dict.items():
                add_match(pol_key, mon_val, pol_val)

        else:
            print('The len of monomer_dict or polymer_dict must be 1.')
            
        return match_data_dict
    except Exception as e:
        raise e
        # print_red(f'Что-то пошло не так!\n{e}')
        
# Наборы референсных остатков для поиска канонических атомов.
# Шаблон - PDB-файл, где референсный остаток стоит в позиции residue_number
# (для белков - тримеры GXG_H.pdb), порядок его атомов в файле считается каноническим.
# Для нуклеиновых кислот достаточно добавить сюда запись со своей папкой шаблонов.
REF_TEMPLATES = {
    'protein': {
        'dir': os.path.join(_PACKAGE_ROOT, 'molecules', 'aminoacids_template'),
        'pattern': '*_H.pdb',
        'residue_number': 2,
        # атомы остова: если они не сопоставились, референс выбран неверно
        'backbone': ['N', 'CA', 'C', 'O'],
    },
}

# Канонические аминокислоты: однобуквенный код -> (трехбуквенный код, название, name)
AMINO_ACIDS = {
    'A': ('Ala', 'аланин', 'alanine'),
    'C': ('Cys', 'цистеин', 'cysteine'),
    'D': ('Asp', 'аспарагиновая кислота', 'aspartate'),
    'E': ('Glu', 'глутаминовая кислота', 'glutamate'),
    'F': ('Phe', 'фенилаланин', 'phenylalanine'),
    'G': ('Gly', 'глицин', 'glycine'),
    'H': ('His', 'гистидин', 'histidine'),
    'I': ('Ile', 'изолейцин', 'isoleucine'),
    'K': ('Lys', 'лизин', 'lysine'),
    'L': ('Leu', 'лейцин', 'leucine'),
    'M': ('Met', 'метионин', 'methionine'),
    'N': ('Asn', 'аспарагин', 'asparagine'),
    'P': ('Pro', 'пролин', 'proline'),
    'Q': ('Gln', 'глутамин', 'glutamine'),
    'R': ('Arg', 'аргинин', 'arginine'),
    'S': ('Ser', 'серин', 'serine'),
    'T': ('Thr', 'треонин', 'threonine'),
    'V': ('Val', 'валин', 'valine'),
    'W': ('Trp', 'триптофан', 'tryptophan'),
    'Y': ('Tyr', 'тирозин', 'tyrosine'),
}
aa_full_names = {one: names[2] for one, names in AMINO_ACIDS.items()}


def aa_label(letter):
    """'C' -> 'цистеин (Cys, C)'. Для неизвестного кода возвращает сам код."""
    if letter not in AMINO_ACIDS:
        return letter
    three, name_ru, _ = AMINO_ACIDS[letter]
    return f'{name_ru} ({three}, {letter})'


def template_aa_letter(template_name):
    """Однобуквенный код остатка шаблона: 'GCG_H' -> 'C', 'AGA_H' -> 'G'."""
    core = template_name.split('_')[0]
    return core[1] if len(core) == 3 and core[1] in AMINO_ACIDS else None


def template_label(template_name):
    """'GCG_H' -> 'цистеин (Cys, C), шаблон GCG_H'."""
    letter = template_aa_letter(template_name)
    return f'{aa_label(letter)}, шаблон {template_name}' if letter else template_name


def resolve_ref_template(ref_base_name, template_paths, residue_type='protein'):
    """
    Находит файл шаблона по имени референсного остатка.

    Аргументы:
        ref_base_name (str) - имя остатка: однобуквенное ('C'), трехбуквенное ('Cys'),
            полное ('cysteine') или имя файла шаблона ('GCG_H')
        template_paths (list) - пути к файлам шаблонов
        residue_type (str) - тип остатка из REF_TEMPLATES
    Возвращает:
        str - путь к файлу шаблона
    """
    by_name = {os.path.basename(path).split('.')[0].upper(): path for path in template_paths}
    key = ref_base_name.strip().upper()
    if key in by_name:
        return by_name[key]

    if residue_type == 'protein':
        letter = None
        if len(key) == 1 and key in AMINO_ACIDS:
            letter = key
        else:
            for one, names in AMINO_ACIDS.items():
                if key in (name.upper() for name in names):
                    letter = one
                    break
        if letter is not None and f'G{letter}G_H' in by_name:
            return by_name[f'G{letter}G_H']
    else:
        # Для остальных типов ищем шаблон, в имени которого есть ref_base_name
        found = [path for name, path in by_name.items() if key in name]
        if len(found) == 1:
            return found[0]

    raise ValueError(f'Шаблон для остатка "{ref_base_name}" не найден среди: {sorted(by_name)}')


@logged
def check_parent_residue(ref_mol, mapping, residue_number, backbone, template_name, strict=True):
    """
    Проверяет, что модификация действительно построена на заявленном родительском остатке.

    Остов (backbone) и CB должны быть сопоставлены полностью, иначе ValueError
    (при strict=False - предупреждение).
    Несопоставленные атомы боковой цепи печатаются: обычно это место модификации
    (NZ ацетиллизина, NH2 цитруллина - один атом). Если не сопоставлена вся боковая цепь
    или не меньше 2 атомов и половины цепи, родитель скорее всего указан неверно:
    при strict=True это ValueError, при strict=False - предупреждение
    (для модификаций, сильно перестраивающих боковую цепь, например кинуренина из Trp).

    Ограничение: Ala и Gly содержатся почти в любом остатке, поэтому ошибочно
    заявленный Ala/Gly проверкой не ловится.

    Аргументы:
        ref_mol (Chem.Mol) - шаблон
        mapping (dict) - {индекс в модификации: индекс в шаблоне}
        residue_number (int) - номер родительского остатка в шаблоне
        backbone (list) - имена атомов остова
        template_name (str) - имя шаблона для сообщений
    """
    residue_atoms = {atom.GetIdx(): atom.GetProp('AtomName') for atom in ref_mol.GetAtoms()
                     if atom.GetAtomicNum() > 1
                     and atom.GetPDBResidueInfo().GetResidueNumber() == residue_number}
    matched = set(mapping.values())
    required = list(backbone) + (['CB'] if 'CB' in residue_atoms.values() else [])

    missing_required = [name for idx, name in residue_atoms.items()
                        if name in required and idx not in matched]
    if missing_required:
        message = (f'Модификация не содержит остов остатка {template_label(template_name)}: '
                   f'не сопоставлены {missing_required}. Проверьте ref_base_name.')
        if strict:
            raise ValueError(message)
        print_red(f'⚠ {message}')

    side_chain = [name for name in residue_atoms.values() if name not in required]
    missing_side = [name for idx, name in residue_atoms.items()
                    if name in side_chain and idx not in matched]
    if not missing_side:
        return
    report = f'атомы боковой цепи {missing_side} ({len(missing_side)} из {len(side_chain)})'
    wrong_parent = (len(missing_side) == len(side_chain)
                    or (len(missing_side) >= 2 and 2 * len(missing_side) >= len(side_chain)))
    if wrong_parent and strict:
        raise ValueError(
            f'Не сопоставлены {report} остатка {template_label(template_name)}: '
            'похоже, родительский остаток указан неверно. Проверьте ref_base_name; '
            'если модификация сильно перестраивает боковую цепь, вызовите с strict_parent=False.')
    if wrong_parent:
        print_red(f'⚠ Не сопоставлены {report}: возможно, родительский остаток указан неверно.')
    else:
        print_red(f'⚠ Не сопоставлены {report}: вероятно, это место модификации.')


@logged
def find_ref_residue(mod_mol_dict, path_to_ref_mol=None, ref_base_name=None,
                     residue_type='protein', match_residue_number=None,
                     only_heavy_mapping=True, main_match_data=True, timeout=3,
                     strict_parent=True, type_match_dict=None):
    """
    Сопоставляет модифицированный остаток с референсными остатками и выбирает референс.

    Аргументы:
        mod_mol_dict (dict) - словарь {имя: Chem.Mol} модифицированного остатка (одна пара)
        path_to_ref_mol (str) - папка с шаблонами или путь к одному шаблону;
            по умолчанию папка из REF_TEMPLATES[residue_type]
        ref_base_name (str) - родительский остаток, из которого собрана модификация:
            'C', 'Cys', 'cysteine', 'цистеин' или имя шаблона 'GCG_H' (см. resolve_ref_template).
            Нужно указывать всегда: по числу совпавших атомов родителя не определить
            (у меток цепи и кольца совпадают с Lys/Trp больше, чем с Cys).
            Старые ноутбуки вызывают функцию без него: тогда шаблон выбирается автоматически,
            как раньше, а все проверки родителя выдают предупреждения вместо ошибок.
        residue_type (str) - тип остатка из REF_TEMPLATES ('protein', ...)
        match_residue_number (int) - номер референсного остатка в шаблоне;
            по умолчанию REF_TEMPLATES[residue_type]['residue_number']
        only_heavy_mapping (bool) - MCS только по тяжелым атомам (быстрее)
        main_match_data (bool) - вернуть данные только для выбранного шаблона
        timeout (int) - таймаут поиска MCS для каждого шаблона, с
        strict_parent (bool) - ошибка, если боковая цепь родителя в основном не сопоставлена
            (см. check_parent_residue); False - только предупреждение
        type_match_dict - устаревший параметр старых ноутбуков, ни на что не влияет
    Возвращает:
        ref_chem_dict (dict) - {имя шаблона: Chem.Mol}
        match_data_dict (dict) - словарь match_mon_to_pol, ключи - имена шаблонов
        ref_name (str) - имя выбранного шаблона-триплета, например 'GKG_H'
    """
    template = REF_TEMPLATES[residue_type]
    if match_residue_number is None:
        match_residue_number = template['residue_number']

    path_to_ref_mol = path_to_ref_mol or template['dir']
    if os.path.isdir(path_to_ref_mol):
        template_paths = sorted(glob.glob(os.path.join(path_to_ref_mol, template['pattern'])))
    else:
        template_paths = [path_to_ref_mol]
    if not template_paths:
        raise FileNotFoundError(f'Шаблоны не найдены в {path_to_ref_mol}')

    if type_match_dict is not None:
        print_red('⚠ Параметр type_match_dict устарел и ни на что не влияет: '
                  'имена атомов всегда возвращаются в match_data_dict["mon_pol_atom_names"].')

    if ref_base_name:
        template_paths = [resolve_ref_template(ref_base_name, template_paths, residue_type)]
    elif len(template_paths) > 1:
        # Старый вызов без родителя: работаем как раньше, но проверки только предупреждают
        strict_parent = False
        print_red('⚠ Родительский остаток не указан (ref_base_name): шаблон будет выбран '
                  'по числу совпавших атомов, как в старых версиях. Результат ненадёжен: '
                  'для меток на цистеине так выбирается Lys или Trp. '
                  "Укажите родителя, например ref_base_name='C' или 'K'.")

    ref_chem_dict = pdb_to_chem(template_paths)
    match_data_dict = match_mon_to_pol(mod_mol_dict, ref_chem_dict,
                                       match_residue_number=match_residue_number,
                                       monomer_key=False, timeout=timeout,
                                       only_heavy_mapping=only_heavy_mapping)
    if not match_data_dict['N_match_atoms']:
        raise ValueError('Не найдено сопоставление ни с одним из шаблонов.')

    max_name = max(match_data_dict['N_match_atoms'], key=match_data_dict['N_match_atoms'].get)
    max_val = match_data_dict['N_match_atoms'][max_name]
    print(f'Родительский остаток: {template_label(max_name)}; сопоставлено атомов: {max_val}.')

    check_parent_residue(ref_chem_dict[max_name], match_data_dict['mon_pol_matches'][max_name],
                         match_residue_number, template.get('backbone', []), max_name,
                         strict=strict_parent)

    if main_match_data:
        match_data_dict = {key: {max_name: val[max_name]} for key, val in match_data_dict.items()}
    return {max_name: ref_chem_dict[max_name]}, match_data_dict, max_name


# Старое имя, используется в ноутбуках
find_ref_aa = find_ref_residue

# ---------------------------------------------

aa_dict = {one: names[0] for one, names in AMINO_ACIDS.items()}


# Импорт из других модулей пакета - в конце файла: функции этого модуля уже
# определены, поэтому взаимные ссылки модулей друг на друга не мешают импорту.
from .fileio import pdb_to_chem  # noqa: E402
