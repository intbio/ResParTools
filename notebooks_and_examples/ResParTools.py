import os
import sys
import time
import re
import glob
import json
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
from typing import Callable


## Вспомогательные мини функции
def get_name(path: str, index=0):
    """
    Аргумент: 
        path - строка с путем к файлу
    Возвращает:
        Имя для ключа в словаре 
    """
    # Сохраняем текущий stderr
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        mol = Chem.MolFromSmiles(path)
        if mol:
            return index
        else:
            name_of_file = os.path.basename(path).split('.')[0]
            return name_of_file
    except Exception as e:
        print(f"Ошибка при обработке '{path}': {e}")
        return None
    finally:
        # Возвращаем stderr на исходное место
        sys.stderr = original_stderr
        
        
## Дектораторы
def data_to_dict(funсtion: Callable):
    '''
    Декоратор для обработки данных различного типа (строка, список, словарь) перед вызовом функции.
    '''
    def wrapper(data: str or list[str] or dict[str, str], **kwarg):
        if isinstance(data, str):
            return {get_name(data): funсtion(data, **kwarg)}
        elif isinstance(data, list):
            return {get_name(str_data, i):funсtion(str_data, **kwarg) for i, str_data in enumerate(data)}
        elif isinstance(data, dict):
            return {legend: funсtion(str_data, **kwarg) for legend, str_data in data.items()}
        else:
            print('Некорректный тип данных ввода', file=sys.stderr)
            return None
    return wrapper

def do_fun_for_2d_list(function: Callable):
    '''
    Декоратор для применения функции к каждому элементу двумерного списка (матрицы).
    '''
    def wrapper(lst: list):
        result = []
        for element in lst:
            if isinstance(element, list):
                # Рекурсивно применяем декоратор к вложенному списку
                result.append(wrapper(element))
            else:
                # Применяем функцию к элементу
                result.append(function(element))
        return result
    return wrapper


## 

@data_to_dict
def read_file(path):
    '''
    Читает первую строку из файла и возвращает её.
    '''
    with open (path, 'r') as f:
        return f.readline().strip()

@data_to_dict
def smi_to_chem(str_smi: str, sanitize=True, addH=True, make_N_root=False,format_coord='2D'):
    """
    Преобразует SMILES-строку в объект молекулы RDKit и добавляет атомы водорода.
    """
    rdkit_mol = Chem.MolFromSmiles(str_smi, sanitize) # переводим во внутренний формат chem
    if addH:
        rdkit_mol = Chem.AddHs(rdkit_mol) # протонируем
        
    if format_coord == '2D':
        # Рассчитываем 2D координаты
        AllChem.Compute2DCoords(rdkit_mol)
    elif format_coord == '3D':
        # Рассчитываем 3D координаты (если еще не рассчитаны)
        Chem.SanitizeMol(rdkit_mol)
        AllChem.EmbedMolecule(rdkit_mol)
        AllChem.UFFOptimizeMolecule(rdkit_mol)
    else:
        raise ValueError("Недопустимый формат. Выберите '2D' или '3D'.")
    # Перенумерация атомов, деля аминогруппу началом молекулы
    if make_N_root:
        # Находим индекс азота из аминогруппы
        root_idx = find_amino_nitrogen(rdkit_mol)
        if root_idx != -1:
            # Получаем SMILES с корневым атомом
            smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, allHsExplicit=True, rootedAtAtom=root_idx)
            # Создаем новую молекулу из SMILES
            rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
            if rdkit_mol is None:
                raise ValueError("Ошибка при создании молекулы после перенумерации атомов")
        else:
            print("Азот аминогруппы не найден. Корень не изменён.")
        # Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE, catchErrors=True)
        Chem.SanitizeMol(rdkit_mol)
        
    # Сортируем атомы, чтобы тяжелые атомы шли первыми, а водороды позже
        atom_order = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() != 'H']  # тяжелые атомы
        atom_order += [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == 'H']  # водороды
    
    # Перенумерация атомов согласно новому порядку
        new_rdkit_mol = Chem.RenumberAtoms(rdkit_mol, atom_order)
        # Пересчитываем валентности
        return new_rdkit_mol
    else:
        return rdkit_mol
    # if make_N_root:
    #     root_idx = find_amino_nitrogen(rdkit_mol)
    #     if root_idx != -1:
        #     smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, 
        #                                      allHsExplicit=True, rootedAtAtom=root_idx)
        #     rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=True)
        # return rdkit_mol


@data_to_dict
def pdb_to_chem(path_to_pdb, removeHs=False, make_N_root=False):
    """
    Открывает PDB файл при помощи RDKit, преобразует его в двухмерную молекулу 
    и сохраняет имена атомов как свойства RDKit-атомов.
    
    Аргументы:
        path_to_pdb (str): Путь к файлу PDB.
        removeHs (bool): Удалять ли гидрогены. По умолчанию False.
        make_N_root (bool): Делать ли атом азота корневым для молекулы. По умолчанию False.
        
    Возвращает:
        Chem.Mol: Молекула RDKit с сохраненными именами атомов.
    """
    # Загрузка молекулы из PDB
    rdkit_mol = Chem.MolFromPDBFile(path_to_pdb, removeHs)
    
    # smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=False, 
    #                                      allHsExplicit=True)
    # rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
    
    if not rdkit_mol:
        raise ValueError(f"Не удалось загрузить PDB файл: {path_to_pdb}")
      
    # Открываем PDB файл для извлечения имен атомов
    with open(path_to_pdb, "r") as pdb_file:
        pdb_lines = pdb_file.readlines()

    atom_names = []
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            atom_name = line.split()[2]  # Имя атома в формате PDB
            atom_names.append(atom_name)

    # Проверка соответствия числа атомов
    # if len(atom_names) != rdkit_mol.GetNumAtoms():
    #     raise ValueError("Число атомов в PDB файле и RDKit молекуле не совпадает!")

    # Присваиваем имена атомов
    for atom, name in zip(rdkit_mol.GetAtoms(), atom_names):
        atom.SetProp("AtomName", name)
        # atom.SetAtomMapNum(atom.GetIdx())
    rdkit_mol.SetProp('AtomNames', str(len(atom_names)))
    # Преобразование трехмерных координат в двумерные
    AllChem.Compute2DCoords(rdkit_mol)
    # rdDepictor.SetPreferCoordGen(True)
    return rdkit_mol

def file_opener(path, **kwargs):
    """
    Открывает файл с молекулой, автоматически определяя формат.
    
    Parameters:
    -----------
    path : str
        Путь к файлу
    **kwargs : dict
        Дополнительные параметры для функций загрузки
    """
    path_obj = Path(path)
    name = path_obj.stem
    extension = path_obj.suffix.lower()
    
    print(f"Загрузка файла: {name} (формат: {extension})")
    
    match extension:
        case ".smi" | ".smiles":
            smi_str = read_file(path)
            rdkit_mol = smi_to_chem(smi_str, **kwargs)
        case ".mol2":
            rdkit_mol = mol2_to_chem(path, **kwargs)
        case ".pdb":
            # Используем kwargs для переопределения параметров по умолчанию
            removeHs = kwargs.get('removeHs', False)
            format_coord = kwargs.get('format_coord', '3D')
            rdkit_mol = pdb_to_chem(path, removeHs=removeHs, format_coord=format_coord)
        case _:
            raise ValueError(f"Неподдерживаемый формат файла: {extension}. "
                           f"Поддерживаемые: .smi, .smiles, .mol2, .pdb")
    
    return rdkit_mol


def draw_mol_with_atom_index(mol, charge_list = None, size=(600,400), prefer_coord_gfen = False):
    """
    Добавляет номера атомов в атрибуты атомов молекулы.
    size=(600,400)
    """
    if charge_list is not None and len(charge_list) == len(mol.GetAtoms()):
        for i, atom in enumerate(mol.GetAtoms()):
            # atom.SetAtomMapNum(atom.GetIdx())
            # atom.SetDoubleProp('PartialCharge', charge_list[i])
            atom.SetProp('atomNote',str("%4.4f" % charge_list[i]))
    else:   
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(atom.GetIdx())
    
    if prefer_coord_gfen:
        rdDepictor.SetPreferCoordGen(True)
        if mol is not None:
            mol.RemoveAllConformers()  # Удалить старые координаты
            rdDepictor.Compute2DCoords(mol)
    else:
        if mol is not None:
            AllChem.Compute2DCoords(mol)
        
    return Draw.MolToImage(mol, size)   


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
    heavy_idx_map = {}        # Словарь соответствия новых индексов → оригинальные

    # Проходим по атомам и собираем индексы протонов
    H_indices = [atom.GetIdx() for atom in mol_rw.GetAtoms() if atom.GetAtomicNum() == 1]

    # Удаляем протонные атомы (любые индексы)
    for idx in sorted(H_indices, reverse=True):
        mol_rw.RemoveAtom(idx)

    # После удаления H формируем mapping для тяжёлых атомов
    new_idx = 0
    for atom in mol_rw.GetAtoms():
        heavy_idx_map[new_idx] = atom.GetIdx()
        new_idx += 1

    # Возвращаем молекулу и mapping
    return mol_rw.GetMol(), heavy_idx_map


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
    """
    import threading
    import warnings
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    # === 1. Фильтрация по остаткам ===
    residue_index_map = None
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

        residue_submol = Chem.PathToSubmol(mol_chem_2, bond_indices, useQuery=False)

        # mapping submol_idx -> polymer_idx
        # residue_index_map = {new_idx: old_idx for new_idx, old_idx in enumerate(sorted(residue_atom_indices))}
        mol2_for_mcs = residue_submol

    # === 2. Heavy mapping: удаляем H для поиска MCS ===
    if only_heavy_mapping:
        mol1_work, mol1_map = remove_hydrogens_preserve_indices(mol_chem_1)
        mol2_work, mol2_map = remove_hydrogens_preserve_indices(mol2_for_mcs)
    else:
        warnings.warn("⚠ MCS с явными протонами может быть очень медленным.")
        mol1_work = mol_chem_1
        mol2_work = mol2_for_mcs

    # === 3. Настройки MCS ===
    params = rdFMCS.MCSParameters()
    params.Timeout = timeout
    params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    params.MatchValences = True
    params.MatchFormalCharge = True
    # params.MatchHydrogens = False
    params.BondCompare = rdFMCS.BondCompare.CompareAny if compare_any_bond else rdFMCS.BondCompare.CompareOrder
    params.RingMatchesRingOnly = True
    params.CompleteRingsOnly = False

    # === 4. Анимация поиска MCS ===
    done = False
    t = threading.Thread(target=animate, args=(lambda: done, "Поиск подструктуры между молекулами"))
    t.start()

    res = rdFMCS.FindMCS([mol1_work, mol2_work], parameters=params)

    done = True
    t.join()

    if res.canceled:
        print_red("⚠ MCS остановлен по timeout")

    if res.numAtoms == 0:
        warnings.warn("⚠ Общая подструктура не найдена.")
        return None, None

    queryMol = res.queryMol

    # === 5. Получаем ВСЕ сопоставления ===
    matches1 = mol1_work.GetSubstructMatches(queryMol)
    matches2 = mol2_work.GetSubstructMatches(queryMol)

    if not matches1 or not matches2:
        warnings.warn("⚠ Сопоставление подструктур не найдено.")
        return None, None

    # Для V6 убираем скоринг, просто берем первую пару
    match1 = matches1[0]
    match2 = matches2[0]

    full_mapping = {mol1_map[i1]: mol2_map[i2] for i1, i2 in zip(match1, match2)}
    # if residue_index_map is not None:
    #     match2 = tuple(residue_index_map[i] for i in match2)

    # mapping = dict(zip(match1, match2))

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

    # === 7. Строим подструктуры ===
    atom_indices_H = sorted(full_mapping.keys())
    bond_indices_H = [
        b.GetIdx() for b in mol_chem_1.GetBonds()
        if b.GetBeginAtomIdx() in atom_indices_H and b.GetEndAtomIdx() in atom_indices_H
    ]

    substructure_H = Chem.PathToSubmol(mol_chem_1, bond_indices_H, useQuery=False)
    substructure_noH = remove_hydrogens_preserve_indices(substructure_H)

    substructure_dict = {"H": substructure_H, "noH": substructure_noH}

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
    
def match_mon_to_pol(monomer_dict, polymer_dict, compare_any_bond = False, 
                     match_residue_number = None, monomer_key = True, timeout = 3):
    """
    Аргументы:
        monomer_dict - словарь мономера (состоит из одной пары имя: rdkit.Chem)
        polymer_dict - словарь полимера (состоит из одной пар имя: rdkit.Chem)
    Возвращает:
        Словарь match_data_dict со следующими подсловарями:
        - substructure - сожердит общие подструктуры мономера и полимера, для каждого из полимеров
        - mon_pol_matches - содержит словари соответствия номеров атомов в общей подструктуре 
        с номерами атомов в полимере 
    """
    try:
        match_data_dict = {'substructure': {}, 
                           'mon_pol_matches': {},
                           'N_match_atoms': {},
                           'mon_pol_atom_names':{}}
        
        if (len(monomer_dict) >= 1 and len(polymer_dict) == 1) and monomer_key is True:

            pol_key, pol_val = next(iter(polymer_dict.items()))

            for mon_key, mon_val in monomer_dict.items():
                sub, dict_mon_pol_matches = match_chem_v6(mon_val, pol_val, compare_any_bond = compare_any_bond,
                                                       match_residue_number = match_residue_number,
                                                        timeout=timeout
                                                        )

                match_data_dict['substructure'][mon_key] = sub
                match_data_dict['N_match_atoms'][mon_key] = sub.GetNumAtoms()
                match_data_dict['mon_pol_matches'][mon_key] = dict_mon_pol_matches['Indexes']
                match_data_dict['mon_pol_atom_names'][mon_key] = dict_mon_pol_matches.get('Names')

        elif len(monomer_dict) == 1 and len(polymer_dict) > 1 or monomer_key is False:
            mon_key, mon_val = next(iter(monomer_dict.items()))

            for pol_key, pol_val in polymer_dict.items():
                sub, dict_mon_pol_matches = match_chem_v6(mon_val, pol_val, compare_any_bond = compare_any_bond,
                                                       match_residue_number = match_residue_number,
                                                        timeout=timeout
                                                        )

                match_data_dict['substructure'][pol_key] = sub
                match_data_dict['N_match_atoms'][pol_key] = sub.GetNumAtoms()
                match_data_dict['mon_pol_matches'][pol_key] = dict_mon_pol_matches['Indexes']
                match_data_dict['mon_pol_atom_names'][pol_key] = dict_mon_pol_matches.get('Names')

        else:
            print('The len of monomer_dict or polymer_dict must be 1.')
            
        return match_data_dict
    except Exception as e:
        raise e
        # print_red(f'Что-то пошло не так!\n{e}')
        
def find_ref_aa(mod_mol_dict, path_to_ref_mol=None, match_residue_number = None, main_match_data=True):

    if path_to_ref_mol:
        ref_aa_chem_dict = pdb_to_chem(path_to_ref_mol, )
    else:
        ref_aa_chem_dict = pdb_to_chem(glob.glob('moleculse/aminoacids_template/*_H.pdb')) 
    match_data_dict = match_mon_to_pol(mod_mol_dict, ref_aa_chem_dict, 
                                       match_residue_number = match_residue_number)

    max_name, max_val = None, -1
    # Находим максимальное совпадение
    max_name = max(match_data_dict['N_match_atoms'], 
                  key=match_data_dict['N_match_atoms'].get)
    max_val = match_data_dict['N_match_atoms'][max_name]
    
    # Формируем выходные данные
    exit_data_dict = {
        'substructure': {max_name: match_data_dict['substructure'][max_name]},
        'N_match_atoms': {max_name: max_val},
        'mon_pol_matches': {max_name: match_data_dict['mon_pol_matches'][max_name]},
        'mon_pol_atom_names': {max_name: match_data_dict['mon_pol_atom_names'][max_name]}
    }

    aa_letter = max_name[1]  # Вторая буква из имени файла
    short_name = aa_dict[aa_letter.upper()]
    chem_ref_mol = pdb_to_chem(f'moleculse/aminoacids_template/{max_name}.pdb')
    
    print(f'Reference amino acid is {short_name}, with {max_val} matched atoms.')
    return chem_ref_mol, exit_data_dict if main_match_data else match_data_dict , max_name


def draw_mon_pol_match(monomer_chem_dict, polymer_chem_dict={}, 
                       match_data = False, prefer_coord_gfen = False, 
                       add_atom_index = False, add_small_atom_index = False, show_any_monomer_matches = False, 
                       n_row = 1, useSVG = True, img_size = (500,300)):
    """
    Отображает сопоставление мономеров и полимеров в виде сетки изображений молекул.
    
    Аргументы:
        monomer_chem_dict (dict): Словарь мономеров.
        polymer_chem_dict (dict): Словарь полимеров.
        match_data (dict): Словарь данных сопоставления мономера с полимерами. По умолчанию False.
        add_atom_index (bool, optional): Добавлять ли номера атомов. По умолчанию False.
        add_small_atom_index (bool, optional): Использовать ли маленькие номера атомов. По умолчанию False.
        show_any_monomer_matches (bool, optional): Показать мономер с подструктурой для каждого полимера. По умолчанию False.
        n_row (int, optional): Количество молекул в строке. По умолчанию = кол-ву полимеров
        useSVG (bool, optional): Использовать ли SVG для изображения. По умолчанию True.
        img_size (tuple, optional): Размер изображения. По умолчанию (500, 300).
    
    Возвращает:
        Drawing: Изображение молекул в виде сетки.
    """
    n_mon, n_pol = len(monomer_chem_dict), len(polymer_chem_dict)
    if n_mon > 1 and n_pol > 1 and n_mon != n_pol:
        raise ValueError("Невозможно отобразить сопоставление для N:M (N,M > 1)")

    # Инициализация базовых структур
    mon_items = list(monomer_chem_dict.items())
    pol_items = list(polymer_chem_dict.items())
    
    
    chem_list = []
    legend_list = []
    highlight = []

    # Обработка случаев с сопоставлением
    if match_data:
        # 1:1 или N:N
        if n_mon == n_pol:
            pairs = zip(monomer_chem_dict.items(), polymer_chem_dict.items())
        # 1:N
        elif n_mon == 1:
            pairs = itertools.product(monomer_chem_dict.items(), polymer_chem_dict.items())
        # N:1
        elif n_pol == 1:
            pairs = itertools.product(monomer_chem_dict.items(), polymer_chem_dict.items())
        
        
        for (m_name, m_chem), (p_name, p_chem) in pairs:
            chem_list.extend([m_chem, p_chem])
            legend_list.extend([f"Monomer: {m_name}", f"Polymer: {p_name}"])
            
            # Обработка highlight атомов
            if match_data['mon_pol_matches'].get(m_name):
                match_info = match_data['mon_pol_matches'][m_name]
                if isinstance(match_info, dict):  # Для случая 1:1
                    highlight.extend([list(match_info.keys()), list(match_info.values())])
                else:  # Для случая 1:N или N:1
                    highlight.extend([[], []])
            elif match_data['mon_pol_matches'].get(p_name):
                match_info = match_data['mon_pol_matches'][p_name]
                highlight.extend([list(match_info.keys()), list(match_info.values())])
            else:
                raise ValueError(f"The match_data does not contain information about {m_name} or {p_name} matching.")
                
    # Без сопоставления
    else:
        for m_name, m_chem in monomer_chem_dict.items():
            chem_list.append(m_chem)
            legend_list.append(f"Monomer: {m_name}")
        for p_name, p_chem in polymer_chem_dict.items():
            chem_list.append(p_chem)
            legend_list.append(f"Polymer: {p_name}")
        
        # Балансировка списков для сетки
    # max_pairs_per_row = n_row if n_row > 0 else 4
    # mols_per_row = min(2 * max_pairs_per_row, len(chem_list))

    # Конфигурация отображения
    IPythonConsole.drawOptions.addAtomIndices = add_small_atom_index
    rdDepictor.SetPreferCoordGen(prefer_coord_gfen)
    
    # Генерация изображения
    drawing = Draw.MolsToGridImage(
        mols=chem_list,
        legends=legend_list,
        highlightAtomLists=highlight if match_data else None,
        molsPerRow=n_row,
        useSVG=useSVG,
        subImgSize=img_size
    )

    IPythonConsole.drawOptions.addAtomIndices = False
    rdDepictor.SetPreferCoordGen(False)
    return drawing

def path_parser(path: str, file_types: list):
    """
    Парсит путь и возвращает путь к директории и имя файла без расширений из списка file_types.
    """
    # Форматирование расширений, добавление '.' если отсутствует
    formated_file_types = ['.' + t if not t.startswith('.') else t for t in file_types]
    
    # Разделение пути и имени файла
    path_to_file, file_name = os.path.split(path)
    if path_to_file:
        os.makedirs(path_to_file, exist_ok=True)
    # else:
    #     path_to_file = 'current directory'

    # Удаление указанных расширений из имени файла
    for type_name in formated_file_types:
        file_name = file_name.replace(type_name, '')

    return path_to_file, file_name
        
def find_amino_nitrogen(mol):
    """
    Находит индекс атома азота в составе аминокислоты.
    Предполагается, что любая а.к. имеет вид основу [H][N]CC=O.
    """
    aa_base = '[H][N]CC=O'
    sub_base = mol.GetSubstructMatch(Chem.MolFromSmarts(aa_base))
    if sub_base:
        print(f"atom number: {sub_base[1]} became the root atom: 0")
        return sub_base[1]
    return -1

def save_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, rootedAtAtom=-1):
    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True) 
    smiles_string = Chem.MolToSmiles(rdkit_mol, canonical = canonical, 
                                     allHsExplicit = allHsExplicit, 
                                     rootedAtAtom = rootedAtAtom)
    with open(f"{path}.smiles", "w") as file:
        file.write(smiles_string)

def save_aa_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, make_N_root=False):
    """
    Сохраняет молекулу в формате SMILES.
    Аргументы:
        rdkit_mol - RDKit.Chem молекула
        path - строка с путем к файлу и его именем для сохранения
        make_N_root - искать ли аминогруппу и делать ли ее началом молекулы
        canonical, allHsExplicit - параметры для генерации SMILES
    """
    if make_N_root:
        rootedAtAtom = find_amino_nitrogen(rdkit_mol)
    else:
        rootedAtAtom = -1

    # Получение пути и имени файла
    path_to_file, file_name = path_parser(path, ['smi', 'smiles'])
    
    # Генерация строки SMILES
    smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=canonical, 
                                     allHsExplicit=allHsExplicit, 
                                     rootedAtAtom=rootedAtAtom)

    # Сохранение в файл
    output_path = f"{path_to_file}/{file_name}" if path_to_file else {file_name}
    with open(f"{output_path}.smiles", "w") as file:
        file.write(smiles_string)
    print(f'{file_name}.smiles saved to {path_to_file or "current directory"}')
        
def save_chem_to_mol(rdkit_mol, path, rootedAtAtom=-1):
    """
    Сохраняет молекулу в формате MOL, устанавливая корневой атом, если задан.
    """
    if rootedAtAtom != -1:
        # Используем SMILES для переупорядочивания атомов
        smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, 
                                         allHsExplicit=True, rootedAtAtom=rootedAtAtom)
        rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=True)
    
    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True) 
    with open(f"{path}.mol", "w") as file:
        file.write(Chem.MolToMolBlock(rdkit_mol))
        
def save_chem_to_pdb(rdkit_mol, path, rootedAtAtom=-1):
    """
    Сохраняет молекулу в формате pdb, устанавливая корневой атом, если задан.
    """
    if rootedAtAtom != -1:
        # Используем SMILES для переупорядочивания атомов
        smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, 
                                         allHsExplicit=True, rootedAtAtom=rootedAtAtom)
        rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
    try:
        Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL, catchErrors=True)
    except ValueError as e:
        print("Ошибка санации:", e)
    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True) 
    AllChem.EmbedMolecule(rdkit_mol)
    AllChem.UFFOptimizeMolecule(rdkit_mol)
    with open(f"{path}.pdb", "w") as file:
        file.write(Chem.MolToPDBBlock(rdkit_mol))
        
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

def rdkit_pdb_modification(rdkit_mol, resname="MOD", resid=1, segid="A",
                           force_numeric=False, C_terminal=False):
    """
    Назначает PDB имена атомов с поуровневым сбором и глобальными индексами на уровне.
    """
    from collections import deque, defaultdict
    
    greek_levels = {1: "B", 2: "G", 3: "D", 4: "E", 5: "Z", 6: "H",
                    7: "T", 8: "I", 9: "K", 10: "L", 11: "M", 12: "N"}

    atom_names = {}
    used_names = set()
    visited = set()
    numeric_mode = force_numeric
    heavy_counter = 0
    processed_hydrogens = set()
    hydrogen_counts = {}

    # ---------------------------------------------------------
    # 1. Поиск backbone
    # ---------------------------------------------------------
    CA_idx, backbone = find_backbone_match(rdkit_mol, C_terminal)

    backbone_names = ["N", "CA", "C", "O"]
    if len(backbone) == 5:
        backbone_names = ["N", "CA", "C", "OC1", "OC2"]

    for idx, name in zip(backbone, backbone_names):
        atom_names[idx] = name
        used_names.add(name)
        if rdkit_mol.GetAtomWithIdx(idx).GetSymbol() != "H":
            heavy_counter += 1
        visited.add(idx)

    N_idx = backbone[0]

    # ---------------------------------------------------------
    # 2. Назначение HA
    # ---------------------------------------------------------
    for nbr in rdkit_mol.GetAtomWithIdx(CA_idx).GetNeighbors():
        if nbr.GetSymbol() == "H" and nbr.GetIdx() not in atom_names:
            atom_names[nbr.GetIdx()] = "HA"
            used_names.add("HA")
            processed_hydrogens.add(nbr.GetIdx())

    # ---------------------------------------------------------
    # 3. Протоны аминогруппы
    # ---------------------------------------------------------
    amine_counter = 1
    for nbr in rdkit_mol.GetAtomWithIdx(N_idx).GetNeighbors():
        if nbr.GetSymbol() == "H" and nbr.GetIdx() not in atom_names:
            name = f"H{amine_counter}"
            atom_names[nbr.GetIdx()] = name
            used_names.add(name)
            processed_hydrogens.add(nbr.GetIdx())
            amine_counter += 1

    # ---------------------------------------------------------
    # 4. Поуровневый сбор атомов
    # ---------------------------------------------------------
    current_level = [(CA_idx, 0)]  # (atom_idx, depth)
    next_level = []
    
    # Словарь для сбора атомов по уровням
    # {depth: [(atom_idx, element, parent_idx)]}
    level_atoms = defaultdict(list)
    
    # BFS для сбора всех атомов по уровням
    level = 0
    while current_level:
        next_level = []
        for current_idx, depth in current_level:
            current_atom = rdkit_mol.GetAtomWithIdx(current_idx)
            
            for nbr in current_atom.GetNeighbors():
                nbr_idx = nbr.GetIdx()
                
                if nbr_idx in visited or nbr_idx in backbone:
                    continue
                    
                if nbr.GetSymbol() != "H":  # Тяжелый атом
                    element = nbr.GetSymbol()
                    
                    # Сохраняем атом для этого уровня
                    level_atoms[depth + 1].append((nbr_idx, element, current_idx))
                    visited.add(nbr_idx)
                    next_level.append((nbr_idx, depth + 1))
        
        current_level = next_level
    
    # ---------------------------------------------------------
    # 5. Назначение имен по уровням
    # ---------------------------------------------------------
    for depth in sorted(level_atoms.keys()):
        atoms_on_level = level_atoms[depth]
        
        print(f"\nУровень {depth}: {len(atoms_on_level)} атомов")
        
        # Определяем букву для этого уровня
        if not numeric_mode and depth in greek_levels:
            level_letter = greek_levels[depth]
            print(f"  Буква уровня: {level_letter}")
        else:
            level_letter = None
            if not numeric_mode:
                numeric_mode = True
                print(f"  Переход в numeric режим на глубине {depth}")
        
        # Глобальный счетчик для этого уровня
        level_counter = 1
        
        # Назначаем имена ВСЕМ атомам на этом уровне
        for atom_idx, element, parent_idx in atoms_on_level:
            name = None
            
            # Greek режим
            if not numeric_mode and level_letter:
                # Базовая часть имени: элемент + буква уровня
                base = f"{element}{level_letter}"
                
                # ВСЕГДА добавляем глобальный индекс уровня
                # Даже если в имени уже есть индекс (OZ1, NZ2), мы его заменяем!
                candidate = f"{base}{level_counter}"
                
                if len(candidate) <= 4 and candidate not in used_names:
                    name = candidate
                    print(f"    {element}{level_letter} (бывший) -> {name} (индекс {level_counter})")
            
            # Numeric режим
            if name is None:
                if not numeric_mode:
                    numeric_mode = True
                    print(f"    Переход в numeric режим для {element} на глубине {depth}")
                
                heavy_counter += 1
                base_name = f"{element}{heavy_counter}"
                name = base_name[:4]
                
                # Проверка уникальности
                counter = 1
                while name in used_names:
                    name = f"{base_name}_{counter}"[:4]
                    counter += 1
                
                print(f"    {element} -> {name} (numeric)")
            
            atom_names[atom_idx] = name
            used_names.add(name)
            
            # Сохраняем информацию о том, какой глобальный индекс получил атом
            atom_global_index = level_counter
            level_counter += 1
            
            # -------------------------------------------------
            # Водороды этого атома
            # -------------------------------------------------
            atom = rdkit_mol.GetAtomWithIdx(atom_idx)
            h_neighbors = []
            
            for h in atom.GetNeighbors():
                h_idx = h.GetIdx()
                if (h.GetSymbol() == "H" and 
                    h_idx not in processed_hydrogens and 
                    h_idx not in backbone):
                    h_neighbors.append(h)
            
            if h_neighbors:
                if name not in hydrogen_counts:
                    hydrogen_counts[name] = 1
                
                # Для водородов используем родительский индекс
                parent_index = "".join(c for c in name if c.isdigit())
                
                if numeric_mode:
                    for h in h_neighbors:
                        hname = f"H{parent_index}{hydrogen_counts[name]}"
                        hydrogen_counts[name] += 1
                        
                        if len(hname) > 4:
                            hname = f"H{parent_index}"[:4]
                        
                        # Проверка уникальности
                        counter = 1
                        base_hname = hname
                        while hname in used_names:
                            hname = f"{base_hname}_{counter}"[:4]
                            counter += 1
                        
                        atom_names[h.GetIdx()] = hname
                        used_names.add(hname)
                        processed_hydrogens.add(h.GetIdx())
                        print(f"      водород {hname}")
                else:
                    # В Greek режиме водороды наследуют родительский суффикс
                    parent_suffix = name[1:]  # все кроме первого символа
                    
                    for h in h_neighbors:
                        if len(h_neighbors) == 1:
                            hname = f"H{parent_suffix}"
                        else:
                            hname = f"H{parent_suffix}{hydrogen_counts[name]}"
                            hydrogen_counts[name] += 1
                        
                        if len(hname) > 4:
                            hname = f"H{parent_suffix}"[:4]
                        
                        # Проверка уникальности
                        counter = 1
                        base_hname = hname
                        while hname in used_names:
                            hname = f"{base_hname}_{counter}"[:4]
                            counter += 1
                        
                        atom_names[h.GetIdx()] = hname
                        used_names.add(hname)
                        processed_hydrogens.add(h.GetIdx())
                        print(f"      водород {hname}")

    # ---------------------------------------------------------
    # 6. Проверка пропущенных атомов
    # ---------------------------------------------------------
    for atom in rdkit_mol.GetAtoms():
        idx = atom.GetIdx()
        if idx not in atom_names:
            print(f"Предупреждение: атом {idx} ({atom.GetSymbol()}) не получил имя")
            element = atom.GetSymbol()
            heavy_counter += 1
            base_name = f"{element}{heavy_counter}"
            name = base_name[:4]
            
            counter = 1
            while name in used_names:
                name = f"{base_name}_{counter}"[:4]
                counter += 1
            
            atom_names[idx] = name
            used_names.add(name)

    # ---------------------------------------------------------
    # 7. Запись PDB
    # ---------------------------------------------------------
    for atom in rdkit_mol.GetAtoms():
        idx = atom.GetIdx()
        name = atom_names.get(idx, "UNK")
        
        info = Chem.AtomPDBResidueInfo()
        info.SetName(name.ljust(4))
        info.SetResidueName(resname)
        info.SetResidueNumber(resid)
        info.SetChainId(segid)
        atom.SetProp("AtomName", name.strip())
        atom.SetMonomerInfo(info)

    check_duplicate_atom_names(rdkit_mol)
    return rdkit_mol


def check_duplicate_atom_names(mol):
    """Проверяет уникальность имен атомов"""
    name_to_indices = {}
    for atom in mol.GetAtoms():
        if atom.HasProp("AtomName"):
            name = atom.GetProp("AtomName")
            idx = atom.GetIdx()
            if name not in name_to_indices:
                name_to_indices[name] = []
            name_to_indices[name].append(idx)
    
    duplicates = {name: indices for name, indices in name_to_indices.items() if len(indices) > 1}
    
    if duplicates:
        print("⚠️ Обнаружены дубликаты имен:")
        for name, indices in duplicates.items():
            print(f"  {name}: индексы {indices}")
        return False
    
    print("✅ Все имена атомов уникальны")
    return True
    
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
                print(f"Не учтенный вариант {atom.GetProp('AtomName')}:",
                     f"{name}, {hyb}, {n_hydrogens}, {n_heavy_neighbors}", sep='\n')
                geom_n = '-'
        elif hyb == Chem.HybridizationType.SP2:
            if n_hydrogens == 1 and n_heavy_neighbors == 2:
                geom_n = 1
            elif n_hydrogens == 2 and n_heavy_neighbors == 1:
                geom_n = 3
            elif n_hydrogens == 1 and n_heavy_neighbors == 1:
                geom_n = 2
            else:
                print(f"Не учтенный вариант {atom.GetProp('AtomName')}:",
                     f"{name}, {hyb}, {n_hydrogens}, {n_heavy_neighbors}", sep='\n')
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
        
def save_aa_chem_to_pdb(rdkit_mol, path, resname = 'MOD', resid = 1, segid = 'A', make_N_root=False, atom_names_list=None):
    """
    Сохраняет молекулу в формате PDB, устанавливая корневой атом, если задан.
    Аргументы:
        rdkit_mol - RDKit.Chem молекула
        path - строка с путем к файлу и его именем для сохранения
        make_N_root - искать ли аминогруппу и делать ли ее началом молекулы
    """
    
    if make_N_root:
        root_idx = find_amino_nitrogen(rdkit_mol)
        if root_idx != -1:
            smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, allHsExplicit=True, rootedAtAtom=root_idx)

            rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
            if rdkit_mol is None:
                raise ValueError("Ошибка при создании молекулы после перенумерации атомов")
    try:
        # Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE, catchErrors=True)
        Chem.SanitizeMol(rdkit_mol)
    except ValueError as e:
        print("Ошибка санации:", e)
        
    # Получение пути и имени файла
    path_to_file, file_name = path_parser(path, ['pdb'])
    
    # Генерация 3D-конформации и оптимизация молекулы
    AllChem.EmbedMolecule(rdkit_mol)
    AllChem.UFFOptimizeMolecule(rdkit_mol)
    
    rdkit_mol = rdkit_pdb_modification(rdkit_mol, 
                     resname = resname, resid = resid, segid = segid)
    rw_mol = Chem.RWMol(rdkit_mol)  # делаем редактируемую копию
    h_indices = [atom.GetIdx() for atom in rw_mol.GetAtoms() if atom.GetAtomicNum() == 1]
    for idx in sorted(h_indices, reverse=True):  # удалять нужно с конца, иначе индексы съедут
        rw_mol.RemoveAtom(idx)
    rdkit_mol_no_H = rw_mol.GetMol()
    
    # Запись в файл
    
    output_path = f"{path_to_file}/{file_name}.pdb" if path_to_file else f"{file_name}.pdb"
    with open(output_path, "w") as file:
        file.write(Chem.MolToPDBBlock(rdkit_mol))
    print(f'{file_name}.pdb saved to {path_to_file or "current directory"}')
    
    if '_H'in file_name:
        file_name = file_name.replace('_H', '_no_H')
    else:
        file_name = file_name + '_no_H'
    output_path = f"{path_to_file}/{file_name}.pdb" if path_to_file else f"{file_name}.pdb"
    with open(output_path, "w") as file:
        file.write(Chem.MolToPDBBlock(rdkit_mol_no_H))
    print(f'{file_name}.pdb saved to {path_to_file or "current directory"}')

def smi_to_mol2(smi_input, output_format, file_name=None, addh=True):
    """
    Converts an SMI string or file to a molecule file format (mol2, mol, mdl).

    Args:
      smi_input: The SMI string or path to an SMI file.
      output_format: The desired output format (mol2, mol, mdl).
      file_name (optional): The desired output file name (excluding extension).

    Returns:
      None. Raises an error for invalid input or format.
    """

    valid_formats = ['mol2', 'mol', 'mdl']
    if output_format not in valid_formats:
        raise ValueError(f"Invalid output format: {output_format}")
        
    # Check if SMI is a string
    if not isinstance(smi_input, str):
        raise TypeError("Input must be a string")

    # Read SMI string from file or argument
    if smi_input.lower().endswith('.smi') or smi_input.lower().endswith('.smiles') :
        smi_str = open_smi(smi_input) 
    else:
        smi_str = smi_input.strip()
        if file_name is None:
            file_name = 'convert_mol' 

    # Print current molecule
    print(f"Current molecule: {smi_str}")

    # Create molecule, add hydrogens, and generate 3D coordinates
    mol = pybel.readstring("smi", smi_str)
    if  addh:
        mol.addh()
    mol.make3D()

    # Construct output path based on input and arguments
    output_path = file_name + '.' + output_format if file_name else smi_input.split('.')[0] + '.'+output_format

    # Write molecule to file
    mol.write(output_format, output_path, overwrite=True)
    print('output path:',output_path)
    return mol



def rename_folder(folder_name):
    """
    Генерирует новое имя папки, добавляя или увеличивая числовой суффикс.
    """
    if '/' in folder_name:
        path, folder_name = folder_name.rsplit('/',1)
        path += '/'
    else: 
        path = ''
    
    if '_' in folder_name:
        head, tail = folder_name.rsplit('_',1)
        if tail.isdigit():
            new_tail = str(int(tail)+1).zfill(2)
            new_name = f'{head}_{new_tail}'
        else: new_name = f"{head}_{tail}_01"
    else:
        head = ''
        new_name = folder_name + '_01'
    return f"{path}{new_name}"

def get_unique_folder_name(folder_name: dict or str):
    """
    Возвращает уникальное имя папки, проверяя наличие папки и изменяя имя при необходимости.
    """
    try:
        if isinstance(folder_name, dict):
            name_of_dir = str(*folder_name.keys())
        if isinstance(folder_name, str):
            name_of_dir = folder_name

        while os.path.isdir(name_of_dir):
            print(f"Директория {name_of_dir} уже существует.")
            name_of_dir = rename_folder(name_of_dir)
        print(f"Новое имя директории: {name_of_dir}")
        return name_of_dir
    except TypeError as e:
        print_red(f'Словарь должен состоять из одного элемента: \nTypeError: {e}')
    except: 
        print_red('Что-то не так')

def show_list_of_conf(directory, show_full = False):
    """
    Показывает список файлов-конформеров в директории.
    """
    files = os.listdir(directory)
    msgpack_files = [file for file in files if file.endswith(".msgpack")]
    n_conf = len(msgpack_files)
    print(f"Колличество конформеров: {n_conf}")
    if show_full:
        conf_text = '\n'.join(msgpack_files)
        print(conf_text)

        
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem

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
    name_count = Counter(atom_names)
    print(name_count)
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
    info.SetName(atom_name.ljust(4))        # Имя должно быть ровно 4 символа
    info.SetResidueName(resname)            # Устанавливаем имя остатка
    info.SetResidueNumber(resid)            # Устанавливаем номер остатка
    info.SetChainId(segid)                  # Устанавливаем ID сегмента
    atom.SetMonomerInfo(info)
    # return info
    

def modifie_residue_info(modified_mol,  index_map, resname='MOD', resid=1, segid='A'):
    """
    Назначает новые имена атомам в молекуле согласно индексному отображению.
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
            set_PDB_residue_info(atom, atom_name, resname, resid, segid)
    return modified_mol

def increment_name(name):
    match = re.match(r'(\D+)(?:(\d*))$', name)
    if match:
        base, num_str = match.groups()
        num = int(num_str or '0')
        return f'{base}{num + 1}'
    else:
        return f'{name}1' # это странно, типа для случая CH1A 

def write_modified_molecule(mol, output_path):
    """
    Записывает измененную молекулу в PDB-файл.
    """
    writer = Chem.PDBWriter(output_path)
    for atom in mol.GetAtoms():
        atom.SetMonomerType(atom.GetProp('AtomName'))
    writer.write(mol)
    writer.close()
    
def save_molecule_as_pdb(molecule, filename, format_coord='3D'):
    """
    Сохраняет молекулу в формате PDB.

    :param molecule: Молекула в формате RDKit Mol.
    :param filename: Имя файла для сохранения.
    :param format: Формат представления ('2D' или '3D'). По умолчанию '3D'.
    """
    if format_coord == '2D':
        # Рассчитываем 2D координаты
        AllChem.Compute2DCoords(molecule)
    elif format_coord == '3D':
        # Рассчитываем 3D координаты (если еще не рассчитаны)
        Chem.SanitizeMol(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
    else:
        raise ValueError("Недопустимый формат. Выберите '2D' или '3D'.")
    
    path_to_file, file_name = path_parser(filename, ['pdb'])
    save_path = f'{path_to_file}/{file_name}_{format_coord}.pdb'
    # Сохраняем молекулу в PDB формате
    with open(save_path, 'w') as pdb_file:
        pdb_file.write(Chem.MolToPDBBlock(molecule))
    print(f"Молекула успешно сохранена в файле {save_path} в формате {format_coord}.")

def add_names_from_residue(modified_chem, index_map, resname='MOD', resid=1, segid='A'):

    
    mod_mol = list(modified_chem.values())[0] # костыль работы с словарем молекулы
    # ref_mol = list(reference_chem.values())[0] # костыль работы с словарем молекулы
   

    key_map = next(iter(match_data_dict['mon_pol_matches'].keys()))
    index_map = match_data_dict['mon_pol_matches'][key_map]
    # Присваиваем новые имена атомам в модифицированной молекуле
    # Перезадаем параметры модифицированного остатка  
    mod_residue = modifie_residue_info(mod_mol, index_map, resname, resid, segid)
    
    # Проверка наличия дублирующих имен атомов
    has_duplicates = check_duplicate_atom_names(mod_residue)

    return mod_residue
        
def save_charges_json(name: str, charge_list: list, path: str = ".") -> None:
    """
    Сохраняет список зарядов в JSON-файл с указанием абсолютного пути
    
    Параметры:
    name (str): Название файла (без расширения)
    charge_list (list): Список зарядов для сохранения
    path (str): Путь для сохранения (по умолчанию текущая директория)
    """
    # Создаем директорию, если она не существует
    os.makedirs(path, exist_ok=True)
    
    # Формируем полный путь к файлу
    file_path = os.path.join(path, f"{name}.json")
    absolute_path = os.path.abspath(file_path)
    
    # Сохраняем данные
    with open(file_path, 'w') as convert_file:
        json.dump(charge_list, convert_file, indent=4)
    
    # Выводим информативное сообщение
    print(f"Файл успешно сохранен: \n{absolute_path}")



## Работа с листом зарядов

def make_substructure_charge_list(pol_name, charge_array, match_dict, ):
    monomer_charge = [0] * (len(match_dict['substructure'][pol_name].GetAtoms()))
    # charge_array = charge_array.round(5)
    for monomer_index, polymer_index in enumerate(match_dict['mon_pol_matches'][pol_name].values()):
        monomer_charge[monomer_index] = charge_array[polymer_index]
    monomer_charge = np.array(monomer_charge).round(4)
    return monomer_charge


def calculate_true_sum(array):
    '''
    checking the actual sum of an array
    '''
    return sum([D(f'{val}') for val in array])

def save_json(name: str, list_data: list, path: str = ".") -> None:
    """
    Сохраняет список зарядов в JSON-файл с указанием абсолютного пути
    
    Параметры:
    name (str): Название файла (без расширения)
    list_data (list): Список зарядов для сохранения
    path (str): Путь для сохранения (по умолчанию текущая директория)
    """
    # Создаем директорию, если она не существует
    os.makedirs(path, exist_ok=True)
    
    # Формируем полный путь к файлу
    file_path = os.path.join(path, f"{name}.json")
    absolute_path = os.path.abspath(file_path)
    
    # Сохраняем данные
    with open(file_path, 'w') as convert_file:
        json.dump(list_data, convert_file, indent=4)
    
    # Выводим информативное сообщение
    print(f"Файл успешно сохранен: \n{absolute_path}")
        
# Финтифлюшки 

def print_green(text):
    """
    Выводит текст зеленым цветом в терминале.
    """
    print("\033[38;5;28m" + text + "\033[0m")        

def print_red(text):
    """
    Выводит текст зеленым цветом в терминале.
    """
    print("\33[31m" + text + "\33[0m") 
    
def animate(done_flag, sh_file_name):
    """
    Отображает анимацию выполнения процесса в терминале с именем sh файла.
    """
    for c in itertools.cycle(['.  ', '.. ', '...']):
        if done_flag():
            break
        sys.stdout.write(f'\rВыполнение {sh_file_name}{c}')
        sys.stdout.flush()
        time.sleep(0.5)
    # sys.stdout.write(f'\r{sh_file_name} выполнен успешно!  ')
    print_green(f'\n{sh_file_name} выполнен успешно!')

def format_time(seconds):
    """
    Форматирует время в секундах в строку формата HH:MM:SS.
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

# ---------------------------------------------

aa_dict = {
    'A': 'Ala', 'C': 'Cys', 'D': 'Asp', 'E': 'Glu',
    'F': 'Phe', 'G': 'Gly', 'H': 'His', 'I': 'Ile',
    'K': 'Lys', 'L': 'Leu', 'M': 'Met', 'N': 'Asn',
    'P': 'Pro', 'Q': 'Gln', 'R': 'Arg', 'S': 'Ser',
    'T': 'Thr', 'V': 'Val', 'W': 'Trp', 'Y': 'Tyr'
}
