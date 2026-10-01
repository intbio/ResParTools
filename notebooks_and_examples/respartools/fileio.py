"""Чтение и запись молекул: SMILES, PDB, mol2, MOL, JSON."""
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
from .utils import data_to_dict, path_parser, print_red


## 

@data_to_dict
@logged
def read_file(path):
    '''
    Читает первую строку из файла и возвращает её.
    '''
    with open (path, 'r') as f:
        return f.readline().strip()

def _mol_from_mapped_smiles(str_smi, sanitize=True):
    """
    Читает SMILES, записанный с atom_map=True (номер у каждого атома, 1..n).
    Водороды берутся из файла как есть (открытые валентности остатка в цепи не
    «залечиваются»), атомы расставляются по номерам, номера снимаются.
    Если номера есть не у всех атомов, возвращает None - SMILES читается как обычно.
    """
    if not re.search(r'\[[^\]]*:\d+\]', str_smi):
        return None
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = sanitize
    mol = Chem.MolFromSmiles(str_smi, params)
    if mol is None:
        return None
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if sorted(maps) != list(range(1, mol.GetNumAtoms() + 1)):
        return None
    order = [0] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        order[atom.GetAtomMapNum() - 1] = atom.GetIdx()
    mol = Chem.RenumberAtoms(mol, order)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


@data_to_dict
@logged
def smi_to_chem(str_smi: str, sanitize=True, addH=True, make_N_root=False,format_coord='2D'):
    """
    Преобразует SMILES-строку в объект молекулы RDKit и добавляет атомы водорода.
    """
    rdkit_mol = _mol_from_mapped_smiles(str_smi, sanitize)
    if rdkit_mol is None:
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


def _finalize_read_mol(rdkit_mol, path, sanitize, removeHs, format_coord):
    """
    Общий хвост для pdb_to_chem и mol2_to_chem: sanitize, удаление H, координаты.
    Свойства атомов (AtomName, PDB residue info, заряды) переживают RemoveHs,
    поэтому имена остаются привязаны к своим атомам.
    """
    if sanitize:
        try:
            Chem.SanitizeMol(rdkit_mol)
        except Exception as e:
            raise ValueError(
                f"{path}: структура не прошла sanitize ({e}). "
                "Обычно это лишняя связь или не указан формальный заряд. "
                "Исправьте файл или вызовите с sanitize=False (без ароматичности и проверки валентностей)."
            )
    if removeHs:
        rdkit_mol = Chem.RemoveHs(rdkit_mol, sanitize=sanitize)

    if format_coord == '2D':
        AllChem.Compute2DCoords(rdkit_mol)
    elif format_coord != '3D':
        raise ValueError("Недопустимый формат. Выберите '2D' или '3D'.")

    rdkit_mol.SetProp('AtomNames', str(rdkit_mol.GetNumAtoms()))
    return rdkit_mol


@data_to_dict
@logged
def pdb_to_chem(path_to_pdb, removeHs=False, make_N_root=False, sanitize=True, format_coord='2D'):
    """
    Открывает PDB файл при помощи RDKit и сохраняет имена атомов как свойства RDKit-атомов.

    Аргументы:
        path_to_pdb (str): Путь к файлу PDB.
        removeHs (bool): Удалять ли водороды. По умолчанию False.
        make_N_root (bool): Не используется, оставлен для совместимости вызовов.
        sanitize (bool): Проверка валентностей и распознавание ароматичности.
            Нужна, чтобы ароматические связи отличались от двойных так же, как в SMILES.
        format_coord (str): '2D' - пересчитать координаты для рисования,
            '3D' - оставить координаты из файла.

    Связи берутся из CONECT, если он есть в файле; иначе RDKit определяет их по расстояниям
    (proximity bonding). Вместе с CONECT proximity bonding не используется: на 2D и
    неоптимизированных структурах он добавляет фантомные связи.

    Возвращает:
        Chem.Mol: Молекула RDKit с сохраненными именами атомов (AtomName).
    """
    with open(path_to_pdb, "r") as pdb_file:
        has_conect = any(line.startswith("CONECT") for line in pdb_file)

    rdkit_mol = Chem.MolFromPDBFile(path_to_pdb, sanitize=False, removeHs=False,
                                    proximityBonding=not has_conect)
    if not rdkit_mol:
        raise ValueError(f"Не удалось загрузить PDB файл: {path_to_pdb}")

    # Имена атомов RDKit читает из фиксированных колонок PDB (13-16)
    for atom in rdkit_mol.GetAtoms():
        atom.SetProp("AtomName", atom.GetPDBResidueInfo().GetName().strip())

    return _finalize_read_mol(rdkit_mol, path_to_pdb, sanitize, removeHs, format_coord)


def _read_mol2_atom_block(path_to_mol2):
    """
    Читает секции @<TRIPOS>MOLECULE и @<TRIPOS>ATOM (первой молекулы в файле).
    Возвращает (имя молекулы, тип зарядов, список словарей по атомам в порядке файла).
    """
    with open(path_to_mol2, "r") as f:
        lines = f.read().splitlines()

    mol_name, charge_type, atoms = "", "", []
    section = None
    mol_line = 0
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            if section == "ATOM":
                break
            section = line[len("@<TRIPOS>"):].strip()
            mol_line = 0
            continue
        if section == "MOLECULE":
            mol_line += 1
            if mol_line == 1:
                mol_name = line.strip()
            elif mol_line == 4:
                charge_type = line.strip()
        elif section == "ATOM" and line.strip():
            fields = line.split()
            atom = {"id": int(fields[0]), "name": fields[1], "type": fields[5]}
            atom["subst_id"] = int(fields[6]) if len(fields) > 6 else 1
            atom["subst_name"] = fields[7] if len(fields) > 7 else "UNL"
            atom["charge"] = float(fields[8]) if len(fields) > 8 else 0.0
            atoms.append(atom)
    return mol_name, charge_type, atoms


@data_to_dict
@logged
def mol2_to_chem(path_to_mol2, sanitize=True, removeHs=False, format_coord='2D'):
    """
    Открывает mol2 файл при помощи RDKit и переносит в молекулу всё, что есть в секции ATOM.

    Порядки связей (включая ароматические 'ar') берутся из секции BOND.
    RDKit сам сохраняет имя, тип SYBYL и заряд атома, но теряет номер и имя остатка,
    поэтому секция ATOM дополнительно разбирается вручную.

    Аргументы:
        path_to_mol2 (str): Путь к файлу mol2 (типы атомов SYBYL; файлы с типами GAFF RDKit не читает).
        sanitize (bool): Проверка валентностей и распознавание ароматичности.
        removeHs (bool): Удалять ли водороды.
        format_coord (str): '2D' - пересчитать координаты для рисования, '3D' - оставить из файла.

    Свойства атомов:
        AtomName (str), Mol2AtomType (str), PartialCharge (float),
        PDB residue info: имя атома, имя и номер остатка.
        Имя остатка: subst_name без хвоста с номером остатка (конвенция Tripos: 'KMA2' -> 'KMA').
    Свойства молекулы:
        AtomNames, Mol2Name, Mol2ChargeType.

    Возвращает:
        Chem.Mol
    """
    rdkit_mol = Chem.MolFromMol2File(path_to_mol2, sanitize=False, removeHs=False)
    if not rdkit_mol:
        raise ValueError(f"Не удалось загрузить mol2 файл: {path_to_mol2} "
                         "(RDKit понимает только типы атомов SYBYL, не GAFF)")

    mol_name, charge_type, atoms = _read_mol2_atom_block(path_to_mol2)
    if len(atoms) != rdkit_mol.GetNumAtoms():
        raise ValueError(f"{path_to_mol2}: в секции ATOM {len(atoms)} атомов, "
                         f"а RDKit прочитал {rdkit_mol.GetNumAtoms()}")

    for atom, info in zip(rdkit_mol.GetAtoms(), atoms):
        # RDKit сохраняет порядок атомов из файла; проверяем, что сопоставление верное
        if atom.HasProp("_TriposAtomName") and atom.GetProp("_TriposAtomName") != info["name"]:
            raise ValueError(f"{path_to_mol2}: порядок атомов RDKit не совпадает с файлом "
                             f"(атом {info['id']}: {atom.GetProp('_TriposAtomName')} != {info['name']})")
        resname = info["subst_name"]
        if resname.endswith(str(info["subst_id"])) and len(resname) > len(str(info["subst_id"])):
            resname = resname[:-len(str(info["subst_id"]))]

        atom.SetProp("AtomName", info["name"])
        atom.SetProp("Mol2AtomType", info["type"])
        atom.SetDoubleProp("PartialCharge", info["charge"])
        set_PDB_residue_info(atom, info["name"], resname=resname, resid=info["subst_id"], segid="A")

    # В mol2 нет поля формального заряда, и RDKit превращает заряженные атомы в радикалы
    # или отвергает их. Восстанавливаем заряды по валентности:
    #   N с 4 связями (аммоний, иминий)                          -> +1
    #   O с одной одинарной связью без H (сульфонат, карбоксилат) -> -1,
    #   только если водороды в файле явные: иначе так же выглядит обычная OH-группа
    rdkit_mol.UpdatePropertyCache(strict=False)
    has_explicit_H = any(atom.GetAtomicNum() == 1 for atom in rdkit_mol.GetAtoms())
    charged = []
    for atom in rdkit_mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            continue
        valence = atom.GetExplicitValence()
        if atom.GetAtomicNum() == 7 and valence == 4:
            atom.SetFormalCharge(1)
            charged.append(atom.GetProp("AtomName") + "(+1)")
        elif atom.GetAtomicNum() == 8 and valence == 1 and has_explicit_H:
            atom.SetFormalCharge(-1)
            atom.SetNumRadicalElectrons(0)
            charged.append(atom.GetProp("AtomName") + "(-1)")
    if charged:
        print(f"{path_to_mol2}: формальные заряды восстановлены по валентности: {', '.join(charged)}")

    rdkit_mol.SetProp("Mol2Name", mol_name)
    rdkit_mol.SetProp("Mol2ChargeType", charge_type)
    return _finalize_read_mol(rdkit_mol, path_to_mol2, sanitize, removeHs, format_coord)

@logged
def file_opener(path, **kwargs):
    """
    Открывает файл с молекулой, автоматически определяя формат (.smi/.smiles, .mol2, .pdb).

    Parameters:
    -----------
    path : str или list[str]
        Путь к файлу или список путей (форматы в списке могут быть разными)
    **kwargs : dict
        Дополнительные параметры для функций загрузки
    Возвращает:
        dict {имя файла без расширения: Chem.Mol}, как pdb_to_chem и smi_to_chem
    """
    if isinstance(path, (list, tuple)):
        result = {}
        for one_path in path:
            opened = file_opener(one_path, **dict(kwargs))
            duplicates = set(result) & set(opened)
            if duplicates:
                print_red(f'⚠ Молекулы с одинаковыми именами {sorted(duplicates)}: '
                          f'{one_path} заменяет открытую ранее.')
            result.update(opened)
        return result

    path_obj = Path(path)
    name = path_obj.stem
    extension = path_obj.suffix.lower()
    
    print(f"Загрузка файла: {name} (формат: {extension})")
    
    if extension in (".smi", ".smiles"):
        smi_str = read_file(path)
        rdkit_mol = smi_to_chem(smi_str, **kwargs)
    elif extension in (".mol2", ".pdb"):
        # Для файлов с геометрией по умолчанию сохраняем координаты из файла
        kwargs.setdefault('format_coord', '3D')
        reader = mol2_to_chem if extension == ".mol2" else pdb_to_chem
        rdkit_mol = reader(path, **kwargs)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {extension}. "
                       f"Поддерживаемые: .smi, .smiles, .mol2, .pdb")
    
    return rdkit_mol
        
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

def _with_atom_map(mol):
    """Копия молекулы, в которой номер атома в SMILES = индекс + 1 (0 в SMILES - «без номера»)."""
    mapped = Chem.Mol(mol)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mapped


def _reroot_via_smiles(rdkit_mol, rootedAtAtom, func_name, sanitize):
    """
    Старое поведение rootedAtAtom в save_chem_to_pdb / save_chem_to_mol: молекула
    переписывается через SMILES с корнем в атоме rootedAtAtom. Индексы атомов меняются,
    имена атомов, параметры остатка, свойства молекулы и координаты теряются.
    """
    print_red(f'⚠ {func_name}(rootedAtAtom={rootedAtAtom}) устарел: атомы переставляются через '
              'SMILES, индексы меняются, имена атомов и параметры остатка теряются. '
              'Для порядка атомов остатка используйте renumber_residue_atoms.')
    smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True,
                                     allHsExplicit=True, rootedAtAtom=rootedAtAtom)
    return Chem.MolFromSmiles(smiles_string, sanitize=sanitize)


@logged
def save_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, rootedAtAtom=-1,
                        atom_map=False):
    """
    Сохраняет молекулу в SMILES (<path>.smiles). Молекула не меняется.

    Аргументы:
        rdkit_mol (Chem.Mol) - молекула
        path (str) - путь к файлу; расширение .smi/.smiles можно не писать
        canonical (bool) - канонический порядок обхода графа (True) или порядок индексов (False)
        allHsExplicit (bool) - писать водороды в квадратных скобках у каждого атома
        rootedAtAtom (int) - атом, с которого начинается запись строки (-1 - по умолчанию).
            Меняет только текст строки, не молекулу
        atom_map (bool) - записать у каждого атома его номер (индекс + 1, т.к. 0 в SMILES -
            «без номера»); smi_to_chem по этим номерам восстановит исходные индексы
    """
    path_to_file, file_name = path_parser(path, ['smi', 'smiles'])
    smiles_string = Chem.MolToSmiles(_with_atom_map(rdkit_mol) if atom_map else rdkit_mol,
                                     canonical=canonical,
                                     allHsExplicit=allHsExplicit,
                                     rootedAtAtom=rootedAtAtom)
    output_path = os.path.join(path_to_file, f'{file_name}.smiles')
    with open(output_path, "w") as file:
        file.write(smiles_string)
    print(f'Сохранено: {output_path}')


@logged
def save_aa_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, make_N_root=False,
                           atom_map=False):
    """
    Старое имя save_chem_to_smiles (параметры те же, порядок позиционных аргументов прежний).
    make_N_root=True - строка начинается с азота аминогруппы (find_amino_nitrogen);
    меняется только текст строки, не молекула.
    """
    rootedAtAtom = find_amino_nitrogen(rdkit_mol) if make_N_root else -1
    save_chem_to_smiles(rdkit_mol, path, canonical=canonical, allHsExplicit=allHsExplicit,
                        rootedAtAtom=rootedAtAtom, atom_map=atom_map)


@logged
def save_chem_to_mol(rdkit_mol, path, rootedAtAtom=-1):
    """
    Сохраняет молекулу в формате MOL (<path>.mol).
    rootedAtAtom - устаревший параметр (см. _reroot_via_smiles): функция предупреждает
    и переставляет атомы, как раньше.
    """
    if rootedAtAtom != -1:
        rdkit_mol = _reroot_via_smiles(rdkit_mol, rootedAtAtom, 'save_chem_to_mol', sanitize=True)

    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True)
    with open(f"{path}.mol", "w") as file:
        file.write(Chem.MolToMolBlock(rdkit_mol))


@logged
def save_chem_to_pdb(rdkit_mol, path, rootedAtAtom=-1, coords='2D', stereo=None, ca_config='L',
                     stereo_default='R', random_seed=42):
    """
    Сохраняет молекулу в PDB (<path>.pdb) с её индексацией атомов. Если у атомов есть
    PDB-информация (имена атомов, остаток - после rdkit_pdb_modification или из PDB-файла),
    она записывается; иначе RDKit пишет свои имена (элемент + номер), остаток UNL, HETATM.
    Переданная молекула не меняется (координаты задаются на копии).

    Аргументы:
        rdkit_mol (Chem.Mol) - молекула
        path (str) - путь без расширения
        rootedAtAtom (int) - устаревший параметр (см. _reroot_via_smiles): функция
            предупреждает и переставляет атомы, как раньше
        coords (str) - '2D' (по умолчанию) или '3D'; stereo, ca_config, stereo_default,
            random_seed - параметры 3D-структуры и стереохимии, см. set_mol_coords.
            До 2026-09 функция всегда строила 3D-структуру со случайной стереохимией.
    """
    rdkit_mol = Chem.Mol(rdkit_mol)
    if rootedAtAtom != -1:
        rdkit_mol = _reroot_via_smiles(rdkit_mol, rootedAtAtom, 'save_chem_to_pdb', sanitize=False)
    failed = Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL, catchErrors=True)
    if failed != Chem.SanitizeFlags.SANITIZE_NONE:
        print_red(f'⚠ Санитизация не прошла ({failed}): молекула сохраняется как есть.')
    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True)
    set_mol_coords(rdkit_mol, coords=coords, stereo=stereo, ca_config=ca_config,
                   stereo_default=stereo_default, random_seed=random_seed)
    pdb_block = Chem.MolToPDBBlock(rdkit_mol)
    with open(f"{path}.pdb", "w") as file:
        file.write(pdb_block)
        
@logged
def save_aa_chem_to_pdb(rdkit_mol, path, resname = 'MOD', resid = 1, segid = 'A', make_N_root=False,
                        atom_names_list=None, naming='greek', parent_atoms=None, coords='2D',
                        stereo=None, ca_config='L', stereo_default='R', random_seed=42):
    """
    Сохраняет остаток в PDB: имена атомов (rdkit_pdb_modification), имя, номер и цепь остатка.
    Пишутся два файла: <path>.pdb и <path>_no_H.pdb. Переданная молекула не меняется
    (всё делается на копии), поэтому повторный запуск ячейки даёт тот же результат.

    Аргументы:
        rdkit_mol - RDKit.Chem молекула с явными водородами
        path - строка с путем к файлу и его именем для сохранения
        resname, resid, segid - имя остатка (до 3 символов), номер, цепь
        make_N_root - устаревший параметр: переставить атомы через SMILES с корнем в аминогруппе
        atom_names_list - не используется, оставлен для совместимости вызовов
        naming - способ именования атомов: 'greek' (по умолчанию) или 'index'
        parent_atoms - индексы атомов родительского остатка для naming='index'
        coords - '2D' (по умолчанию) или '3D'; stereo, ca_config, stereo_default,
            random_seed - параметры 3D-структуры и стереохимии, см. set_mol_coords.
            До 2026-09 функция всегда строила 3D-структуру со случайной стереохимией.
    """
    rdkit_mol = Chem.Mol(rdkit_mol)
    if make_N_root and parent_atoms is not None:
        raise ValueError('make_N_root переставляет атомы, и индексы parent_atoms перестают им '
                         'соответствовать: используйте что-то одно.')
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
    
    set_mol_coords(rdkit_mol, coords=coords, stereo=stereo, ca_config=ca_config,
                   stereo_default=stereo_default, random_seed=random_seed)

    if len(resname) > 3:
        print_red(f"⚠ Имя остатка '{resname}' длиннее 3 символов: RDKit запишет в PDB только "
                  f"'{resname[:3]}'.")
    rdkit_mol = rdkit_pdb_modification(rdkit_mol, resname=resname, resid=resid, segid=segid,
                                       naming=naming, parent_atoms=parent_atoms)
    # RemoveHs с пересчётом валентностей: иначе у ароматических атомов, потерявших явные H,
    # не определено число водородов, кольцо не кекулизуется и PDB не записывается
    # (так было при удалении H через RemoveAtom и при RemoveHs(sanitize=False))
    rdkit_mol_no_H = Chem.RemoveHs(rdkit_mol)

    # Запись в файл: текст PDB формируется до открытия файла, чтобы при ошибке
    # не оставить на месте старого файла пустой

    output_path = f"{path_to_file}/{file_name}.pdb" if path_to_file else f"{file_name}.pdb"
    pdb_block = Chem.MolToPDBBlock(rdkit_mol)
    with open(output_path, "w") as file:
        file.write(pdb_block)
    print(f'{file_name}.pdb saved to {path_to_file or "current directory"}')

    if '_H'in file_name:
        file_name = file_name.replace('_H', '_no_H')
    else:
        file_name = file_name + '_no_H'
    output_path = f"{path_to_file}/{file_name}.pdb" if path_to_file else f"{file_name}.pdb"
    pdb_block = Chem.MolToPDBBlock(rdkit_mol_no_H)
    with open(output_path, "w") as file:
        file.write(pdb_block)
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
        smi_str = next(iter(read_file(smi_input).values()))
    else:
        smi_str = smi_input.strip()
        if file_name is None:
            file_name = 'convert_mol' 

    # Print current molecule
    print(f"Current molecule: {smi_str}")

    # Create molecule, add hydrogens, and generate 3D coordinates
    from openbabel import pybel  # openbabel нужен только этой функции
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

def write_modified_molecule(mol, output_path):
    """
    Записывает измененную молекулу в PDB-файл.
    """
    writer = Chem.PDBWriter(output_path)
    for atom in mol.GetAtoms():
        atom.SetMonomerType(atom.GetProp('AtomName'))
    writer.write(mol)
    writer.close()
    
@logged
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

@logged
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


# Импорт из других модулей пакета - в конце файла: функции этого модуля уже
# определены, поэтому взаимные ссылки модулей друг на друга не мешают импорту.
from .residue import rdkit_pdb_modification, set_PDB_residue_info, set_mol_coords  # noqa: E402
