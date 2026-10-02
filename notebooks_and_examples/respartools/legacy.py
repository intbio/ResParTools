"""Функции для совместимости со старыми ноутбуками."""
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
from .utils import (animate, format_time, get_unique_folder_name, print_green, print_red,
                    show_list_of_conf)


# =============================================================================
# ФУНКЦИИ СТАРЫХ НОУТБУКОВ, КОДА КОТОРЫХ НЕТ В ИСТОРИИ GIT
# Восстановлены по смыслу вызовов через новые функции; при вызове выводят
# предупреждение с заменой (правило проекта: старый вызов не падает).
# =============================================================================

@logged
def renumber_amino_acid_atoms(mol, smart_ref=None):
    """
    Старая перенумерация атомов аминокислоты (азот аминогруппы первым). Кода старой версии
    нет в репозитории; вызов передаётся в renumber_residue_atoms без родительского остатка:
    шаблон выбирается по числу совпавших атомов, как в старых версиях (ненадёжно).
    Замена: renumber_residue_atoms(mol, ref_base_name='K').
    """
    print_red("⚠ renumber_amino_acid_atoms устарела: используется renumber_residue_atoms "
              "без родительского остатка. Укажите его явно: "
              "pt.renumber_residue_atoms(mol, ref_base_name='K').")
    if smart_ref is not None:
        print_red(f'⚠ smart_ref={smart_ref!r} не используется: остов находится по шаблону остатка.')
    return renumber_residue_atoms(mol)


@logged
def save_substructure_mol(match_dict, path, names, resname='MOD', resid=1, segid='A'):
    """
    Старое сохранение подструктур из match_mon_to_pol. Кода старой версии нет в репозитории;
    для каждого имени из names подструктура match_dict['substructure'][имя] сохраняется
    в <path>/<имя>.pdb (с именами атомов и параметрами остатка, save_aa_chem_to_pdb)
    и <path>/<имя>.smiles (с номерами атомов).
    Замена: шаги 2.2 и 3 нового 1_charge_calculation.ipynb.
    """
    print_red('⚠ save_substructure_mol устарела: подструктуры сохраняются через '
              'save_aa_chem_to_pdb и save_chem_to_smiles (см. шаги 2.2 и 3 нового ноутбука).')
    for name in names:
        substructure = match_dict['substructure'][name]
        save_aa_chem_to_pdb(substructure, f'{path}/{name}', resname=resname, resid=resid, segid=segid)
        save_chem_to_smiles(substructure, f'{path}/{name}', canonical=False, atom_map=True)


# =============================================================================
# СТАРЫЙ ЗАПУСК RESP (RESP_calculations/param_tool.py)
# Перенесён без изменения логики. В этом режиме psiresp без сервера QCFractal
# не оптимизирует геометрию: psi4 считает один градиент, ESP считается на исходных
# конформациях RDKit. Новый расчёт - resp_charges (шаг 6).
# psiresp импортируется внутри функций: модуль работает и без него.
# =============================================================================

_OLD_RESP_WARNING = ('⚠ Старый запуск RESP: геометрия НЕ оптимизируется (psiresp без QCFractal '
                     'считает только градиент), ESP - на конформациях RDKit. '
                     'Для новых расчётов используйте шаг 6 (resp_charges).')


def add_constraint(RESP_mol, symmetric_list: list, charge_dict: dict, loc_charge: float,
                   loc_indices=None, constraints=None):
    """
    Создает ограничения по расчетам зарядов для одной молекулы
    Аргументы:
        RESP_mol - молеккула открытая в psiresp
        symmetric_list - лист с листами симитричных атомов
        charge_dict - словарь, где ключ - индекс атома, значение - заряд атома
        loc_indices - лист атомов сумарный заряд которых задается отдельно (loc_charge)
        loc_charge - значение сумарного заряда для подгруппы атомов (loc_indices) в молекуле
        constraints - уже созданные ограничения (по умолчанию - новые)
    Возвращает:
        constraints - ограничения по расчету зарядов для молекулы
    """
    import psiresp
    if constraints is None:
        constraints = psiresp.ChargeConstraintOptions()
    if loc_charge:
        constraints.add_charge_sum_constraint_for_molecule(RESP_mol, charge=loc_charge,
                                                           indices=loc_indices or [])
    if symmetric_list:
        for pair_list in symmetric_list:
            constraints.add_charge_equivalence_constraint_for_molecule(RESP_mol, indices=pair_list)
    if charge_dict:
        for index, charge in charge_dict.items():
            constraints.add_charge_sum_constraint_for_molecule(RESP_mol, charge=charge, indices=index)
    return constraints


def prepere_constraints(mol_name, mol_chem_dict, match_dict, constraint_dict, optimize_geometry=True,
                        charge=0, conformer_generation_options=None, symmetric_atoms_are_equivalent=True):
    """Старая подготовка молекулы psiresp и ограничений по constraint_dict (см. add_constraint)."""
    import psiresp
    if conformer_generation_options is None:
        conformer_generation_options = dict(n_conformer_pool=1000, n_max_conformers=3,
                                            energy_window=100, keep_original_conformer=False)
    psiresp_dict = {}
    RESP_mol = psiresp.Molecule.from_rdkit(mol_chem_dict[mol_name],
                                           optimize_geometry=optimize_geometry, charge=charge,
                                           conformer_generation_options=conformer_generation_options)
    psiresp_dict[mol_name] = RESP_mol
    constraints = psiresp.ChargeConstraintOptions(
        symmetric_atoms_are_equivalent=symmetric_atoms_are_equivalent)
    constraints = add_constraint(RESP_mol,
                                 constraint_dict[mol_name]['symmetric_list'],
                                 constraint_dict[mol_name]['charge_atom_dict'],
                                 constraint_dict[mol_name]['charge_of_monomer'],
                                 list(match_dict['mon_pol_matches'][mol_name].values()),
                                 constraints=constraints)
    return psiresp_dict, constraints


def run_job(RESP_mol, name_of_system, constraints, n_processes=None, resp_options=None):
    """Старый запуск psiresp.Job: выход SystemExit означает, что ждут расчёты psi4 (.sh)."""
    import psiresp
    RESP_list = list(RESP_mol.values()) if isinstance(RESP_mol, dict) else list(RESP_mol)
    kwargs = {'resp_options': resp_options} if resp_options is not None else {}
    job = psiresp.Job(molecules=RESP_list, working_directory=name_of_system,
                      n_processes=n_processes, **kwargs)
    job.charge_constraints = constraints
    try:
        job.run()
    except SystemExit as e:
        print(f'SystemExit: {e}')
    return job


def do_sh(name_sh: str):
    """
    Запускает .sh скрипт в указанной директории и отображает прогресс выполнения.
    """
    start_time = time.time()
    done = False
    t = threading.Thread(target=animate, args=(lambda: done, os.path.basename(name_sh),))
    t.start()
    try:
        if '/' in name_sh:
            cwd, name_sh = name_sh.rsplit('/', 1)
        else:
            cwd = None
        env = dict(os.environ)
        env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + env.get('PATH', '')
        subprocess.run(["bash", name_sh], cwd=cwd, check=True, env=env)
        done = True
        t.join()
        print_green(f'Время выполнения: {format_time(time.time() - start_time)}')
    except FileNotFoundError:
        done = True
        t.join()
        print(f"Не верно указана cwd: {cwd}")
    except subprocess.CalledProcessError as e:
        done = True
        t.join()
        print(f"Ошибка выполнения скрипта: {e}")


@logged
def resp_calculation(psiresp_dict, constraints, n_processes=None, folder_name='RESP_data/',
                     resp_options=None):
    """
    Старый расчёт RESP через psiresp (RESP_calculations). Геометрия не оптимизируется
    (см. _OLD_RESP_WARNING). Возвращает (job, папка расчёта).
    """
    print_red(_OLD_RESP_WARNING)
    path_folder = f'{folder_name}/{str(*psiresp_dict.keys())}'
    name_of_system = get_unique_folder_name(path_folder)
    print(f'Данные psiresp будут записаны по адресу: {name_of_system}')
    job = run_job(psiresp_dict, name_of_system, constraints, n_processes=n_processes,
                  resp_options=resp_options)
    time.sleep(0.7)
    show_list_of_conf(f'{name_of_system}/optimization/', False)
    do_sh(name_sh=f'{name_of_system}/optimization/run_optimization.sh')
    job = run_job(psiresp_dict, name_of_system, constraints, n_processes=n_processes,
                  resp_options=resp_options)
    time.sleep(0.7)
    do_sh(name_sh=f'{name_of_system}/single_point/run_single_point.sh')
    job = run_job(psiresp_dict, name_of_system, constraints, n_processes=n_processes,
                  resp_options=resp_options)
    return job, name_of_system


# Импорт из других модулей пакета - в конце файла (см. docs/PIPELINE.md, правило импортов).
from .fileio import save_aa_chem_to_pdb, save_chem_to_smiles  # noqa: E402
from .residue import renumber_residue_atoms  # noqa: E402
