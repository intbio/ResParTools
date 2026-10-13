"""Функции для совместимости со старыми ноутбуками."""
import os
import sys
import time

import threading
import subprocess


from rdkit import Chem
from rdkit.Chem import AllChem


from .log import logged
from .utils import (animate, format_time, get_unique_folder_name, path_parser, print_green, print_red,
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
    в <path>/<имя>.pdb (с именами атомов и параметрами остатка, save_chem_to_pdb)
    и <path>/<имя>.smiles (с номерами атомов).
    Замена: шаги 2.2 и 3 нового 1_charge_calculation.ipynb.
    """
    print_red('⚠ save_substructure_mol устарела: подструктуры сохраняются через '
              'save_chem_to_pdb(..., no_H=True) и save_chem_to_smiles (см. шаги 2.2 и 3 нового ноутбука).')
    for name in names:
        substructure = match_dict['substructure'][name]
        save_chem_to_pdb(substructure, f'{path}/{name}', resname=resname, resid=resid, segid=segid, no_H=True)
        save_chem_to_smiles(substructure, f'{path}/{name}', canonical=False, atom_map=True)


# =============================================================================
# УСТАРЕВШЕЕ ИЗ fileio: перестановка атомов через SMILES (make_N_root, rootedAtAtom),
# старые функции сохранения. Работают, как раньше, с предупреждением.
# Замена: порядок атомов остатка - renumber_residue_atoms; сохранение - save_chem_to_pdb,
# save_chem_to_smiles(atom_map=True), save_mol.
# =============================================================================

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


def _make_N_root(rdkit_mol, func_name):
    """
    Старое make_N_root=True в smi_to_chem: молекула переписывается через SMILES с корнем в
    азоте аминогруппы, затем тяжёлые атомы ставятся перед водородами.
    """
    print_red(f'⚠ {func_name}(make_N_root=True) устарел: атомы переставляются через SMILES. '
              'Для порядка атомов остатка используйте renumber_residue_atoms.')
    root_idx = find_amino_nitrogen(rdkit_mol)
    if root_idx != -1:
        smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, allHsExplicit=True, rootedAtAtom=root_idx)
        rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
        if rdkit_mol is None:
            raise ValueError("Ошибка при создании молекулы после перенумерации атомов")
    else:
        print("Азот аминогруппы не найден. Корень не изменён.")
    Chem.SanitizeMol(rdkit_mol)
    atom_order = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() != 'H']
    atom_order += [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == 'H']
    return Chem.RenumberAtoms(rdkit_mol, atom_order)


@logged
def save_aa_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, make_N_root=False,
                           atom_map=False):
    """
    Старое имя save_chem_to_smiles (параметры те же, порядок позиционных аргументов прежний).
    make_N_root=True - строка начинается с азота аминогруппы (find_amino_nitrogen);
    меняется только текст строки, не молекула.
    """
    print_red('⚠ save_aa_chem_to_smiles устарела: используйте save_chem_to_smiles '
              '(для порядка атомов - atom_map=True).')
    rootedAtAtom = find_amino_nitrogen(rdkit_mol) if make_N_root else -1
    save_chem_to_smiles(rdkit_mol, path, canonical=canonical, allHsExplicit=allHsExplicit,
                        rootedAtAtom=rootedAtAtom, atom_map=atom_map)


@logged
def save_aa_chem_to_pdb(rdkit_mol, path, resname='MOD', resid=1, segid='A', make_N_root=False,
                        atom_names_list=None, naming='greek', parent_atoms=None, coords='2D',
                        stereo=None, ca_config='L', stereo_default='R', random_seed=42):
    """
    Старое сохранение остатка в PDB: теперь save_chem_to_pdb(..., resname=..., no_H=True).
    Пишет <path>.pdb и <path>_no_H.pdb. make_N_root - устаревший параметр (перестановка атомов
    через SMILES с корнем в аминогруппе); atom_names_list не используется.
    """
    print_red('⚠ save_aa_chem_to_pdb устарела: используйте save_chem_to_pdb(mol, path, resname=..., '
              'resid=..., segid=..., naming=..., parent_atoms=..., no_H=True).')
    if make_N_root and parent_atoms is not None:
        raise ValueError('make_N_root переставляет атомы, и индексы parent_atoms перестают им '
                         'соответствовать: используйте что-то одно.')
    if make_N_root:
        root_idx = find_amino_nitrogen(rdkit_mol)
        if root_idx != -1:
            rdkit_mol = _reroot_via_smiles(rdkit_mol, root_idx, 'save_aa_chem_to_pdb', sanitize=False)
            if rdkit_mol is None:
                raise ValueError("Ошибка при создании молекулы после перенумерации атомов")
    save_chem_to_pdb(rdkit_mol, path, coords=coords, stereo=stereo, ca_config=ca_config,
                     stereo_default=stereo_default, random_seed=random_seed, resname=resname,
                     resid=resid, segid=segid, naming=naming, parent_atoms=parent_atoms, no_H=True)


def write_modified_molecule(mol, output_path):
    """
    Записывает измененную молекулу в PDB-файл.
    """
    print_red('⚠ write_modified_molecule устарела: используйте save_chem_to_pdb.')
    writer = Chem.PDBWriter(output_path)
    for atom in mol.GetAtoms():
        atom.SetMonomerType(atom.GetProp('AtomName'))
    writer.write(mol)
    writer.close()


@logged
def save_molecule_as_pdb(molecule, filename, format_coord='3D'):
    """
    Сохраняет молекулу в формате PDB (<filename>_<format_coord>.pdb). Меняет координаты
    переданной молекулы; 3D - со случайной стереохимией.
    """
    print_red("⚠ save_molecule_as_pdb устарела: используйте save_chem_to_pdb(mol, path, coords='3D') - "
              'молекула не меняется, стереохимия задаётся.')
    if format_coord == '2D':
        AllChem.Compute2DCoords(molecule)
    elif format_coord == '3D':
        Chem.SanitizeMol(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
    else:
        raise ValueError("Недопустимый формат. Выберите '2D' или '3D'.")
    path_to_file, file_name = path_parser(filename, ['pdb'])
    save_path = f'{path_to_file}/{file_name}_{format_coord}.pdb'
    with open(save_path, 'w') as pdb_file:
        pdb_file.write(Chem.MolToPDBBlock(molecule))
    print(f"Молекула успешно сохранена в файле {save_path} в формате {format_coord}.")


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
from .fileio import save_chem_to_pdb, save_chem_to_smiles  # noqa: E402
from .residue import renumber_residue_atoms  # noqa: E402
