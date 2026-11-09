"""
Сборка ноутбуков папок модификаций из шаблонов (templates/).

Путь словарей и тримеров (old): 1_charge_calculation_old.ipynb, 2_generate_topology_old.ipynb,
3_edd_topology_old.ipynb - из шаблонов templates/<то же имя>. Всё, что зависит от модификации, задаётся
в ячейке параметров (PARAMS ниже); остальные ячейки одинаковые во всех папках. Ноутбук, не собранный
из шаблона, не перезаписывается (ошибка); собранный - только с --overwrite.

Запуск (из notebooks_and_examples/):
    python make_charge_notebooks.py                 # все папки из PARAMS
    python make_charge_notebooks.py Lysine_Formyl   # одна папка
    python make_charge_notebooks.py --props [папки] # версия без словарей: 1_charge_calculation_props.ipynb
        (шаблон templates/1_charge_calculation_props.ipynb, перезаписывается всегда)
"""
import argparse
import copy
from datetime import date
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
OLD_NOTEBOOKS = ('1_charge_calculation_old.ipynb', '2_generate_topology_old.ipynb', '3_edd_topology_old.ipynb')
PROPS_TEMPLATE = os.path.join(ROOT, 'templates', '1_charge_calculation_props.ipynb')
PROPS_NOTEBOOK = '1_charge_calculation_props.ipynb'
WORKFLOW_NOTEBOOKS = ('1_charge_calculation.ipynb', '2_acpype.ipynb', '3_forcefield_md.ipynb')
WORKFLOW_PARAMS_CELL = """# Параметры модификации (сгенерировано для этой папки)
FOLDER = {folder!r}
PTM_name = {PTM_name!r}
base_name = {base_name!r}
monomer_file = {monomer_file!r}
naming = {naming!r}
parent_override = {parent_override!r}  # None - выбрать по сопоставлению
run_resp_now = False
OUT = os.path.abspath(os.path.join('..', 'build', 'notebooks', FOLDER))
FIGURES = os.path.join(OUT, 'figures')
os.makedirs(FIGURES, exist_ok=True)"""

# Параметры модификаций. naming: 'greek' - буквенные имена (лизины), 'index' - для больших
# меток. resp_run: где считать RESP на шаге 6 ('local' - эта машина, 'slurm' - кластер).
# exclusion_atoms - атомы родителя, не наследующие его тип в 3_edd_topology_old (старое правило).
_LYS = dict(parent_residue='K', naming='greek', resp_run='local', exclusion_atoms=['HE1', 'HE2'])
_DYE = dict(parent_residue='C', naming='index', resp_run='slurm', exclusion_atoms=[])
PARAMS = {
    'AF_546_cys': dict(PTM_name='C_AF546', base_name='546', monomer_file='molecules/C_AF_546.smiles',
                       trimer_file='molecules/ACA_AF_546.smiles', **_DYE),
    'AF_647_cys': dict(PTM_name='AF_647', base_name='647', monomer_file='molecules/AF_647.smiles',
                       trimer_file='molecules/ACA_AF_647.smiles', **_DYE),
    'Lysine_1M': dict(PTM_name='Lysine_1M', base_name='K1M', monomer_file='molecules/Lysine_1M.smiles',
                      trimer_file='molecules/AKA_1M.smiles', **_LYS),
    'Lysine_1M_ACC': dict(PTM_name='Lysine_1M_ACC', base_name='KMA', monomer_file='molecules/Lysine_1M_ACC.smiles',
                          trimer_file='molecules/AKA_1M_ACC.smiles', **_LYS),
    'Lysine_2M': dict(PTM_name='Lysine_2M', base_name='K2M', monomer_file='molecules/Lysine_2M.smiles',
                      trimer_file='molecules/AKA_2M.smiles', **_LYS),
    'Lysine_3M': dict(PTM_name='Lysine_3M', base_name='K3M', monomer_file='molecules/Lysine_3M.smiles',
                      trimer_file='molecules/AKA_3M.smiles', **_LYS),
    'Lysine_ACC': dict(PTM_name='Lysine_ACC', base_name='KAC', monomer_file='molecules/K_ACC.smiles',
                       trimer_file='molecules/AKA_ACC.smiles', **_LYS),
    'Lysine_Benzoyl': dict(PTM_name='Lysine_Benzoyl', base_name='KBO', monomer_file='molecules/K_benzoyl.smi',
                           trimer_file='molecules/G_K_benzoyl_G.smi', **_LYS),
    'Lysine_Butyryl': dict(PTM_name='Lysine_Butyryl', base_name='KBU', monomer_file='molecules/Lysine_butyryl.smiles',
                           trimer_file='molecules/AK_butyrulA.smiles', **_LYS),
    'Lysine_Cro': dict(PTM_name='Lysine_Cro', base_name='KCR', monomer_file='molecules/Kcr_H.smiles',
                       trimer_file='molecules/GKcrG.smiles', **_LYS),
    'Lysine_Formyl': dict(PTM_name='Lysine_Formyl', base_name='KFO', monomer_file='molecules/K_form.smiles',
                          trimer_file='molecules/AK_formA.smiles', **_LYS),
    'Lysine_Malonyl': dict(PTM_name='Lysine_Malonyl', base_name='KML', monomer_file='molecules/Lysine_malonyl.smiles',
                           trimer_file='molecules/AK_malonylA.smiles', **_LYS),
    'Lysine_lac': dict(PTM_name='Lysine_lac', base_name='KLA', monomer_file='molecules/Lysine_lac.smi',
                       trimer_file='molecules/G_Lysine_lac_G.smi', **_LYS),
    'Lysine_prop': dict(PTM_name='Lysine_prop', base_name='KPR', monomer_file='molecules/K_prop.smiles',
                        trimer_file='molecules/AKA_prop.smiles', **_LYS),
}
LYSINE_FOLDERS = tuple(name for name in PARAMS if name.startswith('Lysine_'))
VALIDATION_PARENTS = {'4PQ': 'TRP', 'ILX': 'ILE', 'SMC': 'CYS', 'SME': 'MET', '2AS': 'ASP'}
VALIDATION_PARAMS = {
    code: dict(PTM_name=code, base_name=code, monomer_file=f'molecules/{code}.smiles',
               naming='greek', parent_override=parent)
    for code, parent in VALIDATION_PARENTS.items()
}
WORKFLOW_PARAMS = {**{folder: PARAMS[folder] for folder in LYSINE_FOLDERS}, **VALIDATION_PARAMS}


PARAM_CELL = """# Параметры модификации (всё, что отличает одну модификацию от другой)
PTM_name = {PTM_name!r}            # имя модификации: файлы шага 4.1 и вход шага 2 ({{PTM_name}}_rn_H_3D.pdb)
base_name = {base_name!r}          # имя остатка в силовом поле (как в rtp)
parent_residue = {parent_residue!r}  # родительский остаток (однобуквенный код): шаблон, имена, ограничения зарядов
monomer_file = {monomer_file!r}    # мономер: SMILES остатка с концевыми группами
trimer_file = {trimer_file!r}      # тример: остаток с соседями (контекст в цепи)
naming = {naming!r}                # имена атомов на шаге 3.2: 'greek' (буквенные) или 'index' (большие метки)
resp_run = {resp_run!r}            # шаг 6 (RESP): 'local' - эта машина, 'slurm' - кластер"""

STEP1_CELL = """monomer_chem = pt.file_opener(monomer_file)
trimers_chem = pt.file_opener([trimer_file])
mon_name = next(iter(monomer_chem))
pol_name = next(iter(trimers_chem))
print(mon_name, pol_name, sep='\\n')"""

S6_RUN_CELL = """# запуск: resp_run из ячейки параметров ('local' - считать здесь, 'slurm' - задача для кластера)
pt.run_resp(resp_folder, run=resp_run, n_threads=4, n_parallel=1,
            slurm=dict(cpus=24, mem='32G', time_limit='08:00:00'))"""

S6_AF_SENTENCE = ("Метка AF546 с кэпами - около 135 атомов: одна конформация - десятки часов на 24 ядрах, "
                  "поэтому для неё `run='slurm'`.")
S6_GENERAL_SENTENCE = ("Для больших остатков (метки, около 100 атомов и больше) одна конформация считается "
                       "десятки часов даже на 24 ядрах - для них `resp_run='slurm'`.")


def _src(cell):
    return ''.join(cell['source'])


def _set_src(cell, text):
    lines = text.split('\n')
    cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]]


def _cell(nb, cell_id):
    for cell in nb['cells']:
        if cell.get('id') == cell_id:
            return cell
    raise KeyError(f'В ноутбуке нет ячейки {cell_id}')


def parametrize(nb, params):
    """Ячейка параметров, шаг 1, имена атомов шага 3 и запуск шага 6 - из параметров."""
    _set_src(_cell(nb, 'd6f22e29-d388-415b-b313-068fea6901cd'), PARAM_CELL.format(**params))
    _set_src(_cell(nb, '04da68d4'), STEP1_CELL)
    s3 = _cell(nb, 's3-check')
    text = _src(s3)
    for old in ("naming='index'", "naming='greek'"):
        text = text.replace(old, 'naming=naming')
    _set_src(s3, text)
    _set_src(_cell(nb, 's6-run'), S6_RUN_CELL)
    s6md = _cell(nb, 's6-md')
    _set_src(s6md, _src(s6md).replace(S6_AF_SENTENCE, S6_GENERAL_SENTENCE))
    return nb


STEP2_PARAMS = """# Параметры старого ноутбука 2_generate_topology_old
PTM_name = {PTM_name!r}   # имя входа {{PTM_name}}_rn_H_3D.pdb
base_name = {base_name!r} # имя остатка в силовом поле
charge_val = None         # None - формальный заряд молекулы"""

WORKFLOW_STEP2_PARAMS = """# Параметры модификации (заполняются генератором)
FOLDER = {folder!r}
PTM_name = {PTM_name!r}
base_name = {base_name!r}
monomer_file = {monomer_file!r}
OUT = os.path.abspath(os.path.join('..', 'build', 'notebooks', FOLDER))
INPUT_BASE = os.path.join(OUT, f'4_{{os.path.splitext(os.path.basename(monomer_file))[0]}}.sdf')
INPUT_H = os.path.join(OUT, f'4_1_{{os.path.splitext(os.path.basename(monomer_file))[0]}}.sdf')
ACPYPE_DIR = os.path.join(OUT, 'acpype')
os.makedirs(ACPYPE_DIR, exist_ok=True)"""

WORKFLOW_STEP3_PARAMS = """# Параметры модификации (заполняются генератором)
FOLDER = {folder!r}
PTM_name = {PTM_name!r}
base_name = {base_name!r}
monomer_file = {monomer_file!r}
OUT = os.path.abspath(os.path.join('..', 'build', 'notebooks', FOLDER))
INPUT = os.path.join(OUT, f'5_{{os.path.splitext(os.path.basename(monomer_file))[0]}}.sdf')
FF_OUT = os.path.abspath(os.path.join('..', 'build', 'amber14sb_mod_smi.ff'))
MD_OUT = os.path.abspath(os.path.join('..', 'build', 'md_from_smiles', FOLDER))
MD_NS = 0.01
run_gromacs_now = False"""

STEP3_PARAMS = """# Параметры модификации
PTM_folder = {folder!r}      # папка модификации: имя rtp-файла в force_field_files/
PTM_name = {PTM_name!r}        # вход шага 2: {{PTM_name}}_rn_H_3D
base_name = {base_name!r}             # имя остатка в силовом поле
parent_residue = {parent_residue!r}          # родительский остаток (однобуквенный код)
mon_name = {mon_name!r}        # мономер: заряды data/AI_chrges_{{mon_name}}.json
charges_file = f'data/AI_chrges_{{mon_name}}.json'   # или f'data/RESP_chrges_{{mon_name}}.json'
# атомы родителя, которые НЕ наследуют его тип (старое правило; новый путь определяет их по окружению):
# у лизинов HE1/HE2 - водороды при NZ, в лизине у них тип HP (рядом с N+)
exclusion_atoms = {exclusion_atoms!r}"""


def _notebook(name, folder):
    """Ноутбук name для папки folder из шаблона templates/<name>."""
    params = dict(PARAMS[folder], folder=folder,
                  mon_name=os.path.splitext(os.path.basename(PARAMS[folder]['monomer_file']))[0])
    nb = copy.deepcopy(json.load(open(os.path.join(ROOT, 'templates', name))))
    if name.startswith('1_'):
        nb = parametrize(nb, params)
    else:
        _set_src(_cell(nb, 'params'), (STEP2_PARAMS if name.startswith('2_') else STEP3_PARAMS).format(**params))
    nb['metadata']['respartools'] = {'template': f'templates/{name}', 'params': PARAMS[folder]}
    return nb


def make_notebook(folder, overwrite=False):
    """Собирает в папке 1_charge_calculation_old, 2_generate_topology_old, 3_edd_topology_old из шаблонов."""
    for key in ('monomer_file', 'trimer_file'):
        if not os.path.exists(os.path.join(ROOT, folder, PARAMS[folder][key])):
            raise FileNotFoundError(f'{folder}: нет файла {PARAMS[folder][key]}')
    paths = []
    for name in OLD_NOTEBOOKS:
        path = os.path.join(ROOT, folder, name)
        if os.path.exists(path):
            if 'respartools' not in json.load(open(path)).get('metadata', {}):
                raise FileExistsError(f'{folder}: {name} не собран из шаблона - перенесите его в archive/notebooks/')
            if not overwrite:
                print(f'{folder}: {name} уже собран из шаблона, пропуск (--overwrite - пересобрать)')
                continue
        json.dump(_notebook(name, folder), open(path, 'w'), ensure_ascii=False, indent=1)
        open(path, 'a').write('\n')
        print(f'{folder}: {name} собран из шаблона')
        paths.append(path)
    return paths


def make_props_notebook(folder):
    """<папка>/1_charge_calculation_props.ipynb - версия без словарей (docs/MOL_PROPS.md)."""
    nb = copy.deepcopy(json.load(open(PROPS_TEMPLATE)))
    cell = _cell(nb, 'params')
    lines = _src(cell).split('\n')
    keys = ('PTM_name', 'base_name', 'parent_residue', 'monomer_file', 'trimer_file', 'naming', 'resp_run')
    _set_src(cell, '\n'.join(f'{line.split(" =")[0]} = {PARAMS[folder][line.split(" =")[0]]!r}'
                              if line.split(' =')[0] in keys else line for line in lines))
    nb['metadata']['respartools'] = {'template': 'templates/1_charge_calculation_props.ipynb',
                                     'params': PARAMS[folder]}
    path = os.path.join(ROOT, folder, PROPS_NOTEBOOK)
    json.dump(nb, open(path, 'w'), ensure_ascii=False, indent=1)
    open(path, 'a').write('\n')
    print(f'{folder}: {PROPS_NOTEBOOK} собран из шаблона')
    return path


def _has_outputs(nb):
    return any(cell.get('cell_type') == 'code'
               and (cell.get('outputs') or cell.get('execution_count') is not None)
               for cell in nb.get('cells', []))


def _archive_existing(path):
    """Сохраняет занятую версию ноутбука в archive/notebooks без перезаписи."""
    folder = os.path.dirname(path)
    archive = os.path.join(folder, 'archive', 'notebooks')
    os.makedirs(archive, exist_ok=True)
    name = os.path.basename(path)
    dst = os.path.join(archive, name)
    if os.path.exists(dst):
        stem, ext = os.path.splitext(name)
        suffix = date.today().isoformat()
        dst = os.path.join(archive, f'{stem}_{suffix}{ext}')
        index = 2
        while os.path.exists(dst):
            dst = os.path.join(archive, f'{stem}_{suffix}_{index}{ext}')
            index += 1
    shutil.move(path, dst)
    print(f'Архивирована предыдущая версия: {dst}')
    return dst


def make_workflow_notebooks(folder, overwrite=False):
    """Собирает новые ноутбуки 1-3 на свойствах RDKit для одной модификации."""
    if folder not in WORKFLOW_PARAMS:
        raise ValueError(f'{folder}: для этого остатка нет параметров нового workflow')
    params = dict(WORKFLOW_PARAMS[folder], folder=folder)
    params.setdefault('parent_override', None)
    outputs = []
    for name in WORKFLOW_NOTEBOOKS:
        path = os.path.join(ROOT, folder, name)
        if os.path.exists(path):
            old = json.load(open(path))
            current_template = old.get('metadata', {}).get('respartools', {}).get('workflow') == 'properties_v1'
            if _has_outputs(old):
                _archive_existing(path)
            elif not current_template:
                if not overwrite:
                    raise FileExistsError(f'{path} уже существует без выполненных ячеек; '
                                          'переместите его в archive/notebooks/ или задайте --overwrite')
                _archive_existing(path)
            elif not overwrite:
                print(f'{folder}: {name} уже собран, пропуск (--overwrite - пересобрать)')
                continue
        nb = copy.deepcopy(json.load(open(os.path.join(ROOT, 'templates', name))))
        if name.startswith('1_'):
            params_cell = WORKFLOW_PARAMS_CELL
        elif name.startswith('2_'):
            params_cell = WORKFLOW_STEP2_PARAMS
        else:
            params_cell = WORKFLOW_STEP3_PARAMS
        _set_src(_cell(nb, 'params'), params_cell.format(**params))
        nb['metadata']['respartools'] = {
            'template': f'templates/{name}', 'params': WORKFLOW_PARAMS[folder], 'workflow': 'properties_v1'}
        with open(path, 'w') as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write('\n')
        print(f'{folder}: {name} собран')
        outputs.append(path)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('folders', nargs='*', help='папки модификаций (по умолчанию все из PARAMS)')
    parser.add_argument('--props', action='store_true', help='версия без словарей (1_charge_calculation_props.ipynb)')
    parser.add_argument('--workflow', action='store_true',
                        help='новый workflow на свойствах: ноутбуки 1, 2 и 3')
    parser.add_argument('--overwrite', action='store_true', help='пересобрать ноутбуки, уже собранные из шаблона')
    args = parser.parse_args(argv)
    supported = WORKFLOW_PARAMS if args.workflow and not args.props else PARAMS
    unknown = [f for f in args.folders if f not in supported]
    if unknown:
        sys.exit(f'Нет параметров для {unknown}: добавьте их в PARAMS')
    if args.props:
        for folder in args.folders or list(PARAMS):
            make_props_notebook(folder)
        return
    if args.workflow:
        folders = args.folders or list(LYSINE_FOLDERS)
        for folder in folders:
            make_workflow_notebooks(folder, overwrite=args.overwrite)
        return
    for folder in args.folders or list(PARAMS):
        make_notebook(folder, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
