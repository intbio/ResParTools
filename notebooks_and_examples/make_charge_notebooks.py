"""
Сборка ноутбуков 1_charge_calculation.ipynb для папок модификаций из шаблона.

Шаблон - templates/1_charge_calculation.ipynb (сделан из AF_546_cys/1_charge_calculation.ipynb).
Всё, что зависит от модификации, задаётся в ячейке параметров (PARAMS ниже); остальные
ячейки одинаковые во всех папках.

Старые ноутбуки не редактируются: если в папке уже есть 1_charge_calculation.ipynb, не
собранный из шаблона, он переименовывается в 1_charge_calculation_old.ipynb. Ноутбук,
собранный из шаблона, перезаписывается только с --overwrite.

Запуск (из notebooks_and_examples/):
    python make_charge_notebooks.py                 # все папки из PARAMS
    python make_charge_notebooks.py Lysine_Formyl   # одна папка
    python make_charge_notebooks.py --template      # пересобрать шаблон из AF_546_cys
"""
import argparse
import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(ROOT, 'templates', '1_charge_calculation.ipynb')
SOURCE = os.path.join(ROOT, 'AF_546_cys', '1_charge_calculation.ipynb')
NOTEBOOK = '1_charge_calculation.ipynb'
OLD_NOTEBOOK = '1_charge_calculation_old.ipynb'

# Параметры модификаций. naming: 'greek' - буквенные имена (лизины), 'index' - для больших
# меток. resp_run: где считать RESP на шаге 6 ('local' - эта машина, 'slurm' - кластер).
_LYS = dict(parent_residue='K', naming='greek', resp_run='local')
_DYE = dict(parent_residue='C', naming='index', resp_run='slurm')
PARAMS = {
    'AF_546_cys': dict(PTM_name='C_AF546', base_name='546', monomer_file='molecules/C_AF_546.smiles',
                       trimer_file='molecules/ACA_AF_546.smiles', **_DYE),
    'AF_546_rec': dict(PTM_name='C_AF546', base_name='C546', monomer_file='molecules/C_AF_546.smiles',
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


def make_template():
    """Шаблон из AF_546_cys: параметры-заглушки, без выводов и без пустых ячеек."""
    nb = json.load(open(SOURCE))
    nb = parametrize(nb, PARAMS['AF_546_cys'])
    nb['cells'] = [c for c in nb['cells'] if _src(c).strip()]
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    nb['metadata']['respartools'] = {'template': 'templates/1_charge_calculation.ipynb',
                                     'source': 'AF_546_cys/1_charge_calculation.ipynb'}
    os.makedirs(os.path.dirname(TEMPLATE), exist_ok=True)
    json.dump(nb, open(TEMPLATE, 'w'), ensure_ascii=False, indent=1)
    open(TEMPLATE, 'a').write('\n')
    print(f'шаблон: {TEMPLATE} ({len(nb["cells"])} ячеек)')


def make_notebook(folder, overwrite=False):
    """Собирает <папка>/1_charge_calculation.ipynb из шаблона по PARAMS[папка]."""
    params = PARAMS[folder]
    path = os.path.join(ROOT, folder, NOTEBOOK)
    if os.path.exists(path):
        existing = json.load(open(path))
        from_template = 'respartools' in existing.get('metadata', {})
        if from_template and not overwrite:
            print(f'{folder}: {NOTEBOOK} уже собран из шаблона, пропуск (--overwrite - пересобрать)')
            return None
        if not from_template:
            old = os.path.join(ROOT, folder, OLD_NOTEBOOK)
            if os.path.exists(old):
                raise FileExistsError(f'{folder}: есть и {NOTEBOOK}, и {OLD_NOTEBOOK} - разберитесь вручную')
            os.rename(path, old)
            print(f'{folder}: старый {NOTEBOOK} переименован в {OLD_NOTEBOOK}')
    for key in ('monomer_file', 'trimer_file'):
        if not os.path.exists(os.path.join(ROOT, folder, params[key])):
            raise FileNotFoundError(f'{folder}: нет файла {params[key]}')
    nb = parametrize(copy.deepcopy(json.load(open(TEMPLATE))), params)
    nb['metadata']['respartools'] = {'template': 'templates/1_charge_calculation.ipynb', 'params': params}
    json.dump(nb, open(path, 'w'), ensure_ascii=False, indent=1)
    open(path, 'a').write('\n')
    print(f'{folder}: {NOTEBOOK} собран из шаблона')
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('folders', nargs='*', help='папки модификаций (по умолчанию все из PARAMS)')
    parser.add_argument('--template', action='store_true', help='пересобрать шаблон из AF_546_cys')
    parser.add_argument('--overwrite', action='store_true', help='пересобрать ноутбуки, уже собранные из шаблона')
    args = parser.parse_args(argv)
    if args.template:
        make_template()
        return
    unknown = [f for f in args.folders if f not in PARAMS]
    if unknown:
        sys.exit(f'Нет параметров для {unknown}: добавьте их в PARAMS')
    for folder in args.folders or [f for f in PARAMS if f != 'AF_546_cys']:
        make_notebook(folder, overwrite=args.overwrite)


if __name__ == '__main__':
    main()
