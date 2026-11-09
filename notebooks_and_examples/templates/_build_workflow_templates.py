"""Собирает новые пользовательские ноутбуки 1-3 для пайплайна на RDKit properties."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
cells = []


def md(text, cid):
    cells.append({'cell_type': 'markdown', 'id': cid, 'metadata': {}, 'source': text.strip('\n')})


def code(text, cid):
    cells.append({'cell_type': 'code', 'id': cid, 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.strip('\n')})


def write(name):
    nb = {'cells': cells.copy(), 'metadata': {
        'kernelspec': {'display_name': 'darwin_ec', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'},
        'respartools': {'template': f'templates/{name}'}}, 'nbformat': 4, 'nbformat_minor': 5}
    for cell in nb['cells']:
        src = cell['source']
        cell['source'] = src.splitlines(keepends=True) if src else []
        if cell['cell_type'] == 'code' and cell['source'] and not cell['source'][-1].endswith('\n'):
            pass
    with open(HERE / name, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write('\n')
    cells.clear()


# Notebook 1: monomer -> parent mapping, atom names, Espaloma/RESP.
code('''%load_ext autoreload
%autoreload 2
import json
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../../espaloma-charge_mod/'))
import ResParTools as pt
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
''', 'imports')
md('''# 1. SMILES → имена атомов и заряды

На входе только модифицированный мономер. Все сопоставления и заряды сохраняются в свойствах RDKit;
между ноутбуками остаток передаётся через SDF. Автоматически найденный структурный родитель может
отличаться от химического родителя модификации: проверьте список кандидатов и картинку шага 3.''', 'intro')
code('''# Параметры модификации (заполняются генератором для этой папки)
FOLDER = 'Lysine_prop'
PTM_name = 'Lysine_prop'
base_name = 'KPR'
monomer_file = 'molecules/K_prop.smiles'
naming = 'greek'
parent_override = None  # при необходимости: точное имя кандидата из библиотеки, например 'LYS'
run_resp_now = False    # RESP долгий; False только готовит задачу для локального/кластерного запуска
OUT = os.path.abspath(os.path.join('..', 'build', 'notebooks', FOLDER))
FIGURES = os.path.join(OUT, 'figures')
os.makedirs(FIGURES, exist_ok=True)
''', 'params')
md('''## Шаг 1. Прочитать модифицированный мономер

Мономер задаёт химический граф и стереохимию. Тример пользователя не нужен.''', 's1-md')
code('''monomer = pt.open_mol(monomer_file)
mon_name = monomer.GetProp('ModName')
print(f'{mon_name}: {monomer.GetNumAtoms()} атомов; формальный заряд {Chem.GetFormalCharge(monomer):+d}')
figure = pt.draw_molecule(monomer, show_index=True)
figure.save(os.path.join(FIGURES, '01_monomer.png'))
figure
''', 's1')
md('''## Шаг 2. Найти структурного родителя

Сопоставление выполняется со всеми кэпированными шаблонами. Рейтинг — число атомов общей
подструктуры; при равенстве программа выбирает первый шаблон в порядке имён файлов. Это структурная
подсказка, а не химическое заключение. При необходимости задайте `parent_override` по точному имени
кандидата.''', 's2-md')
code('''ref_dict, match_data, auto_ref = pt.find_ref_residue(
    {'residue': monomer}, residue_type='protein_capped', main_match_data=False,
    timeout=3, strict_parent=False)
scores = match_data['N_match_atoms']
ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
for ref_name, n_atoms in ranked:
    print(f'{ref_name:5s}  общих атомов: {n_atoms}')

if parent_override is None:
    selected_ref = auto_ref
else:
    if parent_override not in scores:
        raise ValueError(f'{parent_override!r} отсутствует среди кандидатов: {sorted(scores)}')
    selected_ref = parent_override
print(f'Автоподбор: {auto_ref}')
print(f'⚠ Выбран структурный кандидат {selected_ref}; проверьте, что это химический родитель.')
''', 's2')
md('''## Шаг 3. Сопоставить с кэпированным остатком и выделить границы

MCS с центральным остатком ACE-X-NME задаёт карту канонических атомов. Несопоставленные ветви
модификации сохраняются. Границы N/C берутся из связей центрального остатка с ACE/NME в PDB-шаблоне;
входная форма SMILES и порядок атомов не задают границы. До зарядов проверьте список исключённых
заместителей и состав открытого остатка.''', 's3-md')
code('''ref_path = os.path.join(pt.REF_TEMPLATES['protein_capped']['dir'], f'{selected_ref}.pdb')
capped_ref = pt.pdb_to_chem(ref_path, format_coord='3D')
if isinstance(capped_ref, dict):
    capped_ref = capped_ref[selected_ref]
renumbered_monomer = pt.renumber_residue_atoms(
    monomer, ref_base_name=selected_ref, residue_type='protein_capped',
    path_to_ref_mol=ref_path, timeout=3, strict_parent=False)
residue = pt.open_peptide_boundaries(renumbered_monomer, capped_ref, residue_number=2)
print('Исключённые граничные заместители:', residue.GetProp('BoundarySubstituents'))
parent_atoms = json.loads(residue.GetProp('ParentAtoms'))
parent = pt.extract_residue(capped_ref, 2)
residue = pt.rdkit_pdb_modification(residue, resname=base_name, resid=2, segid='A',
                                    naming=naming, parent_atoms=parent_atoms,
                                    parent_reference=parent)
''', 's3-prepare')
code('''# Общая карта строится по ParentName, а не по совпавшим локальным индексам.
shared_residue = [a.GetIdx() for a in residue.GetAtoms() if a.HasProp('ParentName')]
shared_names = [residue.GetAtomWithIdx(i).GetProp('ParentName') for i in shared_residue]
shared_monomer = [a.GetIdx() for a in renumbered_monomer.GetAtoms()
                  if a.HasProp('ParentName') and a.GetProp('ParentName') in set(shared_names)]

parent_idx = {a.GetProp('AtomName'): a.GetIdx() for a in parent.GetAtoms()}
missing = sorted(set(shared_names) - set(parent_idx))
if missing:
    raise ValueError(f'Имена общей подструктуры отсутствуют в шаблоне: {missing}')
shared_parent = [parent_idx[name] for name in shared_names]

figure = pt.draw_mol_grid(
    [renumbered_monomer, residue, parent],
    legends=['Мономер после перенумерации (индексы)', 'Остаток с именами атомов',
             f'Родитель {selected_ref} с именами шаблона'],
    highlight_atoms=[shared_monomer, shared_residue, shared_parent],
    show_index=[True, False, False], show_names=[False, True, True], mols_per_row=3)
figure.save(os.path.join(FIGURES, '03_parent_mapping.png'))
figure
''', 's3-draw')
md('''## Шаг 4. Проверить и при необходимости поправить имена

`rename_map` задаётся как `{текущий индекс RDKit: 'новое имя'}`. Индексы отсчитываются
с нуля после выделения остатка. Все замены проверяются вместе до расчёта зарядов.''', 's4-md')
code('''rename_map = {}
residue = pt.modifie_residue_info(residue, rename_map, resname=base_name, resid=2, segid='A')
duplicates = pt.check_duplicate_atom_names(residue)
if duplicates:
    raise ValueError(f'Повторяющиеся имена атомов: {duplicates}')
for atom in residue.GetAtoms():
    print(f"{atom.GetIdx():3d} {atom.GetProp('AtomName'):5s} "
          f"родитель: {atom.GetProp('ParentName') if atom.HasProp('ParentName') else '-'}")
''', 's4')
md('''### Шаг 4.1. Подготовить водороды и 3D для acpype

`AddedH` отмечает HW на концах остатка; эти атомы нужны для топологии/проверки и не меняют набор
атомов самого остатка.''', 's41-md')
code('''h_residue = pt.add_protons_and_renumber_H(residue, resname=base_name, resid=2, segid='A')
h_residue = pt.set_mol_coords(h_residue, coords='3D', ca_config='L')
''', 's41')
md('''## Шаг 5. Заряды Espaloma

Заряды N, H, CA, HA, C и O фиксируются по `aminoacids.rtp` выбранного родительского типа;
остальные рассчитывает Espaloma. После пользовательских ограничений сумма зарядов должна остаться
равной формальному заряду остатка.''', 's5-md')
code('''from espaloma_charge import charge

parent_code = residue.GetProp('Parent')
rtp_parent = pt.AMINO_ACIDS[parent_code][0].upper()
constraints = pt.charge_constraints_from_rtp(residue, rtp_parent)

# Пример: задать известный заряд отдельного атома по его индексу.
# constraints[residue.GetAtomWithIdx(atom_idx).GetIdx()] = -0.30
# Пример: зафиксировать одинаковый заряд на эквивалентных атомах (ограничение задаётся
# отдельно для каждого атома; это не ограничение только суммы группы).
# equivalent_atoms = [idx_1, idx_2]
# constraints.update({idx: -0.15 for idx in equivalent_atoms})

q = charge(residue, constraints=constraints)
pt.set_charges(residue, q, 'Espaloma', constraints)
total = float(q.sum())
formal = Chem.GetFormalCharge(residue)
print(f'Сумма зарядов: {total:+.5f}; формальный заряд: {formal:+d}')
if abs(total - formal) > 1e-3:
    raise ValueError('Сумма частичных зарядов не совпадает с формальным зарядом.')
figure = pt.draw_molecule(residue, charge_list=q, size=(1500, 1200),
                          show_names=False, show_index=False)
figure.save(os.path.join(FIGURES, '05_espaloma_charges.png'))
figure
''', 's5')
md('''## Шаг 6. RESP (необязательно)

Подготовка сохраняет отдельные геометрии и контрольные точки. Для локального расчёта включите
`run_resp_now`; на кластере после проверки `spec.json` сгенерируйте SLURM-скрипт и отправьте его
командой `sbatch`. Здесь кластерная задача не отправляется автоматически.''', 's6-md')
code('''resp_folder = pt.prepare_resp_job(
    residue, constraints, f'{mon_name}_capped', working_dir=os.path.join(OUT, 'resp'),
    fix_backbone=True, n_sidechain=3, monomer=monomer)
print('Подготовлена RESP-задача:', resp_folder)

# Локально (долго):
if run_resp_now:
    q_resp = pt.run_resp(resp_folder, run='local', n_threads=4, n_parallel=1)
    pt.set_charges(residue, q_resp, 'RESP', constraints)
    print('Для следующего шага выбран RESP.')

# Кластер: сначала создать run_resp.sbatch; затем вручную отправить его через sbatch.
# slurm = {'cpus': 24, 'mem': '30G', 'time_limit': '08:00:00', 'partition': 'intbio'}
# pt.run_resp(resp_folder, run='slurm', slurm=slurm)
# В терминале: cd <путь resp_folder> && sbatch run_resp.sbatch
# После завершения: q_resp = pt.load_resp_result(resp_folder); pt.set_charges(residue, q_resp, 'RESP', constraints)
''', 's6')
md('''## Сохранить результат

Основной SDF содержит свойства атомов и молекулы; второй SDF включает водороды/3D для acpype.
Заряд Espaloma остаётся выбранным по умолчанию, пока не назначен RESP.''', 'save-md')
code('''sdf_path = pt.save_mol(residue, os.path.join(OUT, f'4_{mon_name}'))
h_sdf_path = pt.save_mol(h_residue, os.path.join(OUT, f'4_1_{mon_name}'))
print('Остаток:', sdf_path)
print('Остаток с H/3D для acpype:', h_sdf_path)
''', 'save')
write('1_charge_calculation.ipynb')


# Notebook 2: acpype-generated types and bonded parameters.
code('''%load_ext autoreload
%autoreload 2
import json
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import ResParTools as pt
''', 'imports')
md('''# 2. Построить топологию acpype

Вход — SDF из ноутбука 1 со свойствами RDKit и водородами, добавленными для топологии. acpype
задаёт атомные типы и недостающие bonded-параметры; его частичные заряды не подменяют выбранные
Espaloma/RESP заряды.''', 'intro')
code('''# Параметры модификации (заполняются генератором)
FOLDER = 'Lysine_prop'
PTM_name = 'Lysine_prop'
base_name = 'KPR'
monomer_file = 'molecules/K_prop.smiles'
OUT = os.path.abspath(os.path.join('..', 'build', 'notebooks', FOLDER))
INPUT_BASE = os.path.join(OUT, f'4_{os.path.splitext(os.path.basename(monomer_file))[0]}.sdf')
INPUT_H = os.path.join(OUT, f'4_1_{os.path.splitext(os.path.basename(monomer_file))[0]}.sdf')
ACPYPE_DIR = os.path.join(OUT, 'acpype')
os.makedirs(ACPYPE_DIR, exist_ok=True)
''', 'params')
md('## Шаг 1. Загрузить SDF и проверить входные атомы', 's1-md')
code('''residue = pt.load_mol(INPUT_BASE)  # без HW: эта копия пойдёт в RTP
acpype_residue = pt.load_mol(INPUT_H)  # HW нужны для корректной валентности acpype
assert residue.GetProp('ChargeMethod') in ('Espaloma', 'RESP')
names = [a.GetProp('AtomName') for a in residue.GetAtoms()]
assert len(names) == len(set(names)), 'Имена AtomName должны быть уникальны.'
assert acpype_residue.GetNumAtoms() > residue.GetNumAtoms()
assert [a.GetProp('AtomName') for a in list(acpype_residue.GetAtoms())[:residue.GetNumAtoms()]] == names
print('Остаток:', residue.GetProp('ResName'), 'атомов:', residue.GetNumAtoms(),
      'заряды:', residue.GetProp('ChargeMethod'),
      '| атомов с HW для acpype:', acpype_residue.GetNumAtoms())
''', 's1')
md('''## Шаг 2. Запустить acpype

`run_acpype` сверяет число и элементы атомов ITP с входной молекулой и записывает `AcpypeType` и
путь `AcpypeItp` в свойства.''', 's2-md')
code('''acpype_residue = pt.run_acpype(acpype_residue, workdir=ACPYPE_DIR)
for atom, atom_h in zip(residue.GetAtoms(), acpype_residue.GetAtoms()):
    if atom.GetProp('AtomName') != atom_h.GetProp('AtomName') or atom.GetSymbol() != atom_h.GetSymbol():
        raise ValueError(f'Порядок атомов SDF и acpype-версии различается на {atom.GetIdx()}')
    atom.SetProp('AcpypeType', atom_h.GetProp('AcpypeType'))
residue.SetProp('AcpypeItp', acpype_residue.GetProp('AcpypeItp'))
if acpype_residue.HasProp('AcpypeCapMap'):
    residue.SetProp('AcpypeCapMap', acpype_residue.GetProp('AcpypeCapMap'))
itp = residue.GetProp('AcpypeItp')
print('ITP:', itp)
atom_table = [(a.GetIdx(), a.GetProp('AtomName'), a.GetSymbol(), a.GetProp('AcpypeType'))
              for a in residue.GetAtoms()]
for row in atom_table:
    print(*row)
''', 's2')
md('''## Сохранить топологические типы

SDF сохраняет имена и заряды предыдущего шага вместе с типами acpype. Файл ITP остаётся в папке
`build/notebooks/<модификация>/acpype/`.''', 'save-md')
code('''OUTPUT = pt.save_mol(residue, os.path.join(OUT, f'5_{os.path.splitext(os.path.basename(monomer_file))[0]}'))
pt.save_mol(acpype_residue, os.path.join(OUT, f'5_1_{os.path.splitext(os.path.basename(monomer_file))[0]}'))
print('SDF остатка без HW для RTP:', OUTPUT)
print('SDF с HW только для acpype:', os.path.join(OUT, f'5_1_{os.path.splitext(os.path.basename(monomer_file))[0]}.sdf'))
''', 'save')
write('2_acpype.ipynb')


# Notebook 3: forcefield files and GROMACS validation.
code('''%load_ext autoreload
%autoreload 2
import json
import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import ResParTools as pt
''', 'imports')
md('''# 3. RTP, силовое поле и тестовая МД

В этом шаге создаются явные файлы силового поля: RTP, HDB, таблица типов и недостающие параметры.
Сначала проверьте содержимое RTP; затем подготовьте и запустите GROMACS в окружении `gmx`.''', 'intro')
code('''# Параметры модификации (заполняются генератором)
FOLDER = 'Lysine_prop'
PTM_name = 'Lysine_prop'
base_name = 'KPR'
monomer_file = 'molecules/K_prop.smiles'
OUT = os.path.abspath(os.path.join('..', 'build', 'notebooks', FOLDER))
INPUT = os.path.join(OUT, f'5_{os.path.splitext(os.path.basename(monomer_file))[0]}.sdf')
FF_OUT = os.path.abspath(os.path.join('..', 'build', 'amber14sb_mod_smi.ff'))
MD_OUT = os.path.abspath(os.path.join('..', 'build', 'md_from_smiles', FOLDER))
MD_NS = 0.01  # тестовая динамика 10 пс, кроме EM/NVT/NPT
run_gromacs_now = False
''', 'params')
md('''## Шаг 1. Загрузить молекулу с зарядами и типами

Имена, заряды Espaloma/RESP и типы acpype должны остаться привязаны к тем же атомам.''', 's1-md')
code('''residue = pt.load_mol(INPUT)
assert residue.GetProp('ChargeMethod') in ('Espaloma', 'RESP')
assert all(a.HasProp('AcpypeType') for a in residue.GetAtoms())
print('Используемые заряды:', residue.GetProp('ChargeMethod'))
''', 's1')
md('''## Шаг 2. Собрать RTP и силовое поле

`write_forcefield` пишет отдельную запись `<base_name>.rtp`, `<base_name>.hdb`, `residuetypes.dat`,
`atomtypes.atp` и нужные дополнения `ffnonbonded.itp`. Выход находится в `build/`, исходное
силовое поле не редактируется.''', 's2-md')
code('''rtp_path, explicit = pt.write_forcefield(residue, FF_OUT)
residue_sdf = pt.save_mol(residue, os.path.join(OUT, f'6_{os.path.splitext(os.path.basename(monomer_file))[0]}'))
print('RTP:', rtp_path)
print('Поле:', FF_OUT)
print('Явные bonded-параметры:', {key: len(value) for key, value in explicit.items()})
with open(rtp_path) as f:
    print(f.read())
print('Файлы .ff:', sorted(os.listdir(FF_OUT)))
''', 's2')
md('''## Шаг 3. Подготовить тест системы и выполнить проверку GROMACS

Входная модель — ACE–остаток–NME, чтобы проверить связь модифицированного остатка с соседями.
`pdb2gmx`, `grompp`, EM и 10-пс MD запускаются через пакетную функцию. По умолчанию создаётся
папка задания; включите `run_gromacs_now` для запуска локально или используйте сгенерированный
SLURM-скрипт на кластере.''', 's3-md')
code('''monomer = pt.open_mol(monomer_file)
job = pt.write_validate_job(residue, monomer, FF_OUT, MD_OUT, md_ns=MD_NS)
print('Проверочная система:', job)

if run_gromacs_now:
    report = pt.run_validate(job, run='local', n_threads=4)
    print(json.dumps(report, ensure_ascii=False, indent=2))
else:
    print('Локально: включите run_gromacs_now в окружении gmx и выполните ячейку.')
    print('Кластер: pt.run_validate(job, run="slurm", slurm={...}); затем sbatch run_validate.sbatch.')
''', 's3')
md('''## Результат

Силовое поле: `build/amber14sb_mod_smi.ff/<base_name>.rtp` и согласованные файлы `.ff`.
Отчёт и входные файлы теста: `build/md_from_smiles/<модификация>/`. Успешный запуск GROMACS не заменяет
проверку имён атомов, зарядов, RTP и границ `-C`/`+N`.''', 'result')
write('3_forcefield_md.ipynb')
