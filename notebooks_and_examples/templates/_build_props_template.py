"""Собирает templates/1_charge_calculation_props.ipynb (версия без словарей)."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = []
def md(text, cid): cells.append({'cell_type': 'markdown', 'id': cid, 'metadata': {}, 'source': text.strip('\n')})
def code(text, cid): cells.append({'cell_type': 'code', 'id': cid, 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.strip('\n')})

code('''
%load_ext autoreload
%autoreload 2
import os
import sys
sys.path.append(os.path.abspath('../../espaloma-charge_mod/'))
sys.path.append(os.path.abspath('../'))
import ResParTools as pt
from espaloma_charge import charge
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG = True
''', 'imports')
md('''
# Расчёт зарядов модифицированного остатка - версия без словарей

Тот же протокол, что в `1_charge_calculation.ipynb`, но все данные хранятся **в самой молекуле RDKit**
(схема - `docs/MOL_PROPS.md`):

| Свойство | у чего | шаг | что значит |
|---|---|---|---|
| `ModName`, `SourceFile` | молекула | 1 | имя и файл исходной молекулы |
| `SrcIdx` | атом | 1 | индекс атома в исходном файле |
| `PolymerIdx` | атом | 2 | индекс атома в тримере |
| `PrevIdx`, `ParentName` | атом | 3 | индекс до перенумерации; имя атома в родительском остатке (у новых атомов свойства нет) |
| `Parent`, `RefResidue`, `Naming` | молекула | 3 | родитель, его шаблон, способ имён |
| `AtomName` | атом | 3-4 | имя атома в остатке |
| `AddedH` | атом | 4.1 | водород, добавленный только для acpype |
| `Charge<Метод>`, `ChargeFixed` | атом | 5-6 | заряды каждого метода (`ChargeEspaloma`, `ChargeRESP`, ...), фиксированные атомы остова |
| `ChargeMethod` | молекула | 5-6 | метод, чьи заряды пойдут в силовое поле |

Между шагами молекула сохраняется одним файлом: `pt.save_mol` / `pt.load_mol` (SDF со всеми свойствами).
Функции, которые раньше возвращали словари (`file_opener`, `match_mon_to_pol`), здесь заменены на
`pt.open_mol` и `pt.find_residue` - они возвращают одну молекулу.
''', 'intro')
code('''
# Параметры модификации
PTM_name = 'Lysine_Formyl'
base_name = 'KFO'
parent_residue = 'K'
monomer_file = 'molecules/K_form.smiles'
trimer_file = 'molecules/AK_formA.smiles'
naming = 'greek'
resp_run = 'local'
run_resp_now = False          # True - запустить RESP в этом ноутбуке (долго: около часа на лизин)
# куда писать результаты (вне git; окончательное место - обсуждается)
OUT = os.path.join('..', 'build', 'props', os.path.basename(os.getcwd()))
os.makedirs(OUT, exist_ok=True)
''', 'params')
md('## Шаг 1. Мономер и тример\n`pt.open_mol` - одна молекула; у атомов `SrcIdx`, у молекулы `ModName`, `SourceFile`.', 's1-md')
code('''
monomer = pt.open_mol(monomer_file)
trimer = pt.open_mol(trimer_file)
mon_name = monomer.GetProp('ModName')
print(mon_name, monomer.GetNumAtoms(), '|', trimer.GetProp('ModName'), trimer.GetNumAtoms())
''', 's1')
md('## Шаг 2. Остаток внутри тримера\n`pt.find_residue` возвращает остаток (атомы мономера); у каждого атома - `PolymerIdx`, его индекс в тримере.', 's2-md')
code('''
residue = pt.find_residue(monomer, trimer, timeout=5)
in_trimer = [a.GetIntProp('PolymerIdx') for a in residue.GetAtoms()]
in_monomer = [a.GetIntProp('SrcIdx') for a in residue.GetAtoms()]
pt.draw_mol_grid([monomer, trimer], legends=[f'мономер: {mon_name}', 'тример'],
                 highlight_atoms=[in_monomer, in_trimer])
''', 's2')
md('## Шаг 3. Перенумерация и имена атомов\nАтомы родительского остатка получают `ParentName`; список `parent_atoms` больше не хранится - он вычисляется из свойств.', 's3-md')
code('''
residue = pt.renumber_residue_atoms(residue, ref_base_name=parent_residue, timeout=5)
parent_atoms = [a.GetIdx() for a in residue.GetAtoms() if a.HasProp('ParentName')]
residue = pt.rdkit_pdb_modification(residue, resname=base_name, resid=2, segid='A',
                                    naming=naming, parent_atoms=parent_atoms)
pt.draw_molecule(residue, highlight_atoms=parent_atoms, show_names=True, show_index=False)
''', 's3')
md('## Шаг 4. Правка имён, проверка дубликатов\n`rename_map = {\'старое имя\': \'новое имя\'}`; пустой - имена не меняются.', 's4-md')
code('''
rename_map = {}
residue = pt.modifie_residue_info(residue, rename_map, resname=base_name, resid=2, segid='A')
duplicates = pt.check_duplicate_atom_names(residue)
if duplicates:
    raise ValueError(f'Повторяющиеся имена атомов (имя: индексы): {duplicates}')
''', 's4')
code('''
# таблица атомов: всё, что известно об атоме, - в его свойствах
for a in residue.GetAtoms():
    print(f"{a.GetIdx():3d} {a.GetProp('AtomName'):5s} родитель: {a.GetProp('ParentName') if a.HasProp('ParentName') else '-':5s} "
          f"исходный индекс: {a.GetIntProp('SrcIdx'):3d} в тримере: {a.GetIntProp('PolymerIdx'):3d}")
''', 's4-table')
md('### Шаг 4.1. Водороды для acpype\nДобавленные водороды помечены свойством атома `AddedH` (на шаге 3 силового поля удаляются по нему, а не по имени).', 's41-md')
code('''
h_residue = pt.add_protons_and_renumber_H(residue, resname=base_name, resid=2, segid='A')
added = [a.GetIdx() for a in h_residue.GetAtoms() if a.HasProp('AddedH') and a.GetBoolProp('AddedH')]
pt.draw_molecule(h_residue, highlight_atoms=added, show_names=True, show_index=False)
''', 's41')
md('## Шаг 5. Заряды Espaloma\nЗаряды записываются в атомы: `ChargeEspaloma`; фиксированные атомы остова - `ChargeFixed`.', 's5-md')
code('''
constraints = pt.charge_constraints_from_rtp(residue, pt.AMINO_ACIDS[parent_residue][0].upper())
q = charge(residue, constraints=constraints)
pt.set_charges(residue, q, 'Espaloma', constraints)
total = Chem.GetFormalCharge(residue)
print(f'сумма зарядов: {q.sum():.5f} (заряд остатка {total})')
if abs(q.sum() - total) > 1e-3:
    raise ValueError('Сумма зарядов не равна заряду остатка.')
pt.draw_molecule(residue, charge_list=q, show_names=False, show_index=False)
''', 's5')
md('## Шаг 6. RESP (по желанию)\nЗадача готовится в `OUT/resp`; считать - `run_resp_now = True` или на кластере. Готовые заряды - в `ChargeRESP`.', 's6-md')
code('''
resp_folder = pt.prepare_resp_job(residue, constraints, f'{mon_name}_capped', working_dir=os.path.join(OUT, 'resp'),
                                  fix_backbone=True, n_sidechain=3, monomer=monomer)
if run_resp_now:
    pt.run_resp(resp_folder, run=resp_run, n_threads=4, n_parallel=1)
# готовый результат: этой задачи или ночного расчёта (build/resp/<папка>/)
done = [f for f in (resp_folder, os.path.join('..', 'build', 'resp', os.path.basename(os.getcwd()), f'{mon_name}_capped'))
        if os.path.exists(os.path.join(f, 'result.json'))]
if done:
    pt.set_charges(residue, pt.load_resp_result(done[0]), 'RESP', constraints)
    pt.compare_charges(residue, {m: [a.GetDoubleProp(f'Charge{m}') for a in residue.GetAtoms()]
                                 for m in ('Espaloma', 'RESP')})
residue.SetProp('ChargeMethod', 'Espaloma')   # заряды для силового поля
''', 's6')
md('## Сохранение\nОдин файл SDF со всеми свойствами (`pt.load_mol` читает его обратно) и 3D PDB для acpype.', 'save-md')
code('''
path = pt.save_mol(residue, os.path.join(OUT, f'4_{mon_name}'))
pt.save_mol(h_residue, os.path.join(OUT, f'4_1_{mon_name}'))
pt.save_chem_to_pdb(h_residue, os.path.join(OUT, f'{PTM_name}_rn_H_3D'), coords='3D')
back = pt.load_mol(path)
print('свойства молекулы:', back.GetPropsAsDict())
print('свойства атома 0:', back.GetAtomWithIdx(0).GetPropsAsDict())
''', 'save')
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'darwin_ec', 'language': 'python', 'name': 'python3'},
      'language_info': {'name': 'python'}, 'respartools': {'template': 'templates/1_charge_calculation_props.ipynb'}},
      'nbformat': 4, 'nbformat_minor': 5}
for c in nb['cells']:
    lines = c['source'].split('\n'); c['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
json.dump(nb, open(os.path.join(HERE, '1_charge_calculation_props.ipynb'), 'w'), ensure_ascii=False, indent=1)
