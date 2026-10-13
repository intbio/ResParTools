# ResParTools: руководство по проекту

Документ для человека, который видит проект впервые. После него должно быть понятно: какую
задачу решает проект, из каких частей он состоит, что делает каждая функция библиотеки, как
пройти путь от SMILES модификации до проверенного силового поля и что изменилось с тех пор,
как вся библиотека была одним файлом `ResParTools.py`.

Состояние описано на ветке `integration/night-2026-10-10` (2026-10-11). Части, которые есть
только в ночных ветках и ещё не в `main`, помечены **[ветки]**.

Другие документы (подробности, которые здесь только упоминаются):
- `docs/PIPELINE.md` - короткая карта пайплайна (таблицы шагов и файлов);
- `docs/MOL_PROPS.md` - полная схема свойств молекулы;
- `docs/REVIEW.md` - ревью кода и ошибок шагов 2-3;
- `docs/PLAN.md` - статус, решения, следующие шаги;
- `docs/NIGHT_REPORT.md` - отчёт о ночной работе 2026-10-10;
- `notebooks_and_examples/cluster/README.md` - RESP на кластере.

---

## 1. Задача

Белки в молекулярной динамике (GROMACS) описываются силовым полем, у нас - **amber14sb**.
В поле есть 20 стандартных аминокислот, а нам нужны **модифицированные**: посттрансляционные
модификации лизина (метилирование, ацетилирование, формилирование, кротонилирование и т. д.)
и цистеин с пришитой флуоресцентной меткой (Alexa Fluor 546/647). Для каждой такой
модификации силовому полю нужно сообщить:

1. **Имена атомов** - по правилам amber (N, CA, CB, HB1...), чтобы `pdb2gmx` узнал остаток в PDB.
2. **Типы атомов** - метки, по которым поле находит параметры связей, углов, диэдралов и
   Леннард-Джонса (CT, C8, N, H1...).
3. **Парциальные заряды** - у каждого атома; сумма равна заряду остатка, а заряды остова
   (N H CA HA C O) - как у родительской аминокислоты, иначе остаток «выпадает» из белка.
4. **Связи и параметры**, которых нет в поле (например, связь CE-NZ у формиллизина, где NZ
   стал амидным азотом).
5. **Правила достройки водородов** (`.hdb`) - `pdb2gmx` строит H заново.

Проект автоматизирует всё это: на входе SMILES модификации, на выходе - файлы силового поля
и отчёт о проверке в GROMACS.

### Словарь
| Термин | Что это |
|---|---|
| остаток (residue) | одна аминокислота внутри цепи белка; у неё нет концевых H2N-/-COOH, вместо них связи с соседями |
| мономер | модифицированная аминокислота как отдельная молекула (SMILES в `molecules/`) |
| тример | та же аминокислота между двумя соседями (A-X-A или G-X-G); из него видно, какие атомы мономера остаются в цепи |
| родитель (parent) | стандартная аминокислота, которую модифицировали: K (лизин), C (цистеин) |
| шаблон родителя | `molecules/aminoacids_template/G<X>G_H.pdb` - трипептид с именами атомов amber; по нему ищется родитель |
| кэпы ACE / NME | ацетил на N-конце и N-метиламид на C-конце: пептид ACE-X-NME - стандартная модель остатка в цепи для зарядов и проверки |
| протоны HW | водороды, которые достраиваются на «оборванные» N и C остатка только для acpype (он не принимает неполную валентность); в поле не попадают |
| Espaloma | нейросеть, предсказывающая заряды (AM1-BCC-подобные) за секунды; у нас модифицированная копия умеет фиксировать заряды выбранных атомов |
| RESP | классический метод зарядов amber: квантовая химия (HF/6-31G*) → электростатический потенциал → подгонка зарядов; часы на остаток |
| ограничения (constraints) | атомы с заранее заданным зарядом (остов из rtp родителя); остальной заряд распределяется по свободным атомам |
| acpype | программа (на основе antechamber AmberTools), которая даёт типы атомов и параметры для молекулы |
| `.rtp` | файл остатков поля: атомы (имя, тип, заряд, группа), связи, явные параметры |
| `.hdb` | правила достройки водородов для `pdb2gmx` |
| `atomtypes.atp` | список типов атомов и их масс |
| `ffbonded.itp` / `ffnonbonded.itp` | параметры связей/углов/диэдралов по типам атомов / Леннард-Джонс по типам |
| `residuetypes.dat` | какие остатки GROMACS считает белком |
| `.r2b` | соответствие имён остатков в PDB и в rtp (для концевых и протонированных форм) |
| `.itp` | топология молекулы GROMACS (выход acpype) |
| свойства RDKit | значения, которые хранятся в самой молекуле RDKit или в её атоме (`atom.SetProp('AtomName', 'CB')`) |
| SDF | формат файла молекулы, который хранит координаты, связи и свойства |
| MCS | максимальная общая подструктура двух молекул (RDKit `rdFMCS`) - так мономер находится в тримере |

---

## 2. Структура репозитория

```
ResParTools/
├── README.md, CLAUDE.md (локально), docs/        документация
├── ResParTools_ec.yml, psiresp_min.yml, ...    окружения conda
├── espaloma-charge/                            исходная Espaloma Charge (не трогается)
├── espaloma-charge_mod/                        наша копия: charge(mol, constraints={...})
├── tests/                                      тесты на эталонном наборе (pytest)
└── notebooks_and_examples/                     рабочая папка проекта
    ├── ResParTools.py                          точка входа: import ResParTools as pt
    ├── respartools/                            библиотека (пакет, раздел 6)
    ├── templates/                              шаблоны ноутбуков
    ├── make_charge_notebooks.py                сборка ноутбуков из шаблона
    ├── cluster/                                RESP на кластере SLURM
    ├── molecules/                              общие молекулы: aminoacids_template/ (шаблоны родителей), трипептиды
    ├── L_aminoacids/                           стандартные аминокислоты (старые проверки hdb)
    ├── aminoacids_dict.json                    разобранный aminoacids.rtp (старый шаг 3)
    ├── amber14sb_parmbsc1_cufix.ff/            базовое силовое поле (раздел 10)
    ├── amber14sb_mod.ff/                       поле с модификациями (старый путь)
    ├── 3_edd_topology.ipynb                    старый шаг 3 (общий для всех папок)
    ├── 4_bild_tri_peptid.ipynb, psiresp.ipynb, Espaloma_ai.ipynb, base_analysis.ipynb,
    │   datasets_parsing.ipynb                  ранние исследовательские ноутбуки
    ├── Lysine_1M, Lysine_2M, ..., Lysine_prop  12 модификаций лизина
    ├── AF_546_cys, AF_647_cys                  цистеин с метками Alexa Fluor
    └── build/                                  [ветки] результаты ночных расчётов, вне git
```

Папка модификации (на примере `Lysine_Formyl/`):

| Файл | Что это |
|---|---|
| `molecules/K_form.smiles`, `AK_formA.smiles` | вход: мономер и тример |
| `molecules/K_form_capped.smiles` | ACE-X-NME для RESP |
| `molecules/substructure/3_*.pdb`, `3_1_*`, `3_2_*`, `4_*`, `4_1_*` | промежуточные файлы шагов 2.2-4.1 (лежат в git, по ним работают тесты) |
| `general_atoms.json` | какие атомы остатка - атомы родителя (старый путь) |
| `AI_chrges_<имя>.json` | заряды Espaloma (шаг 5) |
| `RESP_chrges_<имя>.json`, `RESP_data/` | заряды RESP и задача RESP (шаг 6; `RESP_data` вне git) |
| `1_charge_calculation.ipynb` | новый ноутбук шагов 1-6 (собран из шаблона) |
| `1_charge_calculation_props.ipynb` | [ветки] он же без словарей |
| `1_charge_calculation_<...>.ipynb`, `*_old.ipynb` | старые ноутбуки - **не редактируются**, это запись того, как работало раньше |
| `2_generate_topology*.ipynb`, `Acpype_data/` | старый шаг 2 (acpype) |
| `force_field_files/` | выход старого шага 3 |
| `logs/` | лог отладки (`pt.start_log()`), вне git |

Эталонный набор (на нём проверяется любое изменение): `Lysine_prop`, `Lysine_Malonyl`,
`Lysine_Formyl`, `Lysine_3M` (лизины разной сложности) и `AF_546_cys` (большая метка на
цистеине).

---

## 3. Окружения и внешние программы

| Окружение | Файл | Python | Что в нём | Для чего |
|---|---|---|---|---|
| `darwin_ec` | `ResParTools_ec.yml` | 3.12 | RDKit, torch, dgl, Espaloma, Jupyter | ноутбуки, шаги 1-5 |
| `darwin_resp` | `psiresp_min.yml` (снимок `darwin_resp.lock.yml`) | 3.9 | psi4 1.6.1, psiresp 0.4.2 | RESP (шаг 6), тесты |
| `topmol2` | нет yml (локальное) | 3.8 | acpype, parmed, AmberTools | шаг 2 (acpype) |
| `gmx` | `conda create -n gmx -c conda-forge gromacs` | - | GROMACS 2026.3 | [ветки] проверка поля |
| `espaloma` | `ai_topmol.yml` | 3.9 | старое окружение Espaloma | старые ноутбуки |

Почему окружений несколько: psi4 1.6.1 существует только для Python ≤ 3.10 и требует старых
библиотек, psiresp 0.4.2 - pydantic 1, а новые psi4 - pydantic 2; acpype стоит в отдельном
старом окружении. Поэтому **ноутбук всегда работает в `darwin_ec`**, а функции сами
запускают нужную программу в соседнем окружении отдельным процессом:
`run_resp(run='local')` → `darwin_resp`, `run_acpype` → `topmol2`, `run_validate` → `gmx`
(ищутся в `anaconda3/envs/<имя>` рядом с текущим окружением).

Установка `darwin_ec` на машине без NVIDIA: `CONDA_OVERRIDE_CUDA=12.4 conda env create -n darwin_ec -f ResParTools_ec.yml`,
затем обязательно `conda remove -n darwin_ec --force -y espaloma_charge` - иначе вместо
нашей копии Espaloma подхватится исходная (симптом: `charge() got an unexpected keyword argument 'constraints'`).

---

## 4. Пайплайн целиком

### 4.1 Шаги

```
 molecules/<мономер>.smiles, <тример>.smiles
   │
 1 чтение                      file_opener / open_mol
 2 мономер в тримере (MCS)     match_mon_to_pol / find_residue      → какие атомы остаются в цепи
 3 перенумерация + имена       renumber_residue_atoms, rdkit_pdb_modification
 4 проверка, правка имён       modifie_residue_info, check_duplicate_atom_names
 4.1 протоны HW для acpype     add_protons_and_renumber_H
   │
   ├─ 5 заряды Espaloma        charge_constraints_from_rtp → charge(..., constraints) → set_charges
   ├─ 6 заряды RESP            prepare_resp_job → run_resp (локально / SLURM) → load_resp_result
   │
 2* acpype: типы и параметры   run_acpype                [ветки; раньше - ноутбук 2_generate_topology]
 3* силовое поле               write_forcefield          [ветки; раньше - 3_edd_topology.ipynb]
 5* проверка в GROMACS         write_validate_job → run_validate → validate_report   [ветки; раньше не было]
   │
 папка .ff + отчёт проверки
```

Нумерация исторически двойная: шаги 1-6 - это ноутбук `1_charge_calculation`, а «шаг 2» и
«шаг 3» без уточнения обычно означают ноутбуки `2_generate_topology` и `3_edd_topology`
(здесь они помечены звёздочкой).

### 4.2 Два способа хранить данные

**Старый (сейчас в `main`): словари и файлы.** Функции принимают и возвращают словари
`{имя файла: молекула}` (декоратор `@data_to_dict`), а сведения о молекуле разнесены по
словарям сопоставления (`match_dict` из 5 словарей), спискам индексов (`ParentAtoms`,
`OldIndices`) и файлам (`general_atoms.json`, `AI_chrges_*.json`, PDB на каждом шаге).
Шаги 2* и 3* - отдельные ноутбуки в каждой папке, параметры вписываются руками.

**Новый [ветки]: всё в самой молекуле.** Каждый атом несёт свои сведения как свойства
RDKit (`AtomName`, `ParentName`, `ChargeEspaloma`, `AtomType`...), молекула - свои
(`ResName`, `Parent`, `ChargeMethod`...). Функции принимают одну молекулу и возвращают её же.
Между шагами - один файл SDF (`save_mol` / `load_mol`). Это нужно и для понятности кода,
и для будущего веб-сервиса (одна молекула = один запрос).

Старые функции не удалены: старые ноутбуки работают как раньше (тест это проверяет).

### 4.3 Свойства молекулы (кратко; полностью - `docs/MOL_PROPS.md`)

| Свойство | У чего | Кто пишет | Смысл |
|---|---|---|---|
| `ModName`, `SourceFile` | молекула | `file_opener` | имя и файл исходной молекулы |
| `SrcIdx` | атом | `file_opener` | индекс атома в исходном файле |
| `PolymerIdx` | атом | `find_residue`, `match_chem_v6` | индекс атома в тримере |
| `PrevIdx` | атом | `renumber_residue_atoms` | индекс до перенумерации |
| `ParentName` | атом | `renumber_residue_atoms` | имя атома в родителе (у новых атомов свойства **нет**) |
| `Parent` | молекула | `renumber_residue_atoms` | буква родителя (`K`) |
| `AtomName` | атом | `rdkit_pdb_modification` | имя атома в остатке |
| `Naming` | молекула | `rdkit_pdb_modification` | `greek` / `index` |
| `ResName`, `ResId`, `Chain` | молекула | `save_mol` | параметры остатка PDB (SDF их не хранит) |
| `AddedH` | атом | `add_protons_and_renumber_H` | водород HW, только для acpype |
| `Charge<Метод>` | атом | `set_charges` | `ChargeEspaloma`, `ChargeRESP`, будущие методы |
| `ChargeFixed` | атом | `set_charges` | заряд атома зафиксирован (остов) |
| `ChargeMethod` | молекула | `set_charges` | чьи заряды пойдут в поле |
| `AcpypeType`, `AcpypeItp` | атом / молекула | `run_acpype` | тип атома по acpype, путь к itp |
| `AtomType`, `TypeSource` | атом | `assign_types` | тип для поля и откуда он (`parent` / `acpype`) |

Что надо знать про свойства RDKit (проверено):
- свойства атомов переживают копирование, `RemoveHs`, `RemoveAtom`, `RenumberAtoms`;
- свойства **молекулы** теряются при `Chem.RenumberAtoms` - наши функции переносят их сами;
- в SDF свойство атома пишется одной строкой через пробел: пустое значение или значение с
  пробелом сдвигает строку, и свойство молча теряется - поэтому у новых атомов нет
  `ParentName`, а `save_mol` отказывается писать такие значения;
- из SDF всё читается строками - `load_mol` приводит типы обратно.

---

## 5. Как пройти путь от SMILES до поля

### 5.1 Новая модификация в старом ноутбуке (то, что в `main`)
1. Положить `molecules/<мономер>.smiles` и `<тример>.smiles` в новую папку.
2. Добавить строку в `PARAMS` в `make_charge_notebooks.py` (имя модификации, трёхбуквенное
   имя остатка, родитель, файлы, способ имён) и выполнить `python make_charge_notebooks.py <папка>`.
3. Открыть `<папка>/1_charge_calculation.ipynb` (ядро `darwin_ec`), выполнить шаги 1-5,
   смотреть на картинки и таблицы имён после шагов 3 и 4.
4. RESP: шаг 6 ноутбука (`resp_run='local'` - долго) или на кластере (раздел 8).
5. Шаги 2* и 3* - старыми ноутбуками (или новым путём ниже).

### 5.2 Без словарей, до проверенного поля [ветки]
Ноутбук: `python make_charge_notebooks.py --props <папка>` → `<папка>/1_charge_calculation_props.ipynb`
(шаги 1-6 и сохранение SDF). Шаги 2*, 3* и проверка пока выполнялись ночными скриптами
(`build/md_code/`); в коде это выглядит так:

```python
import ResParTools as pt
mon = pt.open_mol('molecules/K_form.smiles')
tri = pt.open_mol('molecules/AK_formA.smiles')
res = pt.find_residue(mon, tri, timeout=5)                         # шаг 2
res = pt.renumber_residue_atoms(res, ref_base_name='K')            # шаг 3.1
res = pt.rdkit_pdb_modification(res, resname='KFO', resid=2, naming='greek',
          parent_atoms=[a.GetIdx() for a in res.GetAtoms() if a.HasProp('ParentName')])  # 3.2
cons = pt.charge_constraints_from_rtp(res, 'LYS')                  # шаг 5
pt.set_charges(res, charge(res, constraints=cons), 'Espaloma', cons)
pt.run_acpype(res, 'build/acpype/Lysine_Formyl')                   # шаг 2*
pt.write_forcefield(res, 'build/amber14sb_mod.ff')                 # шаг 3*
job = pt.write_validate_job(res, mon, 'build/amber14sb_mod.ff', 'build/validate/KFO', md_ns=0)
pt.run_validate(job, run='local')                                  # проверка
```

---

## 6. Библиотека `respartools/` - модули и функции

### 6.0 Как устроен пакет
- `notebooks_and_examples/ResParTools.py` - точка входа. Он собирает **все** имена всех
  модулей (включая служебные `_...`) в одно пространство, поэтому в ноутбуках
  `import ResParTools as pt; pt.<функция>` - как в те времена, когда библиотека была одним
  файлом. Старые ноутбуки работают без правок.
- `respartools/__init__.py` - список модулей `MODULES` и импорт.
- Порядок импортов внутри пакета: `log` и `utils` - в начале модуля, остальные модули пакета
  - **в конце** модуля (иначе взаимные ссылки ломают импорт).
- Код совместим с Python 3.8 (окружение `topmol2`): без `match` и аннотаций `X | Y`.
- Два декоратора встречаются почти везде:
  - `@data_to_dict` (utils): функция принимает путь, список или словарь и возвращает
    `{имя: результат}`. Наследие однофайловой версии; новые функции его не используют.
  - `@logged` (log): если включён лог отладки, записывает вызов функции, параметры,
    молекулы, записанные файлы. Без `start_log()` ничего не делает.

Размер: 12 модулей, ~6 300 строк, 179 функций (106 публичных); в `main` - ~5 400 строк.

Ниже модули идут в порядке пайплайна. Для служебных функций (`_...`) - только назначение.

### 6.1 `fileio` - чтение и запись молекул
**Чтение**
- `file_opener(path, **kwargs)` - главный вход шага 1: открывает `.smi/.smiles`, `.pdb`,
  `.mol2` (формат по расширению), путь или список путей → `{имя: молекула}`. Ставит
  свойства `ModName`, `SourceFile`, у атомов `SrcIdx` [ветки].
- `open_mol(path)` [ветки] - то же для одного файла, без словаря; `.sdf` читается через `load_mol`.
- `read_file(path)` - первая строка файла (SMILES).
- `smi_to_chem(str_smi, sanitize, addH, make_N_root, format_coord='2D')` - SMILES → молекула
  с водородами и 2D-координатами. Понимает SMILES с номерами атомов (`_mol_from_mapped_smiles`:
  порядок атомов из номеров, водороды как в строке - так сохраняется порядок атомов остатка).
- `pdb_to_chem(path, removeHs, sanitize, format_coord='2D')` - PDB → молекула; имена атомов
  и остаток - в свойствах атомов, связи из CONECT.
- `mol2_to_chem(path, ...)` - mol2 → молекула; порядки связей из секции BOND, имя/тип/заряд
  атома и номер остатка из секции ATOM (`_read_mol2_atom_block`).
- `_finalize_read_mol` - общий конец чтения: sanitize, удаление H, координаты.

**Запись**
- `save_chem_to_smiles(mol, path, canonical, allHsExplicit, rootedAtAtom, atom_map=False)` -
  SMILES; `atom_map=True` пишет номер каждого атома, чтобы при чтении восстановить порядок.
- `save_aa_chem_to_smiles(...)` - старое имя; `make_N_root=True` - строка начинается с N
  аминогруппы (`find_amino_nitrogen`).
- `save_chem_to_pdb(mol, path, coords='2D', stereo, ca_config='L', ...)` - PDB с порядком
  атомов молекулы; `coords='3D'` строит 3D с нужной стереохимией (`set_mol_coords`).
- `save_aa_chem_to_pdb(mol, path, resname, resid, segid, naming, parent_atoms)` - остаток в PDB
  с именами атомов (вызывает `rdkit_pdb_modification` на копии); пишет `<path>.pdb` и `<path>_no_H.pdb`.
- `save_chem_to_mol(mol, path)` - формат MOL.
- `save_charges_json(name, charges, path)` - список зарядов в JSON (`AI_chrges_*.json`).
- `save_mol(mol, path)` [ветки] - SDF со **всеми** свойствами атомов и молекулы; проверяет,
  что значения не пустые и без пробелов.
- `load_mol(path)` [ветки] - чтение `save_mol`: типы свойств восстанавливаются
  (`_INT_PROPS`, `_BOOL_PROPS`, `Charge*` → float), PDB-информация остатка - из `AtomName`,
  `ResName`, `ResId`, `Chain`.
- Устаревшие, оставлены для старых ноутбуков: `smi_to_mol2` (через obabel, не вызывается),
  `write_modified_molecule`, `save_molecule_as_pdb`; `_reroot_via_smiles` - старое поведение
  `rootedAtAtom` с предупреждением.

### 6.2 `matching` - мономер в тримере, родительский остаток
- `match_mon_to_pol(monomer_dict, polymer_dict, timeout, only_heavy_mapping, ...)` - шаг 2:
  находит мономер в тримере (MCS) и возвращает словарь `match_dict` (`_MatchData`):
  подструктуры с H и без, сопоставление индексов `mon_pol_matches`, имена. Старые ключи
  (`substructure_noH`) перенаправляются на новые.
- `find_residue(monomer, polymer, timeout=5)` [ветки] - то же без словарей: возвращает
  остаток (атомы мономера, которые есть в тримере) с `PolymerIdx` у атомов. Атомам на месте
  разрыва (N и C, потерявшим связь с концевыми группами) возвращает неявные H - иначе RDKit
  оставлял N двухвалентным (так было у Benzoyl и lac).
- `match_chem_v6(mol1, mol2, only_heavy_mapping=True, timeout=30)` - рабочая версия MCS:
  сначала тяжёлые атомы, водороды сопоставляются по ним. `match_chem` и `match_chem_v3` -
  ранние версии (`match_chem_v3` не вызывается). `remove_hydrogens_preserve_indices` -
  молекула без H с картой индексов.
- `REF_TEMPLATES` - где лежат шаблоны родителей (`molecules/aminoacids_template/*_H.pdb`,
  остаток №2, атомы остова N CA C O).
- `AMINO_ACIDS` - `{буква: (Xxx, русское имя, английское имя)}` для 20 аминокислот.
- `aa_label('C')` → «цистеин (Cys, C)»; `template_aa_letter('GCG_H')` → `C`;
  `template_label` → «цистеин (Cys, C), шаблон GCG_H» (так аминокислоты называются в сообщениях).
- `resolve_ref_template(ref_base_name, paths)` - файл шаблона по любому имени родителя
  (`C`, `Cys`, `cysteine`, `GCG_H`).
- `find_ref_residue(mod_mol_dict, ref_base_name, ...)` - сопоставляет остаток с шаблоном
  родителя. Родитель указывается **явно**; старый вызов без него выбирает шаблон по числу
  совпавших атомов (ненадёжно) и предупреждает.
- `check_parent_residue(...)` - остов и CB должны совпасть с родителем полностью, иначе ошибка;
  несовпавшие атомы боковой цепи печатаются (это и есть модификация).

### 6.3 `residue` - перенумерация, имена, координаты, протоны
- `renumber_residue_atoms(mol, ref_base_name, ...)` - шаг 3.1: атомы родителя получают номера
  в порядке шаблона, новые атомы - обходом графа от родителя. Пишет `PrevIdx`, `ParentName`,
  `Parent` [ветки] и старые списки `ParentAtoms`/`OldIndices`.
- `rdkit_pdb_modification(mol, resname, resid, segid, naming='greek', parent_atoms, C_terminal)`
  - шаг 3.2, имена атомов:
  - остов `N CA C O` (C-конец - `OC1 OC2`), найденный `find_backbone_match`;
  - `greek`: тяжёлые атомы боковой цепи = элемент + буква уровня (число связей от CA: B G D E
    Z H T I K L M N) + номер, если на уровне их несколько: `CB`, `CD1`; водороды - `H` + имя
    без элемента + номер: `HB1`, `HD11`. Если уровней больше 12 или имя длиннее 4 символов -
    сообщение и переход на `index`;
  - `index` (для больших меток): атомы родителя - как в `greek`, остальные - элемент + индекс
    (`C8`, `N15`, `H122`).
  Служебные: `_greek_heavy_names`, `_hydrogen_names`, `_pdb_atom_name_field` (колонки имени в PDB).
- `find_backbone_match(mol, C_terminal)` - атомы остова по шаблону H-N-CA(H)-C(=O).
- `modifie_residue_info(mol, index_map, resname, resid, segid)` - шаг 4: ручная правка имён
  `{старое: новое}`; при совпадении имени атом получает следующее свободное (`increment_name`).
- `check_duplicate_atom_names(mol)` - повторяющиеся имена (должно быть пусто).
- `set_PDB_residue_info`, `check_PDB_residue_info`, `add_names_from_residue` - запись и
  проверка PDB-информации атома.
- `add_protons_and_renumber_H(mol, resname, ...)` - шаг 4.1: достраивает H с неполной
  валентностью (HW1, HW2 на N и C), помечает `AddedH`.
- `set_mol_coords(mol, coords='2D'|'3D', ca_config='L', stereo_default='R', random_seed)` -
  координаты: 2D однозначна; 3D строится с заданной стереохимией (CA - L по ориентированному
  объёму `_ca_volume`, не по R/S, т. к. L-цистеин - это R).
- Устаревшие: `generate_atom_names_by_ref_aa` (не вызывается), `rdkit_pdb_modification_old`.

### 6.4 `charges` - заряды
- `read_rtp_charges(resname, rtp_path)` - заряды остатка из rtp `{имя: заряд}` (через `read_rtp`).
- `charge_constraints_from_rtp(mol, rtp_residue, atom_names=N H CA HA C O)` - ограничения для
  Espaloma и RESP: атомы остова получают заряды одноимённых атомов родителя из amber14sb
  `{индекс: заряд}`. Сопоставление **по имени**, не по индексу.
- `set_charges(mol, charges, method, constraints)` [ветки] - заряды в атомы: `Charge<method>`,
  `ChargeFixed`, `ChargeMethod`.
- `group_charges(mol, charges)` - заряд группы «тяжёлый атом + его H» (так сравнивают схемы
  зарядов: разные методы по-разному делят заряд внутри CH2).
- `compare_charges(mol, {'Espaloma': q1, 'RESP': q2})` - суммы и наибольшие расхождения групп.
- `make_substructure_charge_list(...)` - старая выборка зарядов подструктуры по `match_dict`.

Сама Espaloma вызывается из ноутбука: `from espaloma_charge import charge;
charge(mol, constraints=...)` (`espaloma-charge_mod`, слой `ChargeEquilibrium_mod` фиксирует
заданные атомы и распределяет остаток заряда по свободным).

### 6.5 `resp` - заряды RESP (шаг 6)
Протокол как у зарядов amber ff94/ff14SB (Cornell 1995):
1. **Молекула.** `capped_monomer(monomer)` - ACE-X-NME из SMILES мономера со стереохимией
   мономера; `cap_residue(residue)` - кэпы к остатку шага 4 (атомы остатка остаются 0..n-1);
   `_with_reference_stereo` - порядок атомов остатка + стереохимия мономера. Для тримера
   (`context='trimer'`): `map_residue_to_context`, `context_backbone_dihedrals`.
2. **Конформации.** `resp_conformers(mol, dihedrals, residue_atoms, fix_backbone=True, n_sidechain=3)`:
   пары φ/ψ (`AMBER_BACKBONE_CONFORMATIONS`: αR −60/−40, β −120/130) × 3 конформации боковой
   цепи: ETKDG → MMFF с замороженными φ/ψ → без контакта боковой цепи с остовом
   (`_polar_contacts`) → окно 10 ккал/моль → максимально непохожие (`_sidechain_rmsd`).
   Стереохимия каждой конформации проверяется (`_check_conformer_stereo`).
3. **Задача.** `prepare_resp_job(residue, constraints, name, working_dir, monomer=...)` - папка
   задачи: `spec.json` (молекула, ограничения, настройки), `conformers/<метка>/start.xyz`.
   Квантовая химия не запускается. `review_resp_job(folder)` - сводка для проверки до запуска.
4. **Расчёт.** `run_resp(folder, run='local'|'slurm')`: локально - `run_resp_job` в окружении
   `darwin_resp` (`_resp_python`, `_run_resp_job_external`); `slurm` - пишет
   `run_resp.sbatch` (`write_slurm_script`). Внутри `run_resp_job`:
   - оптимизация HF/6-31G* в psi4 с замороженными φ/ψ, одним расчётом до 400 шагов
     (`_optimize_conformer`, `_run_parallel`); не сошлось - продолжение с `current.xyz`, упал psi4 -
     геометрия в `failed.xyz`, причина в `status.json`. `prepare_resp_job` в папке с другой задачей
     пишет новую в `<имя>_01`, `_02`... (`_job_folder`), та же задача остаётся на месте;
   - ESP на сетке MSK (psiresp, `_run_esp`);
   - двухстадийный RESP (`_fit_two_stage`): стадия 1 - все свободные атомы, стадия 2 -
     только CH2/CH3 (`_stage2_atoms`); линейно зависимые ограничения убираются
     (`_independent_constraints`); проверка суммы и ограничений (`_check_resp_charges`).
5. **Результат.** `result.json` в папке задачи; `load_resp_result(folder)` читает и проверяет.
   `resp_charges(...)` - всё сразу (подготовка + запуск).
- `prepare_cluster_check(...)` - проверка узла кластера (скорость psi4 на 1..24 потоках и
  маленькая задача RESP).
- `_main` - вход для запуска модуля отдельным процессом.

### 6.6 `forcefield` - файлы силового поля, acpype
**Чтение и запись файлов**
- `read_rtp(path)` [ветки] - **единый** парсер rtp → `(bondedtypes, {остаток: {'atoms': [(имя, тип, заряд, группа)], 'bonds': [...], ...}})`.
  Раньше в проекте было три разных парсера.
- `write_rtp(path, residues, bondedtypes)` [ветки] - запись; заряды без округления, чтение → запись → чтение ничего не меняет.
- `read_itp(path)` [ветки] - секции itp (atoms, bonds, angles, dihedrals).
- `read_ffbonded(path)` [ветки] - какие связи, углы, диэдралы есть в `ffbonded.itp` по типам (с шаблоном `X`).

**Шаг 2*: acpype**
- `run_acpype(mol, workdir, atom_type='amber', charge_type='gas')` [ветки] - топология acpype:
  вход - MOL прямо из RDKit (порядки связей, 3D; obabel не нужен), заряд - `GetFormalCharge`
  молекулы (раньше вписывался руками и местами неверно), водороды HW добавляются внутри,
  если их нет; порядок атомов itp сверяется с молекулой по элементам. Пишет `AcpypeType`, `AcpypeItp`.

**Шаг 3*: типы, rtp, hdb, поле**
- `assign_types(mol, parent_res)` [ветки] - тип атома: атом родителя сохраняет тип родителя,
  если его окружение то же (элемент, число соседей, число H; `_env`, `_rtp_env`); водород - если
  не изменилось окружение и его атома, и соседей; иначе - тип acpype. Заменяет зашитые
  `aminoacid='LYS'` и `exclusion_atoms=['HE1','HE2']` старого шага 3.
- `residue_rtp(mol, ff_dir)` [ветки] - запись остатка для rtp: атомы с типами и зарядами
  `Charge<ChargeMethod>`, все связи + `-C N`, импроперы остова; **явные параметры** (из itp
  acpype, уже в единицах GROMACS) - только для связей, углов, диэдралов, которых нет в
  `ffbonded.itp` по типам (`_dihedral_known`).
- `hdb_entries(atoms, bonds)` [ветки] - правила достройки H по графу: тип по числу H и соседей
  (1 плоский, 2 OH/SH, 3 NH2, 4 метил, 5 CH sp3, 6 CH2 sp3), управляющие атомы. Сверено с
  базовым `aminoacids.hdb` на всех стандартных остатках.
- `write_forcefield(mol, ff_out, ff_base)` [ветки] - добавляет остаток в поле: если папки нет -
  копия базы (`_copy_forcefield` пропускает rtp с неизвестными типами и мусорные файлы);
  `<ResName>.rtp` и `<ResName>.hdb` отдельными файлами, строка `residuetypes.dat` заменяется,
  недостающие типы - в `atomtypes.atp`, проверка Леннард-Джонса. Повторный запуск ничего не
  дублирует.

**Старые (для старых ноутбуков)**: `hdb_generator` (тип H по порядку индексов - ошибочно, см.
`REVIEW.md`), `check_atomtypes` (пишет atp, но не ffnonbonded), `make_r2b`, `remove_extra_H`
(удаление HW по имени из топологии parmed).

### 6.7 `gmx` - проверка в GROMACS [ветки]
- `capped_peptide(residue, monomer)` - пептид ACE-X-NME в 3D с именами атомов и
  стереохимией мономера, атомы в порядке остатков (так требует `pdb2gmx`).
- `write_validate_job(residue, monomer, ff_dir, workdir, md_ns=0)` - папка задачи:
  `peptide.pdb`, ссылка на поле, `residuetypes.dat`, mdp-файлы (`_MDP`), `run_validate.sh`.
  `md_ns > 0` добавляет МД в воде.
- `run_validate(workdir, run='local'|'slurm', n_threads=4)` - локально запускает скрипт и
  возвращает отчёт; `slurm` пишет `run_validate.sbatch`.
- Что делает скрипт: `pdb2gmx -ignh` (все H строятся заново по нашему hdb) → `grompp` →
  минимизация в вакууме (reaction-field; PME с заряженным пептидом даёт предупреждение);
  с МД: коробка, TIP3P, ионы до нейтральности, минимизация, NVT 100 пс, NPT 100 пс, МД.
  Ход - в `status.txt`.
- `validate_report(workdir)` - итог по файлам (работает и с папкой, скачанной с кластера):
  пройденные этапы, совпали ли атомы после `pdb2gmx` с исходными, предупреждения `grompp`,
  сходимость минимизации, для МД - LINCS, средние T, P, плотность, энергия, дрейф длин связей
  остатка относительно минимума в вакууме (`em_vac.gro`). Служебные `_pdb_atoms`, `_gro_atoms`.

### 6.8 `draw` - картинки (только для ноутбука)
- `draw_molecule(mol, charge_list, highlight_atoms, show_names, show_index, orient_backbone)` -
  молекула с подписями «индекс:имя: заряд»; рисуется копия, исходная не меняется; остов внизу
  (`_orient_backbone_down`). Старое имя `draw_mol_with_atom_index`.
- `draw_mol_grid(mols, legends, highlight_atoms, ...)` - несколько молекул сеткой (этапы рядом).
- `draw_mon_pol_match(...)` - мономер и тримеры с подсветкой сопоставления (старый путь).

### 6.9 `log` - лог отладки
- `start_log(folder='logs')` / `stop_log()` - включить/выключить; папка запуска
  `logs/<дата_время>/`: `log.txt` (читаемый журнал вызовов), `calls.jsonl` (то же для машины),
  `files/` (молекулы из вызовов в PDB/SDF/mol2/SVG). В заголовке - ноутбук, окружение,
  версии пакетов, ветка git, коммит, хэш кода.
- `log_note(text, **data)` - своя запись из ноутбука.
- `logged` - декоратор; остальное (`_describe*`, `_write_*`, `_StdoutTee`...) - внутренности.

### 6.10 `utils` - мелочи
`get_name` (ключ словаря по пути или SMILES), `data_to_dict`, `path_parser`, `print_green` /
`print_red`, `animate` и `format_time` (анимация долгих процессов), `rename_folder` /
`get_unique_folder_name`, `show_list_of_conf`, `increment_name`, `calculate_true_sum`,
`save_json`, `_PACKAGE_ROOT` (путь `notebooks_and_examples/`, от него строятся пути к данным).
`do_fun_for_2d_list` не вызывается.

### 6.11 `legacy` - совместимость со старыми ноутбуками
`renumber_amino_acid_atoms`, `save_substructure_mol` (кода старых версий нет - вызов
передаётся в новые функции с предупреждением); старый RESP через psiresp без оптимизации
геометрии: `add_constraint`, `prepere_constraints` (не вызывается), `run_job`, `do_sh`,
`resp_calculation` - с предупреждением `_OLD_RESP_WARNING`.

---

## 7. Ноутбуки, шаблоны, генератор

- `templates/1_charge_calculation.ipynb` - шаблон нового ноутбука шагов 1-6 (сделан из
  `AF_546_cys`). Всё, что отличает модификации, - в ячейке параметров.
- `templates/1_charge_calculation_props.ipynb` [ветки] - то же без словарей; собирается
  скриптом `templates/_build_props_template.py`.
- `make_charge_notebooks.py` - словарь `PARAMS` (по папке: `PTM_name`, `base_name`,
  `parent_residue`, `monomer_file`, `trimer_file`, `naming`, `resp_run`) и сборка:
  `python make_charge_notebooks.py [папки]`, `--props` [ветки], `--template`. Старый ноутбук с
  тем же именем переименовывается в `*_old.ipynb`, а не перезаписывается.
- Ноутбук запускается из папки модификации (относительные пути); если ноутбук делает
  `os.chdir`, пути строятся от `ROOT_DIR`.

## 8. Кластер (`cluster/resp_cluster.py`)
Команды из `notebooks_and_examples/` в `darwin_resp`: `check` (проверка узла), `prepare`
(задачи RESP для лизинов из файлов `4_*.pdb` в git), `review` (сводка до запуска),
`submit --chain` (задачи SLURM по очереди), `status`, `collect` (заряды → `RESP_chrges_*.json`
и сравнение с Espaloma). По умолчанию раздел `intbio`, 24 ядра, 30 ГБ, 8 ч.
Подробно - `cluster/README.md`.

## 9. Тесты (`tests/test_reference.py`)
Запуск в `darwin_resp`: `python -m pytest tests -q -m "not slow"` (~30 с) или без `-m` (~3 мин,
с AF546). Запускать до и после любой правки пакета.

| Тест | Что проверяет |
|---|---|
| `test_atom_names_match_committed`, `test_general_atoms_match_committed` | шаги 1-4 на эталонных папках дают те же имена, порядок атомов и атомы родителя, что файлы в git |
| `test_protonation_hw` | протоны HW шага 4.1 |
| `test_charge_constraints_from_rtp` | ограничения остова по именам |
| `test_capped_monomer` | ACE-X-NME, стереохимия |
| `test_resp_prepare`, `test_independent_constraints_drop_redundant` | подготовка RESP без квантовой химии |
| `test_rtp_roundtrip` [ветки] | чтение-запись rtp без изменений |
| `test_props_path_matches_committed` [ветки] | путь без словарей = старый путь (+ Benzoyl, lac) |
| `test_save_load_mol_roundtrip` [ветки] | SDF сохраняет все свойства |
| `test_run_acpype` [ветки] | acpype на трёх лизинах (пропускается без `topmol2`) |
| `test_hdb_entries_match_base_hdb` [ветки] | hdb по графу = базовый `aminoacids.hdb` |
| `test_write_forcefield_formyl` [ветки] | запись поля, повторный запуск без дублей |
| `test_validate_vacuum_formyl` [ветки] | GROMACS: pdb2gmx, grompp, минимизация (пропускается без `gmx`) |

## 10. Силовые поля в репозитории
- `amber14sb_parmbsc1_cufix.ff/` - базовое поле (amber14sb + parmbsc1 для ДНК + CUFIX).
  **Не править.** В нём же лежат посторонние файлы: rtp меток `CY3_*`, `CY5_*`, `Cy3_syn.rtp`
  (типы GAFF, которых нет в `atomtypes.atp`), `*-Copy1*`, `Untitled.ipynb` - из-за rtp меток
  `pdb2gmx` с этим полем не работает (вопрос 3 раздела 12).
- `amber14sb_mod.ff/` - поле с модификациями старого пути: `<Mod>.rtp` (RESP/вручную) и
  `<Mod>_ai.rtp` (Espaloma). Старые rtp содержат известные ошибки (`PLAN.md`).
- `build/amber14sb_mod.ff/` [ветки, вне git] - новое поле: 12 лизинов, заряды Espaloma,
  проверено в GROMACS.

---

## 11. Что изменилось с однофайловой версии

Точка отсчёта - коммит `a15a9b8` (2026-09-25): вся библиотека - один файл
`ResParTools.py`, 2 244 строки, 53 функции; тестов и документации нет.

| Область | Было (`a15a9b8`) | Стало |
|---|---|---|
| Устройство кода | один файл | пакет из 12 модулей; `ResParTools.py` - точка входа, старые вызовы работают |
| Объём | 2 244 строки, 53 функции | ~6 300 строк, 179 функций (в `main` ~5 400) |
| Чтение | `smi_to_chem`, `pdb_to_chem` | + `mol2_to_chem`, связи из CONECT, SMILES с номерами атомов (порядок сохраняется), `open_mol` / `save_mol` / `load_mol` [ветки] |
| Родительский остаток | выбирался по числу совпавших атомов | указывается явно, проверка остова и CB (`check_parent_residue`), сообщения с полными именами аминокислот |
| Перенумерация | ранняя `renumber_residue_atoms`; результат зависел от исходной нумерации атомов | родитель - в порядке шаблона, новые атомы - обходом графа от места пришивки; результат не зависит от исходной нумерации |
| Имена атомов | только буквенная схема (`greek_levels`) | `greek` и `index` (для больших меток, с автоматическим переходом); имена строятся только по графу - повторный вызов даёт те же имена |
| Координаты | `save_chem_to_pdb` при каждом сохранении строил 3D заново (Embed + UFF), стереохимия случайная | 2D по умолчанию; 3D - по запросу, с заданной стереохимией (CA - L, E/Z) |
| Заряды Espaloma | ограничения - индексы и заряды, скопированные между ноутбуками (у AF546 стояли от Lysine_1M); вход из PDB терял порядок связи C=O (неверные заряды ацилированных лизинов и AF546) | ограничения из rtp родителя по именам атомов, вход - остаток шага 4 с порядками связей; заряды пересчитаны |
| RESP | psiresp без оптимизации геометрии (без QCFractal он считает только градиент), без проверки результата | свой протокол amber: ACE-X-NME, 6 конформаций, оптимизация psi4, двухстадийный фит, проверка ограничений; локально и на SLURM; скрипт кластера |
| Шаги 2*-3* | ноутбуки; `chargeVal` руками; имена по `zip`; ошибка `(a and b) in`; LYS зашит; файлы поля дописываются при каждом запуске | [ветки] функции `run_acpype`, `assign_types`, `residue_rtp`, `hdb_entries`, `write_forcefield`; повторный запуск безопасен |
| Проверка результата | не было | тесты на эталонном наборе; [ветки] GROMACS: pdb2gmx + grompp + минимизация, МД |
| Хранение данных | словари `{имя: мол}` и JSON | [ветки] свойства атомов и молекулы, один SDF |
| Ноутбуки | копировались между папками и правились руками | шаблон + генератор; старые сохранены |
| Отладка | `print` | лог `start_log()` с вызовами, параметрами и молекулами |
| Документация | README | `docs/`: PIPELINE, PLAN, REVIEW, MOL_PROPS, GUIDE; `CLAUDE.md` |
| Окружения | одно старое | `darwin_ec`, `darwin_resp` (зафиксированы), функции сами вызывают нужное |

Путь по коммитам:
1. 2026-09-25…30 - работа в одном файле: чтение mol2, явный родитель, лог, перенумерация,
   имена атомов, рисование, протонирование, Espaloma с ограничениями из rtp (на `AF_546_cys`).
2. 2026-10-01 - `8bd1614`: файл разбит на пакет (поведение не менялось), `legacy`.
3. Октябрь - `resp.py` (шаг 6), кластер, шаблон и генератор ноутбуков, новые ноутбуки во всех
   папках, тесты и документы.
4. Ночь 2026-10-10 [ветки] - единый rtp, свойства молекулы, шаги 2*-3* в пакете, проверка в
   GROMACS, МД 12 лизинов.

Ночные ветки идут цепочкой, каждая от предыдущей:
`feature/rtp-io` → `feature/mol-props` → `feature/acpype-step2` → `feature/ff-step3` →
`feature/validate-md` → `integration/night-2026-10-10` (+ отчёт). В `main` - ничего из них.

---

## 12. Известные проблемы и открытые вопросы

Сейчас (подробности - `PLAN.md`, `NIGHT_REPORT.md`):
1. Тип CE ацилированных лизинов (`C8` или `CT`).
2. Где хранить результаты пути без словарей (`build/props/` вне git).
3. Посторонние rtp меток в базовом поле.
4. Порядок переноса веток в `main`.
5. RESP 1M и ACC: не сходятся конформации beta_2 (оба) и alphaR_2 (ACC).

Отложено: типы GAFF меток в поле (пункт 5 ревью), шаблоны концевых остатков, именование Trp,
RESP для меток AF (~130 атомов), лактил R у `Lysine_lac` (вероятно ошибка входа), пересчёт
старых rtp, переход старых ноутбуков на новые имена (`OT1`, `CT2`...).

Мелочи, замеченные при написании этого документа: docstring `ResParTools.py` перечисляет
модули без `resp` и `gmx`; в ячейке-описании шаблона без словарей упомянуто свойство
`RefResidue`, а в `MOL_PROPS.md` - `ParentTemplate`. Для шагов 2*, 3* и проверки пока нет
ячеек в ноутбуке без словарей (ночью они выполнялись скриптами).

## 13. Правила работы с проектом
- Любую правку функции проверять на эталонном наборе - по **содержанию** результата (имена,
  порядок атомов, имена остатков), а не только «не упало».
- Старые ноутбуки не редактировать; несовместимый старый вызов - предупреждение, не ошибка.
- Никаких молчаливых подмен: если функция что-то выбрала или поправила - она сообщает.
- Не зашивать конкретный остаток в код: библиотека для любых аминокислот (и нуклеиновых кислот).
- Большие этапы - в ветке `feature/...`, в `main` - после тестов; на GitHub - только с согласия.
- Квантовая химия - на кластере; локально - подготовка, проверка, тесты.
