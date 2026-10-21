# Карта пайплайна ResParTools

Обновлено: 2026-10-21. Планы - `docs/ROADMAP.md`.

Цель: параметры модифицированного остатка (посттрансляционная модификация, флуоресцентная
метка) для GROMACS amber14sb - имена атомов, топология, заряды, файлы силового поля, проверка в GROMACS.

Два пути с одним протоколом: **old** (словари `{имя: молекула}` и тримеры, ноутбуки `*_old.ipynb`)
и **на свойствах** (данные в свойствах молекулы RDKit, `docs/MOL_PROPS.md`). Шаги 1-6 у них одинаковые,
различаются хранение данных и шаги 2-3.

```mermaid
flowchart TD
    IN["molecules/&lt;мономер&gt;.smiles<br/>molecules/&lt;тример&gt;.smiles"]

    subgraph S1["Шаги 1-4.1: остаток"]
        A1["1. Чтение мономера и тримера"]
        A2["2. Мономер в тримере (MCS) → остаток"]
        A3["3. Перенумерация по шаблону родителя<br/>+ имена атомов greek / index"]
        A4["4. Правка имён, проверка дубликатов"]
        A41["4.1 Протоны HW на N и C (для acpype), 3D"]
        A1 --> A2 --> A3 --> A4 --> A41
    end

    subgraph Q["Шаги 5-6: заряды (остов фиксирован по rtp родителя)"]
        Q5["5. Espaloma (espaloma-charge_mod)"]
        Q6["6. RESP: ACE-X-NME, αR/β × 3 конформации,<br/>psi4 HF/6-31G* (φ/ψ заморожены),<br/>ESP psiresp, свой двухстадийный фит"]
        CL["кластер SLURM: resp_cluster.py<br/>prepare → review → submit → status → collect"]
        Q6 -. долгие расчёты .-> CL
    end

    IN --> A1
    A4 --> Q5
    A4 --> Q6
    IN -- "стереохимия мономера" --> Q6

    Q5 --> CH["data/AI_chrges_&lt;имя&gt;.json"]
    Q6 --> CHR["data/RESP_chrges_&lt;имя&gt;.json"]
    CL --> CHR

    subgraph OLD["Путь old (по папкам модификаций)"]
        O2["2_generate_topology_old:<br/>obabel + acpype, parmed → Acpype_data/*_mod.itp"]
        O3["3_edd_topology_old:<br/>типы родителя, заряды → force_field_files/<br/>rtp, hdb, residuetypes.dat, atp"]
        O2 --> O3
    end

    subgraph NEW["Путь на свойствах (пакет)"]
        N2["run_acpype(mol): заряд и HW из молекулы"]
        N3["write_forcefield: assign_types, residue_rtp,<br/>hdb_entries → build/amber14sb_mod.ff"]
        NV["проверка gmx: ACE-X-NME, pdb2gmx -ignh, grompp,<br/>EM в вакууме, МД в TIP3P → build/md/"]
        N2 --> N3 --> NV
    end

    A41 -- "&lt;PTM&gt;_rn_H_3D.pdb" --> O2
    CH --> O3
    CHR --> O3
    A4 -- "general_atoms.json" --> O3
    O3 --> FF["amber14sb_mod.ff (сборка вручную)"]

    A41 -- "SDF со свойствами" --> N2
    CH -. "Charge&lt;Метод&gt; в атомах" .-> N3
    CHR -.-> N3
```

## Путь old: шаги ноутбука `1_charge_calculation_old.ipynb`
Ноутбуки `1_charge_calculation_old`, `2_generate_topology_old`, `3_edd_topology_old` собираются
в каждой папке из `notebooks_and_examples/templates/<то же имя>` скриптом `make_charge_notebooks.py`;
всё, что отличает модификацию, - в `PARAMS`: `PTM_name`, `base_name`, `parent_residue`,
`monomer_file`, `trimer_file`, `naming`, `resp_run`. Окружение ядра шага 1 - `darwin_ec`.
Совсем старые ноутбуки (`_K_*`, `_Kcr`, `_resp`, ...) - в `<папка>/archive/`.

| Шаг | Что делает | Функции (модуль пакета `respartools/`) | Выход (`molecules/substructure/`, если не указано) |
|---|---|---|---|
| 1 | чтение мономера и тримера | `file_opener` (fileio) | — |
| 2 | мономер в тримере (MCS), подструктура | `match_mon_to_pol` (matching), `draw_mol_grid` (draw) | — |
| 2.2 | остаток как есть, 2D | `save_chem_to_pdb`, `save_chem_to_smiles(atom_map=True)` (fileio) | `3_<имя>.pdb`, `3_<имя>.smiles` (+ `_no_H`) |
| 3.1 | перенумерация: родительский остаток в порядке шаблона, новые атомы - обходом графа | `renumber_residue_atoms` (residue) | `3_1_<имя>.pdb` |
| 3.2 | имена атомов (`greek` - amber-подобные, `index` - для меток), параметры остатка | `rdkit_pdb_modification`, `save_chem_to_pdb(..., resname=..., no_H=True)` (residue, fileio) | `3_2_<имя>.pdb` |
| 4 | проверка, правка имён (`rename_map`), дубликаты | `modifie_residue_info`, `check_duplicate_atom_names` (residue) | `4_<имя>.pdb`, `data/general_atoms.json` |
| 4.1 | протоны HW1/HW2 на N и C (для acpype) | `add_protons_and_renumber_H` (residue), `set_mol_coords` 3D | `4_1_<имя>.pdb/.smiles`, `<PTM>_rn_H_3D.pdb` |
| 5 | заряды Espaloma, остов N H CA HA C O фиксирован (rtp родителя) | `charge_constraints_from_rtp` (charges), `charge` (espaloma-charge_mod), `save_charges_json(atom_names=)` (fileio) | `data/AI_chrges_<имя>.json` |
| 6 | заряды RESP (альтернатива 5) | `prepare_resp_job`, `run_resp`, `load_resp_result`, `compare_charges` (resp, charges) | `RESP_data/<имя>_capped/`, `data/RESP_chrges_<имя>.json` |
| 2_generate_topology_old | 3D-вход шага 4.1, заряд acpype - сумма формальных зарядов молекулы; obabel PDB → MDL, acpype (`amber`, `gas`), parmed переносит имена (порядок проверяется по элементам) | `pdb_to_chem`, `draw_molecule` | `Acpype_data/<имя>.acpype/<имя>_mod.itp` |
| 3_edd_topology_old | без HW; типы родителя у общих атомов (`general_atoms.json`), новые типы в atp; заряды (порядок сверяется по именам); rtp; residuetypes.dat и hdb | `remove_extra_H`, `read_rtp`, `check_atomtypes`, `load_charges_json`, `hdb_generator` (forcefield, fileio) | `force_field_files/<папка>.rtp`, `aminoacids.hdb`, `residuetypes.dat`, `atomtypes.atp` |

`.rtp` из `force_field_files/` собираются в `notebooks_and_examples/amber14sb_mod.ff/`
(`<Mod>.rtp` - RESP/ручные, `<Mod>_ai.rtp` - Espaloma). rtp меток (`CY3_*`, `CY5_*`, `Cy3_syn`) и
типы GAFF из базового поля вынесены в `notebooks_and_examples/dyes_wip/` до решения по типам.

## Путь на свойствах
Данные - в свойствах молекулы (`docs/MOL_PROPS.md`), ноутбук -
`templates/1_charge_calculation_props.ipynb` (`make_charge_notebooks.py --props`), результаты -
в `notebooks_and_examples/build/props/<папка>/` (`build/` вне git). Шаги 2-3 и проверка пока
вызываются из кода (`build/md_code/night_validate.py`); ноутбуки `1_residue_charges`, `2_forcefield`
и `build_residue` без тримеров - в планах (`docs/ROADMAP.md`).

| Шаг | Функции | Что пишется в молекулу / файлы |
|---|---|---|
| 1 | `open_mol` (fileio) | у атомов `SrcIdx`, у молекулы `ModName`, `SourceFile` |
| 2 | `find_residue` (matching) | `PolymerIdx`; открытая валентность N и C - неявные H |
| 3 | `renumber_residue_atoms`, `rdkit_pdb_modification` (residue) | `PrevIdx`, `ParentName`, `AtomName`, `Parent`, `Naming` |
| 4 | `modifie_residue_info`, `check_duplicate_atom_names` | — |
| 4.1 | `add_protons_and_renumber_H` (residue) | `AddedH` (на шаге 3 поля удаляются по нему, не по имени) |
| 5/6 | `charge_constraints_from_rtp`, `charge`, `set_charges` (charges); RESP - `prepare_resp_job(working_dir=OUT/resp)`, `run_resp` | `Charge<Метод>`, `ChargeFixed`, `ChargeMethod` |
| сохранение | `save_mol` / `load_mol` (fileio) | `4_<имя>.sdf`, `4_1_<имя>.sdf` (все свойства), `<PTM>_rn_H_3D.pdb` |
| 2 (acpype) | `run_acpype(mol, workdir, net_charge=None)` (forcefield, acpype из `topmol2`) | `AcpypeType`, `AcpypeItp`; элементы сверяются |
| 3 (поле) | `write_forcefield` (forcefield): `assign_types`, `residue_rtp`, `hdb_entries` | `AtomType`, `TypeSource`; `<ResName>.rtp`, `<ResName>.hdb`, `residuetypes.dat`, `atomtypes.atp` → `build/amber14sb_mod.ff` |
| проверка | `write_validate_job`, `run_validate(run='local'\|'slurm')`, `validate_report` (gmx) | ACE-X-NME: pdb2gmx -ignh, grompp, EM в вакууме (`em_vac`), при `md_ns > 0` - вода TIP3P, ионы, EM, NVT, NPT, МД; отчёт: предупреждения, Fmax, длины связей против `em_vac.gro` |

Правила шага 3: тип родителя сохраняется, если окружение атома то же (элемент, число соседей,
число H, гибридизация/ароматичность: родитель - по шаблону, остаток - в ACE-X-NME), у водорода - если
окружение не изменилось и у соседей его атома; типы acpype у новых атомов не меняются (CE
ацилированных лизинов - `C8`). Явные параметры (из itp acpype, единицы GROMACS) - только для связей,
углов, диэдралов, которых нет в `ffbonded.itp` по типам. Один парсер rtp - `read_rtp` / `write_rtp` (forcefield).

## Шаг 6 (RESP) подробно
Протокол как для зарядов amber ff94/ff14SB/ff19SB (Cornell 1995, Cieplak 1995):
1. **Молекула** - `capped_monomer`: ACE-X-NME из SMILES мономера, стереохимия мономера
   (CA - L, центры боковой цепи, E/Z), перенумерована в порядок остатка шага 4.
   Файлы `molecules/<имя>_capped.smiles` - для всех модификаций.
2. **Конформации** - `resp_conformers`: пары (φ, ψ) из `fix_backbone` (по умолчанию αR −60/−40
   и β −120/130) × `n_sidechain=3`; ETKDG → углы → MMFF с замороженными φ/ψ → без контакта
   боковой цепи с остовом → окно 10 ккал/моль от самой низкой → непохожие. Стереохимия каждой
   конформации проверяется.
3. **Оптимизация** HF/6-31G* в psi4, φ/ψ заморожены, одним расчётом до `max_opt_steps=400` шагов.
   Оптимизатор - `optimizer='geometric'` по умолчанию (свой цикл geomeTRIC 1.1: отвергает шаги с ростом
   энергии; `psi4.optimize(engine='geometric')` в psi4 1.6.1 с ним несовместим); `'optking'` - встроенный
   psi4, его же берут старые задачи без ключа в `spec.json`. Не сошлось - геометрия в `current.xyz`,
   повторный запуск продолжает с неё ещё на `continue_steps=100` шагов (история энергий - в `status.json`
   и выводе `status`). Связи против `start.xyz` проверяются после каждой оптимизации (`_broken_bonds`):
   разрыв или падение psi4 - геометрия в `failed.xyz`, с неё конформация не продолжается. После
   оптимизации - проверка контакта. Повторный `prepare` с другими конформациями пишет задачу в
   `<имя>_01`, `<имя>_02`, ...
4. **ESP** (psiresp, сетка MSK) и **двухстадийный RESP** (свой, `_fit_two_stage`): стадия 1 -
   ограничение 0.0005, стадия 2 - CH2/CH3, 0.001. Фиксированы: остов из rtp родителя,
   кэпы ACE/NME из rtp; эквивалентны симметричные и резонансные атомы (по графу без порядков связей).
5. **Запуск**: `run='local'` (в окружении `darwin_resp` отдельным процессом) или
   `run='slurm'`; на кластере - `notebooks_and_examples/cluster/resp_cluster.py`:
   `check` → `prepare` (текущая задача - в `<имя>_capped.current`) → `review` → `submit`
   (параллельно; `--chain` - по очереди) → `status` → `collect` (`data/RESP_chrges_<имя>.json`
   с именами атомов). См. `cluster/README.md`.

## Окружения
| Окружение | Файл | Для чего |
|---|---|---|
| `darwin_ec` | `ResParTools_ec.yml` | ноутбуки, шаги 1-5 (Espaloma), Python 3.12 |
| `darwin_resp` | `psiresp_min.yml` (снимок `darwin_resp.lock.yml`) | шаг 6: psi4 1.6.1, psiresp 0.4.2, Python 3.9; тесты |
| `topmol2` | — (локальное) | шаги 2-3: acpype, parmed, Python 3.8; `run_acpype` вызывает acpype отсюда |
| `gmx` | `conda create -n gmx -c conda-forge gromacs` | проверка поля: GROMACS 2026.3 (`run_validate`) |

Общее окружение `darwin_ec` + `darwin_resp` не собирается: psi4 1.6.1 только для Python ≤ 3.10
со старыми библиотеками, новые psi4 требуют pydantic 2, а psiresp 0.4.2 - pydantic 1.

Ноутбуки остаются на ядре `darwin_ec`: `pt.run_resp(..., run='local')` запускает RESP в `darwin_resp`
отдельным процессом (окружение ищется рядом с текущим), `run='slurm'` пишет `run_resp.sbatch`.
Сохранённые в старых ноутбуках ядра - `.conda-espaloma` (шаг 1) и `.conda-topmol2` (шаги 2-3).
`ai_topmol.yml` - старое окружение `espaloma` (Python 3.9, dgl 1.1.2).

Установка `darwin_ec` (yml выгружен на кластере: там `prefix:` и сборки CUDA 12). Имя - явно;
без драйвера NVIDIA - подменить виртуальный пакет CUDA (не ниже 12.4: этого требует закреплённый
ffmpeg), torch и dgl тогда работают на CPU:
```bash
CONDA_OVERRIDE_CUDA=12.4 conda env create -n darwin_ec -f ResParTools_ec.yml
conda remove -n darwin_ec --force -y espaloma_charge
```
Вторая команда обязательна: yml ставит upstream `espaloma_charge=0.0.8`, а ноутбуки добавляют
`espaloma-charge_mod` через `sys.path.append` (в конец пути), и копия из site-packages его заслоняет.
Признак: `charge() got an unexpected keyword argument 'constraints'`.

## `espaloma-charge_mod/`
Изменённая копия Espaloma Charge (нетронутая - `espaloma-charge/`, отличия в `app.py`, `models.py`,
`utils.py`). Главное: `charge(mol, constraints={индекс_атома: заряд})` - последний слой
`ChargeEquilibrium` заменяется на `ChargeEquilibrium_mod`, который фиксирует заданные атомы и
распределяет оставшийся заряд только по свободным. Так модифицированный остаток сохраняет заряды
остова amber14sb. Веса модели скачиваются при первом вызове (URL в `app.py`). Тесты upstream:
`cd espaloma-charge && pytest espaloma_charge/tests`; у `_mod` своих тестов нет.

## Особенности ноутбуков
- Относительные пути считаются от папки модификации (там Jupyter запускает ядро). Ноутбуки с
  `os.chdir` задают в первой ячейке `ROOT_DIR` (абсолютный путь `notebooks_and_examples/`, строка
  защищена от повторного запуска) и строят от него все пути. Абсолютные пути кластера не добавлять
  (старый `/home/_projects/2022_md_FRET_nv/param_R_CIT` - та же папка `notebooks_and_examples/`).
- Сохранённые выводы старых ноутбуков содержат старые пути и трассировки `param_tool` (так модуль
  назывался до 7f1d61a) - это история, их не переписывали.

## Лог для отладки
Включается в ячейке ноутбука, по умолчанию выключен:
```python
pt.start_log()        # новая папка logs/<дата_время>/ в текущей папке модификации
pt.log_note('заряды Espaloma', charges=charges)   # своя запись из ноутбука
pt.stop_log()
```
В папке запуска: `log.txt` (вызовы с параметрами и вложенностью, сообщения функций, словари
сопоставления, записанные файлы, время), `calls.jsonl` (то же в машинном виде), `files/` (молекулы
вызовов верхнего уровня в PDB, SDF, mol2, SVG с подписями «индекс:имя»). В заголовке: ноутбук,
окружение, версии пакетов, ветка, коммит, незакоммиченные файлы, sha256 модуля. Логируются функции
с `@logged`; у функций с `@data_to_dict` он ставится под ним. `charge()` Espaloma вызывается из
ноутбука, поэтому в лог попадает только через `log_note`.

## Файлы в папке модификации
| Папка | Что | В git |
|---|---|---|
| `molecules/` | SMILES мономера, тримера, `<имя>_capped.smiles`; `substructure/` - файлы шагов 2-4.1 | да |
| `data/` | `AI_chrges_*.json`, `RESP_chrges_*.json`, `general_atoms.json` | да |
| `Acpype_data/` | acpype и `*_mod.itp` (путь old) | да |
| `force_field_files/` | rtp, hdb, residuetypes.dat, atp (путь old) | да |
| `RESP_data/` | задачи и расчёты RESP | нет |
| `archive/` | совсем старые ноутбуки, заряды, запуски psiresp | да |
| `logs/` | лог `start_log` | нет |

## Тесты
`tests/test_reference.py` - эталонный набор (Lysine_prop, Lysine_Malonyl, Lysine_Formyl,
Lysine_3M, AF_546_cys): шаги 1-4.1 сравниваются с файлами в git, ограничения зарядов,
кэпированный мономер, подготовка RESP (без квантовой химии).
```bash
conda activate darwin_resp
python -m pytest tests -q -m "not slow"   # ~30 с, лизины
python -m pytest tests -q                 # ~3 мин, вместе с AF546
```

## Пакет `respartools/`
`log` (лог `start_log`), `utils`, `fileio`, `draw`, `matching`, `residue`, `forcefield`
(rtp/itp, acpype, типы, поле), `charges`, `resp`, `gmx` (проверка в GROMACS), `legacy` (старые
функции и старые вызовы с предупреждением). Ноутбуки подключают его как раньше: `import ResParTools as pt`.

Правило импортов внутри пакета: `log` и `utils` импортируются в начале модуля (декораторы,
пути), остальные модули пакета - в **конце** модуля, чтобы взаимные ссылки не ломали импорт.
Код совместим с Python 3.8: пакет импортируется в `topmol2` (3.8), `espaloma`, `darwin_resp` (3.9) и
`darwin_ec` (3.12), поэтому без `match` и без аннотаций `X | Y` и `list[int]`.
`ResParTools.py` реэкспортирует все имена всех модулей (включая `_private`), так что старые ноутбуки
работают без изменений. Функции с `@data_to_dict` принимают путь, список или словарь и возвращают
словарь по имени файла.
