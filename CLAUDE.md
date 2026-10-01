# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research toolkit (docs and notebook text are in Russian) for parameterizing **modified amino acids / post-translational modifications** for the GROMACS **amber14sb** force field. Partial charges come from the **Espaloma Charge** neural network (a patched copy that can pin selected atoms to fixed charges). Topologies come from **acpype**. The work is almost entirely Jupyter notebooks. There is no build system, and the project itself has no test suite.

## Docs and tests
- `docs/PIPELINE.md` - pipeline map (steps, functions, files, environments); `docs/PLAN.md` - status, decisions, next steps, known data issues. Keep both updated when the pipeline or plan changes.
- `tests/test_reference.py` - reference-set tests (Lysine_prop, Lysine_Malonyl, Lysine_Formyl, Lysine_3M, AF_546_cys): steps 1-4.1 compared with committed files, rtp constraints, capped monomer, RESP preparation (no QM). Run in `darwin_resp`: `python -m pytest tests -q -m "not slow"` (~30 s) or all (~3 min). Run them after any library change; if behaviour changes on purpose, regenerate the files with the notebooks and commit them.

## Environments

- **`ResParTools_ec.yml`**: creates a conda env named `darwin_ec`. This is the primary env for current work. It has Python 3.12, rdkit 2024.03, dgl 2.3 and pytorch 2.3.1 (CUDA 12 builds), plus MDAnalysis, openbabel, nglview and JupyterLab. It covers step 1 (the charge calculation with `espaloma-charge_mod`). It does **not** include `acpype`, `parmed`, `ambertools` or `psiresp`, so steps 2 and 3 and the RESP notebooks still need another env. The file was exported on the cluster, which is why it has a cluster `prefix:` and pinned builds. Install it with an explicit name. On a machine without an NVIDIA driver, also override the CUDA virtual package. Use at least 12.4, because the pinned ffmpeg build requires `__cuda>=12.4`. torch and dgl then run on CPU:
  ```
  CONDA_OVERRIDE_CUDA=12.4 conda env create -n darwin_ec -f ResParTools_ec.yml
  conda run -n darwin_ec python ...   # or: conda activate darwin_ec
  ```
  The yml also pulls upstream `espaloma_charge=0.0.8` from conda-forge. The notebooks add `espaloma-charge_mod` with `sys.path.append`, which puts it at the *end* of the path, so the site-packages copy shadows it. The symptom is `charge() got an unexpected keyword argument 'constraints'`. After you create the env, remove the upstream copy: `conda remove -n darwin_ec --force -y espaloma_charge`.
- `ai_topmol.yml`: creates a conda env named `espaloma` (Python 3.9, dgl 1.1.2, torch 2.3.1, rdkit). It is used for charge calculation with `espaloma-charge_mod`.
- `psiresp_min.yml`: creates a conda env named `darwin_resp` (Python 3.9, psi4 1.6.1, psiresp 0.4.2, `pydantic` 1.10; versions pinned). Full snapshot: `darwin_resp.lock.yml`. It runs RESP (step 6, `respartools/resp.py`, `notebooks_and_examples/cluster/`). Notebooks stay on the `darwin_ec` kernel: `pt.run_resp(..., run='local')` runs the RESP job in `darwin_resp` as a subprocess (found next to the current env), `run='slurm'` writes `run_resp.sbatch`. psi4 1.6.1 exists only for Python <= 3.10.
- The notebooks' saved kernelspecs name `.conda-espaloma` for step 1 and `.conda-topmol2` for steps 2 and 3. Locally, the `topmol2` env (Python 3.8, not defined by any yml in the repo) is what has acpype and parmed.

The vendored Espaloma has upstream tests, which you run from inside that package: `cd espaloma-charge && pytest espaloma_charge/tests` (single test: `pytest espaloma_charge/tests/test_app.py::<name>`). `espaloma-charge_mod` has no tests of its own.

## Architecture

### `notebooks_and_examples/respartools/`: the shared helper library (package)
Since 2026-10 the library is a package; `notebooks_and_examples/ResParTools.py` is only an entry point that re-exports every name of every module (including `_private` ones), so notebooks keep using `import ResParTools as pt` and old notebooks work unchanged. Modules:
- **`log`**: debug log (`start_log`, `stop_log`, `log_note`, `@logged`).
- **`utils`**: `get_name`, `@data_to_dict`, `path_parser`, `print_red`/`print_green`, `_PACKAGE_ROOT` (= `notebooks_and_examples/`, base for data paths).
- **`fileio`**: `file_opener`, `smi_to_chem`, `pdb_to_chem`, `mol2_to_chem`, `save_chem_to_smiles`/`_pdb`/`_mol`, `save_aa_chem_to_pdb`, `save_charges_json`. Functions with `@data_to_dict` accept a path, a list or a dict and **return a dict keyed by file basename**.
- **`draw`**: `draw_molecule` (old name `draw_mol_with_atom_index`), `draw_mol_grid`, `draw_mon_pol_match`.
- **`matching`**: `match_chem*`, `match_mon_to_pol`, `REF_TEMPLATES`, `AMINO_ACIDS`, `find_ref_residue` (old name `find_ref_aa`).
- **`residue`**: `renumber_residue_atoms`, `rdkit_pdb_modification` (atom names), `find_backbone_match`, `set_mol_coords` (2D/3D, stereo), `modifie_residue_info`, `check_duplicate_atom_names`, `add_protons_and_renumber_H`.
- **`forcefield`**: `hdb_generator`, `check_atomtypes` (`.atp`), `make_r2b`, `remove_extra_H`.
- **`charges`**: `read_rtp_charges`, `charge_constraints_from_rtp` (Espaloma/RESP constraints from amber14sb by atom name).
- **`legacy`**: compatibility for functions old notebooks call.

Imports between package modules: `log`/`utils` are imported at the top of a module (decorators, paths); all other intra-package imports are at the **end** of the module, so mutual references do not break import. Keep that rule when adding functions.

The module was named `param_tool.py` before commit 7f1d61a, and stale `__pycache__/param_tool.*.pyc` files are still present. It is imported under Python 3.8 through 3.12 (`topmol2`, `espaloma`, `darwin_ec`, `darwin_resp`), so keep it compatible with 3.8. That rules out `match` statements and PEP 604 `X | Y` annotations.

### `espaloma-charge_mod/`: patched Espaloma Charge
This differs from the untouched upstream copy in `espaloma-charge/` in `app.py`, `models.py` and `utils.py`. The key addition is `charge(mol, constraints={atom_idx: fixed_q, ...})`. When constraints are given, the model's final `ChargeEquilibrium` layer is replaced with `ChargeEquilibrium_mod` (in `models.py`). That layer fixes the constrained atoms and spreads the remaining total charge over the free atoms only. The notebooks import this package with `sys.path.append("../../espaloma-charge_mod/")`, not with pip. On first use the model weights are downloaded from the GitHub release URL in `app.py`.

The constraints are how modified residues keep the amber14sb backbone charges. The notebooks pin the backbone atoms (N, H, CA, C, O, …) to the standard amber values and let Espaloma assign charges to the side chain.

### Per-modification pipeline (`notebooks_and_examples/<Lysine_XXX | AF_*>/`)
Each modification folder follows the same steps. Use `Lysine_3M/` as a reference example.
New notebooks (2026-10) are built from `notebooks_and_examples/templates/1_charge_calculation.ipynb` by `make_charge_notebooks.py` (`PARAMS` holds per-folder parameters: PTM_name, base_name, parent_residue, monomer/trimer files, naming, resp_run). Old notebooks are kept untouched (`*_old.ipynb` where names collided). Steps of the new notebook: 1 open → 2 match monomer in trimer → 2.2 save as is → 3 renumber + atom names (check cell with `draw_mol_grid`, then save cell) → 4 check/edit residue, `general_atoms.json` → 4.1 add HW protons, 3D input for acpype → 5 Espaloma (constraints from rtp by atom name) → 6 RESP (same inputs; ACE/NME caps, αR/β × side-chain conformers, psi4 optimization with frozen φ/ψ, psiresp ESP, own two-stage fit). Cluster runs: `notebooks_and_examples/cluster/README.md`.

1. `1_charge_calculation*.ipynb`: reads the monomer and trimer SMILES from `molecules/`, matches the monomer inside the trimers (N/M/C positions), writes substructures to `molecules/substructure/`, renames atoms to canonical names using the reference amino acids (writes `general_atoms.json`), protonates atoms with non-standard valence (acpype rejects them), then runs constrained `charge(...)` and writes `AI_chrges_<name>.json`.
2. `2_generate_topology*.ipynb`: runs acpype `ACTopol` on the single residue in `Acpype_data/` (obabel converts PDB to MDL first), then uses parmed to copy the atom names and residue name back into a `*_mod.itp`.
3. `../3_edd_topology.ipynb` (shared, top level; `Lysine_Cro` has its own copy): parses `aminoacids.rtp`, strips atoms that are not part of the residue, assigns atom types from the parent amino acid, inserts the Espaloma charges, and writes `.rtp`, `.r2b`, `residuetypes.dat`, `.hdb`, `Makefile.am`/`.in` and `.atp` entries into `<mod>/force_field_files/`.

The generated `.rtp` files are then collected into the modified force field `notebooks_and_examples/amber14sb_mod.ff/`. It holds `<Mod>.rtp` (hand/RESP) and `<Mod>_ai.rtp` (Espaloma) variants side by side. `amber14sb_parmbsc1_cufix.ff/` is the unmodified base force field. `aminoacids_dict.json` is the parsed base `aminoacids.rtp`.

### Notebook gotchas
- Relative paths assume that the cwd is the modification folder, which is where Jupyter starts the kernel. Notebooks that `os.chdir` define `ROOT_DIR` (the absolute path of `notebooks_and_examples/`) in their first cell and build every chdir target from it, for example `os.chdir(f'{ROOT_DIR}/{PTM_folder}/Acpype_data')`. The `ROOT_DIR` line is guarded, so re-running the first cell after a chdir does not change it. Never add absolute cluster paths. The old code used `/home/_projects/2022_md_FRET_nv/param_R_CIT`, which is the same directory as `notebooks_and_examples/`.
- Saved cell outputs still contain the old paths and `param_tool` tracebacks. Those outputs are historical and were not rewritten.
- `data/`, `datasets/` and `RESP_data/` are deliberately untracked (they hold large calculation outputs).

### Лог для отладки
Включается в ячейке ноутбука, по умолчанию выключен:
```python
pt.start_log()        # новая папка logs/<дата_время>/ в текущей папке модификации
...                   # ячейки пайплайна
pt.log_note('заряды Espaloma', charges=charges)   # своя запись из ноутбука
pt.stop_log()
```
В папке запуска: `log.txt` (читаемый журнал: вызовы с параметрами, вложенные вызовы с отступом, сообщения функций, словари сопоставления, записанные файлы, время), `calls.jsonl` (то же в машинном виде) и `files/` (молекулы из вызовов верхнего уровня в PDB, SDF, mol2, SVG с подписями «индекс:имя»). В заголовке: ноутбук, окружение, версии пакетов, ветка, коммит и незакоммиченные файлы, sha256 модуля. Логируются функции с декоратором `@logged`; для функций с `@data_to_dict` он ставится под ним. `logs/` в `.gitignore`.

## Правила работы (согласованы с автором проекта)

### Проверка изменений
- Любое изменение функции проверять на эталонном наборе: `Lysine_prop`, `Lysine_Malonyl`, `Lysine_Formyl`, `Lysine_3M`, `AF_546_cys` (лизины разной сложности и флуоресцентная метка на цистеине).
- Проверять не только то, что код не падает, а **содержание результата**: имена и нумерацию атомов, имена остатков, сопоставления подструктур, соответствие номенклатуре GROMACS/amber14sb. Главный риск — код, который «работает», но выдаёт файлы с неверными именами.
- Перед изменением поведения фиксировать результат «до» и сравнивать с «после» на данных проекта.

### Обратная совместимость
- Старые ноутбуки — запись того, как работал функционал раньше. Их **не редактировать**.
- Если новая версия функции несовместима со старым вызовом, функция должна **распознать старый вызов и выдать предупреждение**, а не падать. Не менять поведение по умолчанию у массово используемых функций (например, `pdb_to_chem(format_coord='2D')`).

### Общность
- Сейчас примеры — модификации лизина и метки на цистеине, но библиотека строится для **любых аминокислот и нуклеиновых кислот** и для работы с датасетами. Не зашивать в код предположения о конкретном остатке или типе полимера; типы остатков описываются данными (`REF_TEMPLATES`, `AMINO_ACIDS`).

### Поведение функций
- Никаких молчаливых запасных вариантов: если функция подменяет результат (берёт первое совпадение, восстанавливает заряд и т. п.), она сообщает об этом.
- Родительский остаток модификации указывается явно (`ref_base_name`); автовыбор по числу совпавших атомов ненадёжен. Старые вызовы без него работают по-старому с предупреждением.
- В сообщениях аминокислоты называть полностью: «цистеин (Cys, C), шаблон GCG_H».
- В консоли остаются только базовые сообщения, анимации долгих операций и предупреждения (как было в библиотеке). Подробности — в файл лога, который включается командой в ячейке ноутбука и по умолчанию выключен.

### Порядок работы
- Ошибки разбираются по одной, с обсуждением. Найденное по пути записывается в список, а не исправляется сразу.
- Коммиты — в рабочую ветку; на GitHub (`intbio`) ничего не отправлять без явного согласия.
