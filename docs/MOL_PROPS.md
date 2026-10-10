# Данные в свойствах молекулы (проект, 2026-10-10)

Цель: отказаться от словарей `{имя: молекула}`, словарей сопоставления (`match_dict`,
`mon_pol_matches`, `OldIndices`, `ParentAtoms`, `AddedH`, `CapAtoms`) и промежуточных файлов
(`general_atoms.json`, `AI_chrges_*.json`, `partial_charges.json`). Всё, что известно об атоме,
хранится в самом атоме; всё, что известно о молекуле, - в молекуле. Работа идёт в ветке
`feature/mol-props`.

## Проверено на RDKit (darwin_ec)
| Операция | свойства атомов | свойства молекулы |
|---|---|---|
| `Chem.Mol(m)` (копия) | сохраняются | сохраняются |
| `Chem.RenumberAtoms` | сохраняются | **теряются** |
| `Chem.RemoveHs` | сохраняются (у оставшихся) | сохраняются |
| `RWMol.RemoveAtom` | сохраняются (у оставшихся) | сохраняются |
| запись/чтение SDF (`CreateAtom...PropertyList`) | сохраняются, **но возвращаются строками** | сохраняются (строки) |
| SDF | - | информация остатка PDB (`GetPDBResidueInfo`) не сохраняется |

Следствия:
1. Индексы атомов в свойствах молекулы (списки `ParentAtoms`, `AddedH`, `OldIndices`)
   ломаются при перенумерации - поэтому **роль атома пишется в сам атом** (флаг/имя), а не
   списком индексов в молекулу.
2. Свои функции, меняющие порядок атомов, переносят свойства молекулы (одна строка в
   `renumber_residue_atoms`).
3. Нужна пара `save_mol(mol, путь)` / `load_mol(путь)` (SDF): при чтении свойства
   приводятся к своим типам по таблице ниже, информация остатка PDB восстанавливается из
   `AtomName`, `ResName`, `ResId`, `Chain`. Это заменяет промежуточные PDB/JSON между шагами;
   PDB остаётся как выходной формат (для acpype и просмотра).

## Свойства молекулы
| Свойство | Тип | Пример | Кто пишет | Что заменяет |
|---|---|---|---|---|
| `_Name` | str | `Lysine_Formyl` | шаг 1 (из параметров) | ключ словаря `{имя: мол}` |
| `ModName` | str | `K_form` | шаг 1 (имя файла мономера) | `mon_name` |
| `Role` | str | `monomer` / `trimer` / `residue` / `capped` | шаги 1, 2, RESP | разные переменные ноутбука |
| `Stage` | str | `3.1`, `4`, `4.1`, `5` | каждый шаг | номер в имени файла (`3_1_`, `4_`) |
| `Parent` | str | `K` | шаг 1 (параметр) | `parent_residue`, `RefResidue` |
| `ParentResName` | str | `LYS` | шаг 1 (`AMINO_ACIDS`) | `rtp_residue` в ячейке |
| `ParentTemplate` | str | `GKG_H` | шаг 3.1 | печать в сообщении |
| `ResName` | str | `KFO` | шаг 3.2 | `base_name`, `residue_params` |
| `ResId`, `Chain` | int, str | `2`, `A` | шаг 3.2 | `residue_params` |
| `Naming` | str | `greek` / `index` | шаг 3.2 | `residue_params` |
| `SourceFile` | str | `molecules/K_form.smiles` | шаг 1 | - |
| `ChargeMethod` | str | `espaloma` / `resp` | шаг 5/6 | выбор файла зарядов на шаге 3 |
| `AcpypeItp` | str | путь к `*_GMX.itp` | шаг 2 | имя `acpype_name` в шаге 3 |

Заряд остатка отдельно не хранится: это `Chem.GetFormalCharge(mol)` (так же берётся
`chargeVal` для acpype - ошибка 2 из `REVIEW.md`).

## Свойства атома
| Свойство | Тип | Пример | Кто пишет | Что заменяет |
|---|---|---|---|---|
| `SrcIdx` | int | `7` | шаг 1: индекс в файле мономера | - (сквозная связь с исходником) |
| `TrimerIdx` | int | `15` | шаг 2: индекс в тримере (у атомов подструктуры) | `match_dict['mon_pol_matches']` |
| `PrevIdx` | int | `4` | шаг 3.1: индекс до перенумерации | `OldIndices` |
| `ParentName` | str | `CB` / пусто | шаг 3.1: имя атома в шаблоне родителя; пусто у новых атомов | `ParentAtoms`, `general_atoms.json` |
| `AtomName` | str | `CB`, `HT21` | шаг 3.2, правка - шаг 4 | (уже есть) |
| `Added` | bool | `True` у HW | шаг 4.1 | `AddedH`, поиск `'W'` в имени (`remove_extra_H`) |
| `ChargeFixed` | bool | `True` у N H CA HA C O | шаг 5/6 (ограничения из rtp) | словарь `constraints` |
| `ChargeEspaloma` | float | `-0.1165` | шаг 5 | `AI_chrges_<имя>.json` |
| `ChargeRESP` | float | `-0.0981` | шаг 6 | `RESP_chrges_<имя>.json` |
| (`Charge<Метод>`) | float | - | будущие методы зарядов | `partial_charges.json` |
| `AcpypeType` | str | `CT` | шаг 2 (из itp, сопоставление по порядку + проверка элементов) | чтение itp в шаге 3 |
| `AtomType` | str | `C8` | шаг 3: тип родителя, если окружение то же, иначе `AcpypeType` | `exclusion_atoms`, `aminoacid='LYS'` |
| `TypeSource` | str | `parent` / `acpype` | шаг 3 | цветная печать таблицы |
| `CapResidue` | str | `ACE` / `NME` | RESP: кэпы | `CapAtoms` |

## Что остаётся не в свойствах
- **Параметры связей, углов, диэдралов** из acpype - это таблицы по парам/четвёркам атомов,
  а не свойства одного атома. Источник - itp acpype (`AcpypeItp`), читается parmed на шаге 3.
- Задача RESP (`RESP_data/<имя>/spec.json`) - вход для кластера, остаётся файлом.
- Файлы силового поля - выход шага 4.

## Вход и выход шагов (было → будет)
| Шаг | Было (вход → выход) | Будет |
|---|---|---|
| 1 | SMILES → `{имя: мол}` ×2 | SMILES → мол (`Role=monomer`) и мол (`Role=trimer`) |
| 2 | 2 словаря → `match_dict` (5 словарей) | мономер + тример → остаток с `SrcIdx`, `TrimerIdx` |
| 2.2 | `3_<имя>.pdb/.smiles` | `save_mol` (по желанию) |
| 3.1 | PDB → мол + `ParentAtoms`, `OldIndices` | остаток → + `ParentName`, `PrevIdx` |
| 3.2 | + `residue_params` (словарь) | + `AtomName`, `ResName`, `ResId`, `Chain`, `Naming` |
| 4 | `4_<имя>.pdb` + `general_atoms.json` | `save_mol` → `4_<имя>.sdf` (+ PDB для просмотра) |
| 4.1 | + `AddedH`, `<PTM>_rn_H_3D.pdb` | + `Added`; PDB для acpype пишется из мол |
| 5/6 | `constraints` + `AI_/RESP_chrges_*.json` | + `ChargeFixed`, `ChargeEspaloma`/`ChargeRESP` |
| 2 (acpype) | `chargeVal` руками, `zip` имён | заряд из мол, + `AcpypeType`, `AcpypeItp` |
| 3 (поле) | itp + json + `aminoacids_dict.json` | мол + itp + `read_rtp` → `AtomType`; пишет rtp/hdb/... |

## Решения (2026-10-10)
- Отдельного `Charge` нет: заряд каждого метода - своё свойство `Charge<Метод>`, используемый
  выбирается по `ChargeMethod`. Методов в будущем станет больше (другие инструменты), поэтому
  имя свойства строится из имени метода, а не перечисляется в коде.
- Старые ноутбуки и функции со словарями: обёртка распознаёт словарь, предупреждает и
  работает по-старому (правило совместимости), новые функции принимают одну молекулу.
