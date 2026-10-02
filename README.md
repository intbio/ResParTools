# ResParTools 🧬

**ResParTools** - набор инструментов для параметризации модифицированных аминокислотных
остатков (посттрансляционные модификации, флуоресцентные метки) для силового поля
**amber14sb** в GROMACS. На выходе - имена атомов по правилам amber, топология остатка
(`.rtp`, `.hdb`, `.r2b`, `.atp`) и парциальные заряды.

Заряды считаются двумя способами на одних и тех же входных данных:
- **Espaloma Charge** - нейросеть ([choderalab/espaloma-charge](https://github.com/choderalab/espaloma-charge)),
  модифицированная копия умеет фиксировать заряды выбранных атомов (остов берётся из amber14sb);
- **RESP** - протокол зарядов amber (Cornell 1995): ACE-X-NME, конформации αR/β,
  оптимизация HF/6-31G* в [psi4](https://psicode.org), ESP через [psiresp](https://github.com/lilyminium/psiresp),
  двухстадийный фит.

---

## Пайплайн

Для каждой модификации - папка `notebooks_and_examples/<Lysine_XXX | AF_*>/`:

1. `1_charge_calculation.ipynb` - остаток из SMILES мономера и тримера: поиск мономера в
   тримере, перенумерация по шаблону родительской аминокислоты, имена атомов, протоны для
   acpype, заряды Espaloma (шаг 5) и RESP (шаг 6).
2. `2_generate_topology.ipynb` - топология `itp` через [acpype](https://github.com/alanwilter/acpype).
3. `../3_edd_topology.ipynb` - `rtp` остатка и файлы для модифицированного силового поля
   (`notebooks_and_examples/amber14sb_mod.ff/`).

Подробная карта шагов, функций и файлов - [`docs/PIPELINE.md`](docs/PIPELINE.md);
текущий статус и план - [`docs/PLAN.md`](docs/PLAN.md).

Новые ноутбуки `1_charge_calculation.ipynb` собираются из шаблона
`notebooks_and_examples/templates/` скриптом `make_charge_notebooks.py` (параметры каждой
модификации - в `PARAMS`). Прежние версии ноутбуков сохранены (`*_old.ipynb`).

## Установка окружений

| Окружение | Файл | Для чего |
|---|---|---|
| `darwin_ec` | `ResParTools_ec.yml` | ноутбуки, шаги 1-5 (Espaloma), Python 3.12 |
| `darwin_resp` | `psiresp_min.yml` (полный снимок `darwin_resp.lock.yml`) | RESP (шаг 6), тесты; psi4 1.6.1, psiresp 0.4.2, Python 3.9 |
| `espaloma` | `ai_topmol.yml` | прежнее окружение для Espaloma, Python 3.9 |

```bash
# без видеокарты NVIDIA torch и dgl работают на CPU
CONDA_OVERRIDE_CUDA=12.4 conda env create -n darwin_ec -f ResParTools_ec.yml
conda remove -n darwin_ec --force -y espaloma_charge   # иначе не подхватится espaloma-charge_mod

conda env create -n darwin_resp -f psiresp_min.yml
```

Для шагов 2-3 нужно окружение с `acpype` и `parmed` (Python 3.8).

## Расчёт RESP на кластере

Квантовая химия считается на кластере SLURM: `notebooks_and_examples/cluster/resp_cluster.py`
(`check`, `prepare`, `review`, `submit --chain`, `status`, `collect`).
Инструкция - [`notebooks_and_examples/cluster/README.md`](notebooks_and_examples/cluster/README.md).

## Тесты

Эталонный набор (Lysine_prop, Lysine_Malonyl, Lysine_Formyl, Lysine_3M, AF_546_cys):
шаги 1-4.1 сравниваются с файлами в репозитории, проверяются ограничения зарядов и подготовка
RESP (без квантовой химии).

```bash
conda activate darwin_resp
python -m pytest tests -q -m "not slow"   # ~30 с
python -m pytest tests -q                 # ~3 мин, вместе с AF546
```

## Структура репозитория

- `notebooks_and_examples/` - ноутбуки, папки модификаций, силовые поля
  - `respartools/` - библиотека (пакет); в ноутбуках подключается как `import ResParTools as pt`
  - `templates/`, `make_charge_notebooks.py` - шаблон и генератор ноутбуков
  - `cluster/` - запуск RESP на SLURM
  - `amber14sb_parmbsc1_cufix.ff/` - базовое силовое поле, `amber14sb_mod.ff/` - с модификациями
- `espaloma-charge/` - оригинальная Espaloma Charge
- `espaloma-charge_mod/` - модифицированная Espaloma Charge (фиксация зарядов атомов)
- `tests/` - тесты на эталонном наборе
- `docs/` - карта пайплайна и план работ

## Лицензия

GPL-3.0, см. [`LICENSE`](LICENSE).

## 📧 Контакты

Автор: Николай Кристовский
Почта: krist179@mail.ru
