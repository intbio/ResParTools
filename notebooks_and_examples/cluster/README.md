# RESP на кластере (шаг 6, модификации лизина)

Скрипт `resp_cluster.py` готовит и запускает расчёт зарядов RESP для модификаций лизина
по протоколу шага 6 (ACE-остаток-NME, 2 конформации остова αR/β × 3 конформации боковой
цепи, оптимизация HF/6-31G* с замороженными φ/ψ, ESP MSK, двухстадийный RESP).
Входные данные - файлы шагов 1-4 (`molecules/substructure/4_<имя>.pdb`), они лежат в git.
Флуоресцентные метки (AF_*) сюда не входят.

## 1. Подготовка (один раз)
```bash
git clone https://github.com/intbio/ResParTools.git      # или git pull в существующей копии
cd ResParTools
conda env create -n darwin_resp -f psiresp_min.yml       # если подбор не сойдётся: -f darwin_resp.lock.yml
cd notebooks_and_examples
```
Если на кластере conda подключается через `module load ...`, допишите эту команду в
`CLUSTER_ENV_SETUP` (`respartools/resp.py`) или передавайте её при подготовке задач.

## 2. Проверка кластера (около 15 минут)
```bash
conda activate darwin_resp
python cluster/resp_cluster.py check --cpus 24 --mem 32G          # --partition <имя>, если нужно
cd RESP_data/cluster_check && sbatch run_check.sbatch
```
Задача замеряет градиент psi4 HF/6-31G* на 1, 6, 12 и 24 потоках и проверяет всю цепочку
RESP на маленькой задаче (HF/STO-3G, одна конформация).

**Прислать после проверки:**
- `RESP_data/cluster_check/bench.txt` - время градиента по числу потоков;
- `RESP_data/cluster_check/slurm_<номер>.out` - весь вывод задачи;
- `RESP_data/cluster_check/cluster_test/result.json` - результат пробной задачи (если есть).

## 3. Расчёт RESP для лизинов (после проверки)
```bash
python cluster/resp_cluster.py prepare             # все лизины; или перечислить папки: Lysine_Formyl Lysine_3M
python cluster/resp_cluster.py submit --chain      # по очереди: каждая задача стартует после предыдущей
python cluster/resp_cluster.py status              # сколько конформаций сошлось
python cluster/resp_cluster.py collect             # заряды -> RESP_chrges_<имя>.json + сравнение с Espaloma
```
Одна модификация - одна задача SLURM (по умолчанию 24 ядра, 32 ГБ, 8 часов): 6 конформаций
считаются одновременно по 4 потока. Если задачу оборвёт лимит времени, повторный
`submit` продолжит с сохранённых геометрий (готовые конформации и расчёты ESP не
пересчитываются).

Файлы задач - `<папка модификации>/RESP_data/<имя>_capped/` (в git не попадают):
`spec.json` (молекула, ограничения, настройки), `conformers/<метка>/` (start.xyz, opt.out,
status.json, optimized.xyz), `psiresp/` (ESP), `result.json` (заряды).
