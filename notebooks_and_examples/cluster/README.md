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
python cluster/resp_cluster.py check          # по умолчанию: 24 ядра, 30G, раздел intbio
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
python cluster/resp_cluster.py review > cluster/reviews/review_capped_NN_<дата>.txt  # проверка ДО запуска (QM ещё не считается)
python cluster/resp_cluster.py submit              # все задачи сразу (параллельно, ~2-3 ч при свободных узлах)
# python cluster/resp_cluster.py submit --chain    # или по очереди: каждая после предыдущей (12 задач - 10-20 ч)
python cluster/resp_cluster.py status              # сколько конформаций сошлось
python cluster/resp_cluster.py collect             # заряды -> <папка>/data/RESP_chrges_<имя>.json + сравнение с Espaloma
git add */data/RESP_chrges_*.json && git commit -m "RESP: заряды с кластера" && git push
```
Если конформация не сошлась за 400 шагов: `status` печатает причину и историю запусков: `сделано/лимит: энергия`
(сделано - вычислений градиента, одно на шаг geomeTRIC, включая отвергнутые; энергия последней
геометрии, Хартри; в задачах до 2026-10-22 - только лимит). Шаг, на котором упал psi4, - в
`conformers/<метка>/geometric.log` и `opt.out`. Если в `conformers/<метка>/opt.out` и `worker.log` нет критических
ошибок (SCF, развал геометрии - тогда есть `failed.xyz`), снова `submit`: несошедшиеся конформации
продолжаются с `current.xyz` ещё на 100 шагов (`continue_steps`), готовые не пересчитываются. Повторять,
пока не сойдётся или энергия не выйдет на плато (почти не меняется между запусками) - тогда решаем отдельно.

Оптимизатор - geomeTRIC (`optimizer='geometric'`, по умолчанию с 2026-10-14; лог шагов -
`conformers/<метка>/geometric.log`): шаг, повышающий энергию, отвергается, радиус доверия уменьшается.
Встроенный optking psi4 (`optimizer='optking'`, задачи до 2026-10-14) на плато у минимума делал шаг
~0.4 бор и не отвергал его: молекула разваливалась, затем падал SCF (8 из 72 конформаций). После
оптимизации связи сверяются со `start.xyz`: связь длиннее в 1.3 раза - конформация не сошлась
(`разорваны связи ...` в `status`), геометрия в `failed.xyz`, продолжать с неё нельзя.

Вывод `review` хранится в git: `cluster/reviews/review_capped_<NN>_<дата>.txt`, NN - номер задачи
(`_01` - первый прогон, optking; `_02` - пересчёт, geomeTRIC). `review` для каждой задачи показывает: молекулу ACE-остаток-NME (SMILES), заряды молекулы и
остатка, фиксированные заряды по остаткам (кэп ACE и NME - из rtp amber14sb, остов - из
родительского остатка) с суммами, эквивалентные атомы, стереохимию (CA - L), замороженные
углы и конформации; картинка `review.png` в папке задачи - подсвечены фиксированные атомы.

Одна модификация - одна задача SLURM (по умолчанию 24 ядра, 30 ГБ, 8 часов, раздел intbio): 6 конформаций
считаются одновременно по 4 потока. Каждая конформация оптимизируется одним расчётом psi4 + geomeTRIC
(до 400 шагов). Повторный `submit` не пересчитывает готовые конформации и расчёты ESP;
несошедшаяся конформация продолжается с `current.xyz`, а оборванная лимитом времени -
начинается заново.

Файлы задач - `<папка модификации>/RESP_data/<имя>_capped/` (в git не попадают):
`spec.json` (молекула, ограничения, настройки), `conformers/<метка>/` (start.xyz, opt.out,
status.json, optimized.xyz; current.xyz - не сошлось, failed.xyz - psi4 упал или разорваны связи; geometric.log), `psiresp/` (ESP),
`result.json` (заряды). Если `prepare` запущен снова с другими конформациями или настройками,
старая задача не трогается, новая пишется в `<имя>_capped_01` (`_02`...), а файл
`<имя>_capped.current` указывает текущую - её берут `submit`, `status`, `review`, `collect`.
