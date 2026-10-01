"""
RESP (шаг 6) для модификаций лизина на кластере SLURM.

Входные данные - результаты шагов 1-4 новых ноутбуков 1_charge_calculation.ipynb в папках
модификаций (molecules/substructure/4_<имя>.pdb, лежат в git). Задачи пишутся в
<папка модификации>/RESP_data/<имя>_capped/ (в git не попадают).

Команды (запуск из notebooks_and_examples/, окружение darwin_resp):
    python cluster/resp_cluster.py check  [--cpus 24 --mem 32G --partition P] [--submit]
        проверка узла: замер psi4 на 1..cpus потоках и маленькая задача RESP целиком
    python cluster/resp_cluster.py prepare [папки ...]
        подготовить задачи RESP (конформации, ограничения, run_resp.sbatch)
    python cluster/resp_cluster.py review  [папки ...]
        проверка перед запуском: молекула ACE-остаток-NME, фиксированные заряды по остаткам,
        эквивалентные атомы, конформации, стереохимия; картинка review.png в папке задачи
    python cluster/resp_cluster.py submit  [папки ...] [--chain]
        отправить задачи; --chain - по очереди (каждая следующая после предыдущей)
    python cluster/resp_cluster.py status  [папки ...]
        сколько конформаций сошлось, есть ли результат
    python cluster/resp_cluster.py collect [папки ...]
        прочитать заряды, проверить, сравнить с Espaloma, записать RESP_chrges_<имя>.json
Папки по умолчанию - все модификации лизина из make_charge_notebooks.PARAMS
(флуоресцентные метки не входят: для них RESP считается отдельно).
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                     # notebooks_and_examples/
sys.path.insert(0, ROOT)

import ResParTools as pt                          # noqa: E402
from make_charge_notebooks import PARAMS          # noqa: E402

LYSINES = [f for f in PARAMS if f.startswith('Lysine_')]
# кластер intbio: один раздел, узлы от 24 ядер и 31871 МБ (32G не помещается)
SLURM = dict(cpus=24, mem='30G', time_limit='08:00:00', partition='intbio')


def _residue(folder):
    """Остаток шага 4 и ограничения остова - как в шаге 5/6 ноутбука."""
    p = PARAMS[folder]
    mon = os.path.splitext(os.path.basename(p['monomer_file']))[0]
    path = os.path.join(ROOT, folder, 'molecules', 'substructure', f'4_{mon}.pdb')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{folder}: нет {path} - выполните шаги 1-4 ноутбука')
    residue = list(pt.file_opener(path).values())[0]
    constraints = pt.charge_constraints_from_rtp(residue, pt.AMINO_ACIDS[p['parent_residue']][0].upper())
    return mon, residue, constraints


def _monomer(folder):
    """Мономер шага 1 - источник стереохимии для RESP (capped_monomer)."""
    return list(pt.file_opener(os.path.join(ROOT, folder, PARAMS[folder]['monomer_file'])).values())[0]


def _job_dir(folder, mon):
    return os.path.join(ROOT, folder, 'RESP_data', f'{mon}_capped')


def cmd_check(args):
    mon, residue, constraints = _residue('Lysine_Formyl')
    folder = os.path.join(ROOT, 'RESP_data', 'cluster_check')
    pt.prepare_cluster_check(residue, constraints, folder=folder, cpus=args.cpus, mem=args.mem,
                             partition=args.partition, monomer=_monomer('Lysine_Formyl'))
    if args.submit:
        _sbatch(folder, 'run_check.sbatch')


def cmd_prepare(args):
    for folder in args.folders:
        mon, residue, constraints = _residue(folder)
        print(f'\n===== {folder}')
        job = pt.prepare_resp_job(residue, constraints, f'{mon}_capped',
                                  working_dir=os.path.join(ROOT, folder, 'RESP_data'),
                                  fix_backbone=True, n_sidechain=3, monomer=_monomer(folder))
        pt.write_slurm_script(job, cpus=args.cpus, mem=args.mem, partition=args.partition,
                              time_limit=args.time)


def _sbatch(folder, script, dependency=None):
    cmd = ['sbatch', '--parsable']
    if dependency:
        cmd.append(f'--dependency=afterany:{dependency}')
    cmd.append(script)
    out = subprocess.run(cmd, cwd=folder, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f'sbatch в {folder}: {out.stderr.strip()}')
    job_id = out.stdout.strip().split(';')[0]
    print(f'отправлено: {folder} -> задача {job_id}' + (f' (после {dependency})' if dependency else ''))
    return job_id


def cmd_submit(args):
    previous = None
    for folder in args.folders:
        mon = os.path.splitext(os.path.basename(PARAMS[folder]['monomer_file']))[0]
        job = _job_dir(folder, mon)
        if os.path.exists(os.path.join(job, pt.RESULT_FILE)):
            print(f'{folder}: уже посчитано, пропуск')
            continue
        if not os.path.exists(os.path.join(job, 'run_resp.sbatch')):
            raise FileNotFoundError(f'{folder}: нет run_resp.sbatch - сначала prepare')
        job_id = _sbatch(job, 'run_resp.sbatch', previous if args.chain else None)
        previous = job_id


def cmd_status(args):
    for folder in args.folders:
        mon = os.path.splitext(os.path.basename(PARAMS[folder]['monomer_file']))[0]
        job = _job_dir(folder, mon)
        if not os.path.exists(os.path.join(job, pt.SPEC_FILE)):
            print(f'{folder:15s} не подготовлено')
            continue
        statuses = []
        for st in sorted(glob.glob(os.path.join(job, 'conformers', '*', 'status.json'))):
            statuses.append(json.load(open(st)).get('converged'))
        n = len(json.load(open(os.path.join(job, pt.SPEC_FILE)))['conformers'])
        done = os.path.exists(os.path.join(job, pt.RESULT_FILE))
        print(f'{folder:15s} конформаций сошлось {sum(1 for s in statuses if s)}/{n}'
              f'{" | не сошлись: " + str(sum(1 for s in statuses if s is False)) if False in statuses else ""}'
              f' | результат: {"есть" if done else "нет"}')


def cmd_review(args):
    for folder in args.folders:
        mon = os.path.splitext(os.path.basename(PARAMS[folder]['monomer_file']))[0]
        job = _job_dir(folder, mon)
        if not os.path.exists(os.path.join(job, pt.SPEC_FILE)):
            print(f'{folder}: не подготовлено (prepare)')
            continue
        print()
        pt.review_resp_job(job)


def cmd_collect(args):
    for folder in args.folders:
        mon, residue, _ = _residue(folder)
        job = _job_dir(folder, mon)
        print(f'\n===== {folder}')
        try:
            q = pt.load_resp_result(job)
        except FileNotFoundError as e:
            print(e)
            continue
        espaloma = os.path.join(ROOT, folder, f'AI_chrges_{mon}.json')
        sets = {'RESP': q}
        if os.path.exists(espaloma):
            sets = {'Espaloma': json.load(open(espaloma)), 'RESP': q}
        pt.compare_charges(residue, sets)
        cwd = os.getcwd()
        os.chdir(os.path.join(ROOT, folder))
        try:
            pt.save_charges_json(name=f'RESP_chrges_{mon}', charge_list=[float(x) for x in q])
        finally:
            os.chdir(cwd)


def main(argv=None):
    parser = argparse.ArgumentParser(description='RESP на кластере SLURM (см. описание в начале файла).')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('check', 'prepare', 'review', 'submit', 'status', 'collect'):
        p = sub.add_parser(name)
        if name != 'check':
            p.add_argument('folders', nargs='*', help='папки модификаций (по умолчанию все лизины)')
        if name in ('check', 'prepare'):
            p.add_argument('--cpus', type=int, default=SLURM['cpus'])
            p.add_argument('--mem', default=SLURM['mem'])
            p.add_argument('--partition', default=SLURM['partition'])
        if name == 'prepare':
            p.add_argument('--time', default=SLURM['time_limit'])
        if name == 'check':
            p.add_argument('--submit', action='store_true', help='сразу отправить в SLURM')
        if name == 'submit':
            p.add_argument('--chain', action='store_true', help='по очереди: каждая задача после предыдущей')
    args = parser.parse_args(argv)
    if getattr(args, 'folders', None) is not None:
        unknown = [f for f in args.folders if f not in PARAMS]
        if unknown:
            sys.exit(f'Неизвестные папки: {unknown}')
        args.folders = args.folders or LYSINES
    {'check': cmd_check, 'prepare': cmd_prepare, 'review': cmd_review, 'submit': cmd_submit,
     'status': cmd_status, 'collect': cmd_collect}[args.command](args)


if __name__ == '__main__':
    main()
