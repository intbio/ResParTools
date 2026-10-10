"""
Проверка остатка в GROMACS: пептид ACE-X-NME на новом силовом поле.

1. pdb2gmx -ignh: остаток есть в rtp, имена атомов совпадают, водороды строятся по hdb
   (число и имена сверяются с исходным пептидом), остаток распознан как белок.
2. grompp + минимизация в вакууме: у всех связей, углов, диэдралов есть параметры, у всех
   типов - параметры Леннард-Джонса, заряд целый, энергия конечна.
3. (md_ns > 0) вода TIP3P, ионы до нейтральности, минимизация, NVT 100 пс, NPT 100 пс,
   md_ns нс: система устойчива (нет предупреждений LINCS, температура, плотность), длины
   связей остатка по траектории не уходят от исходной геометрии.

Все команды GROMACS - в одном скрипте run_validate.sh в папке задачи: локально он
запускается напрямую (run='local'), на кластере - через run_validate.sbatch (run='slurm').
"""
import os
import re
import shutil
import subprocess
import sys

import numpy as np
from rdkit import Chem

from .log import logged
from .utils import print_red, print_green

_MDP_COMMON = """cutoff-scheme = Verlet
pbc = xyz
coulombtype = PME
rcoulomb = 1.0
rvdw = 1.0
DispCorr = EnerPres
"""
_MDP = {
    # вакуум: заряженный пептид (метиллизины +1, малонил -1) с PME даёт предупреждение
    # grompp о заряде системы - для проверки поля берётся reaction-field
    'em_vac.mdp': "integrator = steep\nnsteps = 5000\nemtol = 100\n"
                  + _MDP_COMMON.replace('coulombtype = PME', 'coulombtype = Reaction-Field'),
    'ions.mdp': "integrator = steep\nnsteps = 0\n" + _MDP_COMMON,
    'em.mdp': "integrator = steep\nnsteps = 5000\nemtol = 1000\n" + _MDP_COMMON,
    'nvt.mdp': """integrator = md
dt = 0.002
nsteps = 50000
constraints = h-bonds
gen-vel = yes
gen-temp = 300
tcoupl = V-rescale
tc-grps = System
tau-t = 0.1
ref-t = 300
nstenergy = 500
nstlog = 5000
""" + _MDP_COMMON,
    'npt.mdp': """integrator = md
dt = 0.002
nsteps = 50000
constraints = h-bonds
continuation = yes
tcoupl = V-rescale
tc-grps = System
tau-t = 0.1
ref-t = 300
pcoupl = C-rescale
tau-p = 1.0
ref-p = 1.0
compressibility = 4.5e-5
nstenergy = 500
nstlog = 5000
""" + _MDP_COMMON,
}

_SCRIPT_VACUUM = """#!/bin/bash
# проверка остатка в GROMACS (respartools.gmx); GMX - команда gromacs, NT - потоки
set -e
cd "$(dirname "$0")"
G=${{GMX:-gmx}}
NT=${{NT:-4}}
rm -f status.txt
$G pdb2gmx -f peptide.pdb -o conf.gro -p topol.top -ff {ff} -water tip3p -ignh > pdb2gmx.log 2>&1
echo pdb2gmx >> status.txt
$G editconf -f conf.gro -o vac_box.gro -c -d 1.5 -bt cubic > editconf_vac.log 2>&1
$G grompp -f em_vac.mdp -c vac_box.gro -p topol.top -o em_vac.tpr -maxwarn 0 > grompp_em_vac.log 2>&1
echo grompp >> status.txt
$G mdrun -deffnm em_vac -nt $NT > mdrun_em_vac.log 2>&1
echo em_vac >> status.txt
"""
_SCRIPT_MD = """cp topol.top topol_vacuum.top
$G editconf -f em_vac.gro -o box.gro -c -d 1.0 -bt dodecahedron > editconf.log 2>&1
$G solvate -cp box.gro -cs spc216.gro -o solv.gro -p topol.top > solvate.log 2>&1
# до genion система может быть заряжена (предупреждение о PME и заряде) - ионы это и исправляют
$G grompp -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 1 > grompp_ions.log 2>&1
echo SOL | $G genion -s ions.tpr -o ions.gro -p topol.top -pname NA -nname CL -neutral > genion.log 2>&1
$G grompp -f em.mdp -c ions.gro -p topol.top -o em.tpr -maxwarn 0 > grompp_em.log 2>&1
$G mdrun -deffnm em -nt $NT > mdrun_em.log 2>&1
echo em >> status.txt
$G grompp -f nvt.mdp -c em.gro -p topol.top -o nvt.tpr -maxwarn 0 > grompp_nvt.log 2>&1
$G mdrun -deffnm nvt -nt $NT > mdrun_nvt.log 2>&1
echo nvt >> status.txt
$G grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 0 > grompp_npt.log 2>&1
$G mdrun -deffnm npt -nt $NT > mdrun_npt.log 2>&1
echo npt >> status.txt
$G grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 0 > grompp_md.log 2>&1
$G mdrun -deffnm md -nt $NT -cpi md.cpt > mdrun_md.log 2>&1
echo md >> status.txt
printf "Temperature\\nPressure\\nDensity\\nPotential\\n\\n" | $G energy -f md.edr -o md_energy.xvg > md_energy.txt 2>&1
echo Protein | $G trjconv -s md.tpr -f md.xtc -o md_peptide.pdb -dt 20 -pbc mol > trjconv.log 2>&1
echo done >> status.txt
"""


def capped_peptide(residue, monomer, random_seed=42):
    """
    Пептид ACE-X-NME для проверки: остаток шага 4 (имена атомов, параметры остатка) с кэпами
    amber, стереохимия - из мономера (CA - L), 3D-координаты, атомы в порядке остатков
    (ACE, X, NME), как требует pdb2gmx.
    """
    capped = cap_residue(residue)
    mol = _with_reference_stereo(capped, capped_monomer(monomer, random_seed=random_seed))
    order = sorted(range(mol.GetNumAtoms()),
                   key=lambda i: (mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetResidueNumber(), i))
    mol = Chem.RenumberAtoms(mol, order)
    return set_mol_coords(mol, coords='3D', ca_config='L', random_seed=random_seed)


@logged
def write_validate_job(residue, monomer, ff_dir, workdir, md_ns=0.0):
    """
    Папка задачи проверки: peptide.pdb (ACE-X-NME), ссылка на поле ff_dir, residuetypes.dat
    поля, mdp-файлы и run_validate.sh. md_ns > 0 - добавить МД в воде такой длины (нс).
    Возвращает путь к папке.
    """
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)
    ff_dir = os.path.abspath(ff_dir)
    ff_name = os.path.basename(ff_dir)[:-3] if ff_dir.endswith('.ff') else os.path.basename(ff_dir)
    link = os.path.join(workdir, f'{ff_name}.ff')
    if os.path.islink(link):
        os.remove(link)
    elif os.path.exists(link):
        shutil.rmtree(link)
    os.symlink(ff_dir, link)
    shutil.copy(os.path.join(ff_dir, 'residuetypes.dat'), workdir)   # GROMACS ищет его в рабочей папке
    peptide = capped_peptide(residue, monomer)
    Chem.MolToPDBFile(peptide, os.path.join(workdir, 'peptide.pdb'))
    for name, text in _MDP.items():
        with open(os.path.join(workdir, name), 'w') as f:
            f.write(text)
    script = _SCRIPT_VACUUM.format(ff=ff_name)
    if md_ns > 0:
        nsteps = int(round(md_ns * 500000))
        md = _MDP['npt.mdp'].replace('nsteps = 50000', f'nsteps = {nsteps}').replace(
            'nstenergy = 500', 'nstenergy = 5000\nnstxout-compressed = 5000')
        with open(os.path.join(workdir, 'md.mdp'), 'w') as f:
            f.write(md)
        script += _SCRIPT_MD
    path = os.path.join(workdir, 'run_validate.sh')
    with open(path, 'w') as f:
        f.write(script)
    os.chmod(path, 0o755)
    resname = residue.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName().strip()
    print(f'Проверка {resname} в GROMACS: {workdir} (ACE-{resname}-NME, {peptide.GetNumAtoms()} атомов'
          + (f', МД {md_ns} нс)' if md_ns > 0 else ', вакуум)'))
    return workdir


def _gmx_bin(gmx=None):
    """gmx: переданный, из PATH или из окружения gmx рядом с текущим."""
    if gmx:
        return gmx
    found = shutil.which('gmx')
    if found:
        return found
    envs = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
    candidate = os.path.join(envs, 'gmx', 'bin', 'gmx')
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError('Нет gmx: conda create -n gmx -c conda-forge gromacs или передайте gmx.')


@logged
def run_validate(workdir, run='local', n_threads=4, gmx=None, slurm=None):
    """
    Запуск run_validate.sh: run='local' - здесь (ждёт окончания, возвращает validate_report),
    run='slurm' - пишет run_validate.sbatch (slurm: cpus, mem, time_limit, partition,
    env_setup - команды, после которых в PATH есть gmx); отправка - sbatch в папке задачи.
    """
    if run == 'slurm':
        s = dict(cpus=8, mem='4G', time_limit='04:00:00', partition=None, env_setup='module load gromacs')
        s.update(slurm or {})
        lines = ['#!/bin/bash', '#SBATCH --job-name=validate', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
                 f"#SBATCH --cpus-per-task={s['cpus']}", f"#SBATCH --mem={s['mem']}",
                 f"#SBATCH --time={s['time_limit']}", '#SBATCH --output=slurm_%j.out']
        if s['partition']:
            lines.append(f"#SBATCH --partition={s['partition']}")
        lines += ['', 'cd "$SLURM_SUBMIT_DIR"', s['env_setup'], f"NT={s['cpus']} bash run_validate.sh", '']
        path = os.path.join(workdir, 'run_validate.sbatch')
        with open(path, 'w') as f:
            f.write('\n'.join(lines))
        print(f'SLURM: {path}. Отправка: cd {workdir} && sbatch run_validate.sbatch')
        return path
    if run != 'local':
        raise ValueError(f"run='{run}': допустимо 'local' или 'slurm'")
    env = dict(os.environ, GMX=_gmx_bin(gmx), NT=str(n_threads))
    subprocess.run(['bash', os.path.join(workdir, 'run_validate.sh')], env=env)
    return validate_report(workdir)


def _read(path):
    return open(path).read() if os.path.exists(path) else ''


def _pdb_atoms(text):
    """[(остаток, номер, имя, xyz)] атомов одной модели PDB."""
    return [(line[17:20].strip(), int(line[22:26]), line[12:16].strip(),
             np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]))
            for line in text.splitlines() if line.startswith(('ATOM', 'HETATM'))]


def _gro_atoms(text):
    """[(остаток, номер, имя, xyz в Å)] атомов .gro."""
    return [(line[5:10].strip(), int(line[0:5]), line[10:15].strip(),
             10 * np.array([float(line[20:28]), float(line[28:36]), float(line[36:44])]))
            for line in text.splitlines()[2:-1]]


def validate_report(workdir, bond_tolerance=0.1):
    """
    Итог проверки по файлам папки задачи (локальной или с кластера). Печатает и возвращает
    словарь: пройденные этапы, совпадение атомов после pdb2gmx -ignh, предупреждения
    grompp, сходимость минимизации, для МД - предупреждения LINCS, средние температура,
    давление, плотность, связи остатка, средняя длина которых в МД отличается от длины
    после минимизации в вакууме (em_vac.gro - минимум этого же поля) больше bond_tolerance (Å).
    """
    stages = _read(os.path.join(workdir, 'status.txt')).split()
    rep = {'stages': stages, 'ok': True, 'problems': []}

    def problem(text):
        rep['ok'] = False
        rep['problems'].append(text)

    pdb = _read(os.path.join(workdir, 'peptide.pdb'))
    start = _pdb_atoms(pdb)
    if 'pdb2gmx' not in stages:
        err = re.findall(r'Fatal error:\n(.+)', _read(os.path.join(workdir, 'pdb2gmx.log')))
        problem(f"pdb2gmx: {err[0] if err else 'не выполнен'}")
    else:
        gro = _read(os.path.join(workdir, 'conf.gro')).splitlines()[2:-1]
        got = sorted((line[5:10].strip(), line[10:15].strip()) for line in gro)
        want = sorted((r, n) for r, _, n, _ in start)
        rep['atoms_match'] = got == want
        if got != want:
            problem(f'pdb2gmx -ignh: атомы не совпали (лишние {sorted(set(got) - set(want))[:6]}, '
                    f'нет {sorted(set(want) - set(got))[:6]}) - проверьте hdb')
    for log in ('grompp_em_vac.log', 'grompp_em.log', 'grompp_md.log'):
        text = _read(os.path.join(workdir, log))
        if text:
            warns = re.findall(r'WARNING \d+ \[[^\]]*\]:\n(.+)', text)
            fatal = re.findall(r'Fatal error:\n(.+)', text)
            if warns or fatal:
                problem(f'{log}: ' + '; '.join(w.strip() for w in fatal + warns)[:300])
    em = _read(os.path.join(workdir, 'em_vac.log'))
    fmax = re.findall(r'Maximum force\s+=\s+(\S+)', em)
    rep['em_vacuum_fmax'] = float(fmax[-1]) if fmax else None
    if 'em_vac' in stages and not re.search(r'converged to Fmax', em):
        problem('минимизация в вакууме не сошлась')
    if 'md' in stages or os.path.exists(os.path.join(workdir, 'md.mdp')):
        md_log = _read(os.path.join(workdir, 'md.log')) + _read(os.path.join(workdir, 'nvt.log')) + \
            _read(os.path.join(workdir, 'npt.log'))
        rep['lincs_warnings'] = md_log.count('LINCS WARNING')
        if rep['lincs_warnings']:
            problem(f"предупреждений LINCS: {rep['lincs_warnings']}")
        if 'done' not in stages:
            problem(f"МД не закончена (этапы: {' '.join(stages)})")
        energy = _read(os.path.join(workdir, 'md_energy.txt'))
        for key in ('Temperature', 'Pressure', 'Density', 'Potential'):
            m = re.search(rf'^{key}\s+(\S+)\s+(\S+)\s+(\S+)', energy, re.M)
            if m:
                rep[key.lower()] = float(m.group(1))
        if 'temperature' in rep and abs(rep['temperature'] - 300) > 5:
            problem(f"средняя температура {rep['temperature']:.1f} K")
        if 'density' in rep and not 950 < rep['density'] < 1050:
            problem(f"средняя плотность {rep['density']:.0f} кг/м3")
        frames = [_pdb_atoms(m) for m in _read(os.path.join(workdir, 'md_peptide.pdb')).split('ENDMDL') if 'ATOM' in m]
        ref = _gro_atoms(_read(os.path.join(workdir, 'em_vac.gro')))
        if frames and pdb and ref:
            mol = Chem.MolFromPDBBlock(pdb, removeHs=False)
            key = [(r, i, n) for r, i, n, _ in start]
            ref_pos = {(r, i, n): x for r, i, n, x in ref}
            frame_key = [(r, i, n) for r, i, n, _ in frames[0]]
            if sorted(frame_key) == sorted(key):
                pos = [{(r, i, n): x for r, i, n, x in fr} for fr in frames]
                drift = []
                for b in mol.GetBonds():
                    a1, a2 = key[b.GetBeginAtomIdx()], key[b.GetEndAtomIdx()]
                    if a1[1] != 2 and a2[1] != 2:
                        continue
                    d0 = np.linalg.norm(ref_pos[a1] - ref_pos[a2])
                    d = np.mean([np.linalg.norm(p[a1] - p[a2]) for p in pos])
                    if abs(d - d0) > bond_tolerance:
                        drift.append(f'{a1[2]}-{a2[2]} {d0:.2f}->{d:.2f}')
                rep['bond_drift'] = drift
                if drift:
                    problem('связи остатка в МД уходят от минимума в вакууме (Å): ' + ', '.join(drift[:8]))
    resname = next((r for r, i, _, _ in start if i == 2), os.path.basename(workdir))
    if rep['ok']:
        extra = ''
        if 'temperature' in rep:
            extra = (f", МД: T {rep['temperature']:.1f} K, плотность {rep.get('density', 0):.0f} кг/м3, "
                     f"LINCS {rep['lincs_warnings']}")
        print_green(f"✓ {resname}: {' -> '.join(stages)}; Fmax в вакууме {rep['em_vacuum_fmax']}{extra}")
    else:
        print_red(f"✗ {resname}: " + ' | '.join(rep['problems']))
    return rep


from .resp import cap_residue, capped_monomer, _with_reference_stereo  # noqa: E402
from .residue import set_mol_coords  # noqa: E402
