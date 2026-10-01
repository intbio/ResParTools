"""
Заряды RESP (шаг 6) - альтернатива Espaloma (шаг 5) с теми же входными данными:
остаток шага 4 (имена атомов, порядок как в топологии) и ограничения charge_constraints_from_rtp.

Протокол (как для зарядов amber ff94/ff14SB/ff19SB, Cornell 1995; Cieplak 1995):
1. Молекула: остаток с кэпами ACE и NME (context='capped', по умолчанию) или тример шага 1
   (context='trimer'). Заряды кэпов фиксированы на значениях ACE/NME из aminoacids.rtp.
2. Конформации: пары углов остова (phi, psi) из fix_backbone (по умолчанию alphaR и beta)
   x n_sidechain конформаций боковой цепи. Пул RDKit ETKDG -> углы остова -> MMFF с
   замороженными phi/psi -> отбор низкоэнергетических и непохожих конформаций.
3. Оптимизация HF/6-31G* в psi4 (optking) с замороженными phi/psi, порциями: прерванный
   расчёт продолжается с последней геометрии. Конформации считаются параллельно.
4. ESP (сетка MSK) и двухстадийный RESP - psiresp, все конформации с равным весом.
   Оптимизацию psiresp не делает: в режиме без QCFractal он её не выполняет (считает
   только градиент), поэтому геометрии оптимизируются здесь, в psi4.

Запуск: локально (run='local'), подготовка задачи для SLURM (run='slurm') или только
подготовка без квантовой химии (run='prepare'). Задача - папка с spec.json; её же
выполняет командная строка (см. write_slurm_script): _main(<папка> --threads N --parallel M).
Нужно окружение darwin_resp (psi4 1.6.1, psiresp 0.4.2); psi4 и psiresp импортируются
внутри функций, остальной пакет работает без них.
"""
import os
import sys
import time
import json
import shutil
import argparse
import subprocess
from collections import defaultdict

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from rdkit.Chem import rdMolTransforms

from .log import logged, log_note
from .utils import format_time, print_red, _PACKAGE_ROOT

# пары (phi, psi) остова в градусах: области alphaR и beta карты Рамачандрана
AMBER_BACKBONE_CONFORMATIONS = {'alphaR': (-60.0, -40.0), 'beta': (-120.0, 130.0)}
RESP_CONTEXTS = ('capped', 'trimer')
RESP_RUN_MODES = ('local', 'slurm', 'prepare')
# активация окружения в задаче SLURM (неинтерактивный bash: conda.sh подключается явно)
CLUSTER_ENV_SETUP = 'source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate darwin_resp'
SPEC_FILE = 'spec.json'
# запуск модуля из командной строки (python -m respartools.resp даёт RuntimeWarning, т.к.
# пакет уже импортирует resp)
_WORKER = 'import sys; from respartools.resp import _main; _main(sys.argv[1:])'
RESULT_FILE = 'result.json'


# -----------------------------------------------------------------------------
# Молекула для расчёта: кэпы ACE / NME, углы остова
# -----------------------------------------------------------------------------

def _set_atom_name(atom, name, resname, resid, chain='A'):
    info = Chem.AtomPDBResidueInfo()
    info.SetName(f' {name:<3}' if len(name) < 4 else name)
    info.SetResidueName(resname)
    info.SetResidueNumber(resid)
    info.SetChainId(chain)
    atom.SetMonomerInfo(info)
    atom.SetProp('AtomName', name)


def cap_residue(residue_chem):
    """
    Пристраивает к остатку кэпы ACE (к N) и NME (к C) по открытым валентностям, как
    в дипептидах ACE-X-NME для зарядов amber. Атомы остатка сохраняют индексы 0..n-1,
    атомы кэпов идут после них и получают имена amber (ACE: HH31 CH3 HH32 HH33 C O;
    NME: N H CH3 HH31 HH32 HH33).

    Аргументы:
        residue_chem (Chem.Mol) - остаток шага 4 (явные водороды, у N и C по одной
            открытой валентности)
    Возвращает:
        Chem.Mol с 3D-координатами не задан (только граф). Свойства молекулы:
        'CapAtoms' - JSON {'ACE': {имя: индекс}, 'NME': {имя: индекс}},
        'BackboneDihedrals' - JSON {'phi': [4 индекса], 'psi': [4 индекса]}.
    """
    CA_idx, backbone = find_backbone_match(residue_chem)
    N_idx, C_idx = backbone[0], backbone[2]
    n = residue_chem.GetNumAtoms()
    for idx, what in ((N_idx, 'N'), (C_idx, 'C')):
        if residue_chem.GetAtomWithIdx(idx).GetNumImplicitHs() < 1:
            raise ValueError(f'У атома остова {what} ({idx}) нет открытой валентности для кэпа: '
                             'на вход нужен остаток шага 4 (без HW шага 4.1).')
    resid = 2
    resname = 'MOD'
    info = residue_chem.GetAtomWithIdx(0).GetPDBResidueInfo()
    if info is not None:
        resid, resname = info.GetResidueNumber(), info.GetResidueName()

    rw = Chem.RWMol(residue_chem)
    for atom in rw.GetAtoms():
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(0)
    # ACE: CH3-C(=O)- к N
    ace_c = rw.AddAtom(Chem.Atom(6))
    ace_o = rw.AddAtom(Chem.Atom(8))
    ace_ch3 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(ace_c, ace_o, Chem.BondType.DOUBLE)
    rw.AddBond(ace_c, ace_ch3, Chem.BondType.SINGLE)
    rw.AddBond(ace_c, N_idx, Chem.BondType.SINGLE)
    # NME: -N(H)-CH3 к C
    nme_n = rw.AddAtom(Chem.Atom(7))
    nme_ch3 = rw.AddAtom(Chem.Atom(6))
    rw.AddBond(nme_n, nme_ch3, Chem.BondType.SINGLE)
    rw.AddBond(C_idx, nme_n, Chem.BondType.SINGLE)
    capped = rw.GetMol()
    for idx in (ace_c, ace_o, ace_ch3, nme_n, nme_ch3):
        capped.GetAtomWithIdx(idx).SetNoImplicit(False)
    Chem.SanitizeMol(capped)
    capped = Chem.AddHs(capped)   # водороды только у атомов кэпов: у остатка NoImplicit

    names = {'ACE': {'C': ace_c, 'O': ace_o, 'CH3': ace_ch3},
             'NME': {'N': nme_n, 'CH3': nme_ch3}}
    for cap, parent in (('ACE', ace_ch3), ('NME', nme_ch3)):
        hs = sorted(a.GetIdx() for a in capped.GetAtomWithIdx(parent).GetNeighbors()
                    if a.GetAtomicNum() == 1)
        for k, h in enumerate(hs, 1):
            names[cap][f'HH3{k}'] = h
    nme_h = [a.GetIdx() for a in capped.GetAtomWithIdx(nme_n).GetNeighbors() if a.GetAtomicNum() == 1]
    names['NME']['H'] = nme_h[0]
    for cap, resid_cap in (('ACE', resid - 1), ('NME', resid + 1)):
        for name, idx in names[cap].items():
            _set_atom_name(capped.GetAtomWithIdx(idx), name, cap, resid_cap)
    extra = capped.GetNumAtoms() - n - 12   # ACE: C, O, CH3 + 3 H; NME: N, H, CH3 + 3 H
    if extra != 0:
        raise ValueError(f'Кэпы добавили неожиданное число атомов ({extra:+d}): проверьте остаток.')
    capped.SetProp('CapAtoms', json.dumps(names))
    capped.SetProp('BackboneDihedrals', json.dumps({'phi': [ace_c, N_idx, CA_idx, C_idx],
                                                    'psi': [N_idx, CA_idx, C_idx, nme_n]}))
    return capped


def context_backbone_dihedrals(context_chem, mapping, residue_chem):
    """phi/psi центрального остатка в тримере: соседние C и N берутся из графа тримера."""
    CA_idx, backbone = find_backbone_match(residue_chem)
    N, CA, C = (mapping[i] for i in (backbone[0], CA_idx, backbone[2]))
    inside = set(mapping)
    prev_c = [a.GetIdx() for a in context_chem.GetAtomWithIdx(N).GetNeighbors()
              if a.GetIdx() not in inside and a.GetAtomicNum() == 6]
    next_n = [a.GetIdx() for a in context_chem.GetAtomWithIdx(C).GetNeighbors()
              if a.GetIdx() not in inside and a.GetAtomicNum() == 7]
    if len(prev_c) != 1 or len(next_n) != 1:
        raise ValueError('В тримере не найдены соседние остатки у N и C центрального остатка.')
    return {'phi': [prev_c[0], N, CA, C], 'psi': [N, CA, C, next_n[0]]}


def map_residue_to_context(residue_chem, context_chem):
    """
    Находит атомы остатка (шаг 4) в тримере (шаг 1) по графу: элементы, связи, заряды.
    Возвращает список: индекс атома тримера для каждого атома остатка. Ошибка, если остаток
    не найден или встречается в разных местах (перестановки симметричных атомов допустимы).
    """
    matches = context_chem.GetSubstructMatches(residue_chem, uniquify=True, useChirality=False,
                                               maxMatches=10)
    if not matches:
        raise ValueError('Остаток не найден в тримере: проверьте, что это тот же остаток '
                         '(шаг 4) и тот же тример (шаг 1), оба с явными водородами.')
    if len({frozenset(m) for m in matches}) > 1:
        raise ValueError('Остаток найден в тримере в нескольких местах: сопоставление неоднозначно.')
    return list(matches[0])


# -----------------------------------------------------------------------------
# Конформации
# -----------------------------------------------------------------------------

def _backbone_pairs(fix_backbone):
    """fix_backbone -> {метка: (phi, psi)} или None (без фиксации)."""
    if fix_backbone is True:
        return dict(AMBER_BACKBONE_CONFORMATIONS)
    if fix_backbone is False or fix_backbone is None:
        return None
    if isinstance(fix_backbone, dict):
        pairs = {str(k): tuple(v) for k, v in fix_backbone.items()}
    else:
        pairs = {f'bb{i + 1}': tuple(v) for i, v in enumerate(fix_backbone)}
    if not pairs:
        raise ValueError('fix_backbone: пустой список пар (phi, psi).')
    for label, pair in pairs.items():
        if len(pair) != 2 or not all(isinstance(x, (int, float)) and -180 <= x <= 180 for x in pair):
            raise ValueError(f'fix_backbone[{label}] = {pair}: нужна пара (phi, psi) в градусах от -180 до 180')
    return {label: (float(p[0]), float(p[1])) for label, p in pairs.items()}


def _sidechain_atoms(mol, residue_atoms, backbone_atoms):
    """Тяжёлые атомы боковой цепи остатка (все тяжёлые атомы остатка, кроме остова)."""
    return [i for i in residue_atoms
            if mol.GetAtomWithIdx(i).GetAtomicNum() > 1 and i not in backbone_atoms]


def _polar_contacts(conf, sidechain_polar, backbone_polar, cutoff):
    if not sidechain_polar or not backbone_polar:
        return False
    pos = conf.GetPositions()
    a = pos[sidechain_polar][:, None, :] - pos[backbone_polar][None, :, :]
    return bool((np.linalg.norm(a, axis=2) < cutoff).any())


def _sidechain_rmsd(mol, cid_a, cid_b, align_atoms, side_atoms):
    # жёсткое совмещение конформации cid_b с cid_a по атомам align_atoms (меняет координаты cid_b)
    rdMolAlign.AlignMol(mol, mol, prbCid=cid_b, refCid=cid_a, atomMap=[(i, i) for i in align_atoms])
    pa = mol.GetConformer(cid_a).GetPositions()[side_atoms]
    pb = mol.GetConformer(cid_b).GetPositions()[side_atoms]
    return float(np.sqrt(((pa - pb) ** 2).sum(axis=1).mean()))


@logged
def resp_conformers(mol, dihedrals, residue_atoms, fix_backbone=True, n_sidechain=3, pool=300,
                    energy_window=10.0, exclude_contacts=True, contact_cutoff=3.2, random_seed=42):
    """
    Конформации для RESP: для каждой пары (phi, psi) из fix_backbone - n_sidechain
    низкоэнергетических (MMFF) и максимально непохожих конформаций боковой цепи.

    Аргументы:
        mol (Chem.Mol) - молекула расчёта (кэпированный остаток или тример) с водородами
        dihedrals (dict) - {'phi': [4 индекса], 'psi': [4 индекса]}
        residue_atoms (list) - индексы атомов остатка в mol
        fix_backbone - True (пары amber: alphaR -60/-40, beta -120/130), False (без фиксации),
            список пар [(phi, psi), ...] или словарь {метка: (phi, psi)}
        n_sidechain (int) - конформаций боковой цепи на каждую пару углов остова
        pool (int) - размер пула ETKDG
        energy_window (float) - окно по энергии MMFF от минимума (ккал/моль) в каждой группе
        exclude_contacts (bool) - не брать конформации, где полярные атомы боковой цепи ближе
            contact_cutoff (Å) к полярным атомам остова/кэпов (свёрнутые в газе), если есть
            достаточно других
        random_seed (int) - воспроизводимость
    Возвращает:
        (Chem.Mol с выбранными конформациями, список описаний конформаций)
    """
    pairs = _backbone_pairs(fix_backbone)
    groups = pairs if pairs is not None else {'free': None}
    base = Chem.Mol(mol)
    base.RemoveAllConformers()
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    params.pruneRmsThresh = 0.1
    params.numThreads = 0   # все ядра
    start = time.time()
    cids = list(AllChem.EmbedMultipleConfs(base, numConfs=pool, params=params))
    if not cids:
        raise ValueError('RDKit не построил ни одной конформации (EmbedMultipleConfs).')
    mp = AllChem.MMFFGetMoleculeProperties(base)
    if mp is None:
        raise ValueError('Для молекулы нет параметров MMFF94: конформации не отбираются.')

    phi, psi = dihedrals['phi'], dihedrals['psi']
    backbone_core = [phi[1], phi[2], phi[3]]     # N, CA, C
    side = _sidechain_atoms(base, residue_atoms, set(phi) | set(psi))
    sym = lambda i: base.GetAtomWithIdx(i).GetSymbol()
    side_polar = [i for i in side if sym(i) in ('N', 'O')
                  and len(Chem.GetShortestPath(base, phi[2], i)) - 1 >= 3]
    nonresidue = [i for i in range(base.GetNumAtoms()) if i not in set(residue_atoms)]
    backbone_polar = [i for i in set(phi + psi) | set(nonresidue) if sym(i) in ('N', 'O')]

    chosen_mol = Chem.Mol(base)
    chosen_mol.RemoveAllConformers()
    described = []
    for label, angles in groups.items():
        work = Chem.Mol(base)
        energies = {}
        for cid in [c.GetId() for c in work.GetConformers()]:
            conf = work.GetConformer(cid)
            if angles is not None:
                rdMolTransforms.SetDihedralDeg(conf, *phi, angles[0])
                rdMolTransforms.SetDihedralDeg(conf, *psi, angles[1])
            ff = AllChem.MMFFGetMoleculeForceField(work, mp, confId=cid)
            if angles is not None:
                ff.MMFFAddTorsionConstraint(*phi, False, angles[0], angles[0], 1e4)
                ff.MMFFAddTorsionConstraint(*psi, False, angles[1], angles[1], 1e4)
            ff.Minimize(maxIts=2000)
            energies[cid] = ff.CalcEnergy()
        order = sorted(energies, key=energies.get)
        e_min = energies[order[0]]
        in_window = [c for c in order if energies[c] - e_min <= energy_window]
        no_contact = [c for c in in_window
                      if not _polar_contacts(work.GetConformer(c), side_polar, backbone_polar, contact_cutoff)]
        candidates = no_contact if (exclude_contacts and len(no_contact) >= n_sidechain) else in_window
        if exclude_contacts and len(no_contact) < n_sidechain:
            print_red(f'⚠ {label}: конформаций без контакта боковой цепи с остовом {len(no_contact)} '
                      f'из нужных {n_sidechain}; берутся и конформации с контактом.')
        if len(candidates) < n_sidechain:
            print_red(f'⚠ {label}: в окне {energy_window} ккал/моль {len(candidates)} конформаций '
                      f'из нужных {n_sidechain}; добавляются следующие по энергии.')
            candidates = order[:max(n_sidechain, len(candidates))]
        align = backbone_core if angles is not None else residue_atoms
        picked = [candidates[0]]
        while len(picked) < min(n_sidechain, len(candidates)):
            best, best_d = None, -1.0
            for c in candidates:
                if c in picked:
                    continue
                d = min(_sidechain_rmsd(work, p, c, align, side) for p in picked)
                if d > best_d:
                    best, best_d = c, d
            picked.append(best)
        for k, cid in enumerate(picked, 1):
            conf = Chem.Conformer(work.GetConformer(cid))
            new_id = chosen_mol.AddConformer(conf, assignId=True)
            got_phi = rdMolTransforms.GetDihedralDeg(conf, *phi)
            got_psi = rdMolTransforms.GetDihedralDeg(conf, *psi)
            described.append({'label': f'{label}_{k}', 'conf_id': new_id,
                              'phi': None if angles is None else angles[0],
                              'psi': None if angles is None else angles[1],
                              'mmff_phi': round(got_phi, 1), 'mmff_psi': round(got_psi, 1),
                              'mmff_dE': round(energies[cid] - e_min, 2),
                              'contact': _polar_contacts(conf, side_polar, backbone_polar, contact_cutoff)})
    print(f'Конформации RESP ({format_time(time.time() - start)}): ' + '; '.join(
        f"{d['label']} (phi {d['mmff_phi']}, psi {d['mmff_psi']}, dE {d['mmff_dE']})" for d in described))
    return chosen_mol, described


# -----------------------------------------------------------------------------
# Задача RESP: подготовка, оптимизация, ESP + фит
# -----------------------------------------------------------------------------

def _xyz_block(symbols, coords, charge, title=''):
    lines = [str(len(symbols)), title]
    lines += [f'{s} {x:.8f} {y:.8f} {z:.8f}' for s, (x, y, z) in zip(symbols, coords)]
    return '\n'.join(lines) + '\n'


def _read_xyz(path):
    with open(path) as f:
        lines = f.read().split('\n')
    n = int(lines[0])
    return np.array([[float(v) for v in line.split()[1:4]] for line in lines[2:2 + n]])


@logged
def prepare_resp_job(residue_chem, constraints, name, working_dir='RESP_data', context='capped',
                     context_chem=None, fix_backbone=True, n_sidechain=3, conformer_pool=300,
                     energy_window=10.0, exclude_contacts=True, random_seed=42, method='hf',
                     basis='6-31g*', g_convergence='gau', opt_chunk=50, max_opt_steps=400,
                     scf_type='df'):
    """
    Готовит папку задачи RESP: молекула расчёта, конформации, ограничения, настройки
    (spec.json). Квантовая химия не запускается. Параметры - см. resp_charges.
    Возвращает путь к папке задачи.
    """
    if context not in RESP_CONTEXTS:
        raise ValueError(f"context='{context}': допустимо {RESP_CONTEXTS}")
    n_res = residue_chem.GetNumAtoms()
    residue_charge = Chem.GetFormalCharge(residue_chem)
    if context == 'capped':
        mol = cap_residue(residue_chem)
        mapping = list(range(n_res))
        dihedrals = json.loads(mol.GetProp('BackboneDihedrals'))
        caps = json.loads(mol.GetProp('CapAtoms'))
        fixed = {mapping[int(i)]: float(q) for i, q in constraints.items()}
        for cap, atoms in caps.items():
            rtp = read_rtp_charges(cap)
            for atom_name, idx in atoms.items():
                fixed[idx] = rtp[atom_name]
    else:
        if context_chem is None:
            raise ValueError("context='trimer': передайте context_chem (тример шага 1).")
        mol = Chem.Mol(context_chem)
        mapping = map_residue_to_context(residue_chem, context_chem)
        dihedrals = context_backbone_dihedrals(context_chem, mapping, residue_chem)
        fixed = {mapping[int(i)]: float(q) for i, q in constraints.items()}

    conf_mol, described = resp_conformers(mol, dihedrals, mapping, fix_backbone=fix_backbone,
                                          n_sidechain=n_sidechain, pool=conformer_pool,
                                          energy_window=energy_window,
                                          exclude_contacts=exclude_contacts, random_seed=random_seed)
    ranks = list(Chem.CanonicalRankAtoms(conf_mol, breakTies=False))
    eq = defaultdict(list)
    for idx, rank in enumerate(ranks):
        if idx not in fixed:
            eq[rank].append(idx)
    groups = [g for g in eq.values() if len(g) > 1]

    folder = os.path.abspath(os.path.join(working_dir, name))
    os.makedirs(folder, exist_ok=True)
    symbols = [a.GetSymbol() for a in conf_mol.GetAtoms()]
    frozen = [] if _backbone_pairs(fix_backbone) is None else [dihedrals['phi'], dihedrals['psi']]
    conformers = []
    for d in described:
        cdir = os.path.join(folder, 'conformers', d['label'])
        os.makedirs(cdir, exist_ok=True)
        coords = conf_mol.GetConformer(d['conf_id']).GetPositions()
        with open(os.path.join(cdir, 'start.xyz'), 'w') as f:
            f.write(_xyz_block(symbols, coords, 0, d['label']))
        conformers.append({k: v for k, v in d.items() if k != 'conf_id'})
    first = Chem.Mol(conf_mol)
    first.RemoveAllConformers()
    first.AddConformer(Chem.Conformer(conf_mol.GetConformer(described[0]['conf_id'])), assignId=True)
    Chem.MolToPDBFile(first, os.path.join(folder, 'molecule.pdb'))
    spec = {
        'name': name, 'context': context, 'molblock': Chem.MolToMolBlock(first),
        'symbols': symbols, 'total_charge': Chem.GetFormalCharge(conf_mol), 'multiplicity': 1,
        'residue_atoms': mapping, 'residue_charge': residue_charge,
        'fixed': {str(k): v for k, v in fixed.items()}, 'equivalent_groups': groups,
        'dihedrals': dihedrals, 'frozen': frozen, 'conformers': conformers,
        'qm': {'method': method, 'basis': basis, 'g_convergence': g_convergence, 'scf_type': scf_type,
               'opt_chunk': opt_chunk, 'max_opt_steps': max_opt_steps},
    }
    with open(os.path.join(folder, SPEC_FILE), 'w') as f:
        json.dump(spec, f, indent=1)
    print(f'Задача RESP: {folder} | молекула {len(symbols)} атомов (заряд {spec["total_charge"]:+d}, '
          f'{context}), остаток {n_res} атомов (заряд {residue_charge:+d}), фиксировано зарядов '
          f'{len(fixed)}, групп эквивалентных атомов {len(groups)}, конформаций {len(conformers)}')
    log_note('RESP: задача', spec={k: v for k, v in spec.items() if k != 'molblock'})
    return folder


def _optimize_conformer(folder, label, n_threads=4, memory='4 GB'):
    """Оптимизирует одну конформацию в psi4 порциями (вызывается в отдельном процессе)."""
    import psi4
    spec = json.load(open(os.path.join(folder, SPEC_FILE)))
    qm = spec['qm']
    cdir = os.path.join(folder, 'conformers', label)
    status_path = os.path.join(cdir, 'status.json')
    status = json.load(open(status_path)) if os.path.exists(status_path) else {'steps': 0}
    if status.get('converged'):
        return status
    current = os.path.join(cdir, 'current.xyz')
    coords = _read_xyz(current if os.path.exists(current) else os.path.join(cdir, 'start.xyz'))
    scratch = os.path.join(cdir, 'scratch')
    os.makedirs(scratch, exist_ok=True)
    psi4.core.IOManager.shared_object().set_default_path(scratch)
    psi4.core.set_output_file(os.path.join(cdir, 'opt.out'), True)
    psi4.set_num_threads(n_threads)
    psi4.set_memory(memory)
    options = {'geom_maxiter': qm['opt_chunk'], 'g_convergence': qm['g_convergence'],
               'scf_type': qm['scf_type']}
    if spec['frozen']:
        options['frozen_dihedral'] = ' '.join(str(i + 1) for quad in spec['frozen'] for i in quad)
    psi4.set_options(options)

    def geometry(xyz):
        body = '\n'.join(f'{s} {x:.8f} {y:.8f} {z:.8f}' for s, (x, y, z) in zip(spec['symbols'], xyz))
        return psi4.geometry(f"{spec['total_charge']} {spec['multiplicity']}\n{body}\n"
                             'units angstrom\nsymmetry c1\nno_com\nno_reorient\n')

    method = f"{qm['method']}/{qm['basis']}"
    while status['steps'] < qm['max_opt_steps']:
        mol = geometry(coords)
        try:
            energy = psi4.optimize(method, molecule=mol)
            coords = mol.geometry().np * psi4.constants.bohr2angstroms
            status.update(converged=True, energy=float(energy))
            status['steps'] += qm['opt_chunk']   # верхняя оценка: последняя порция могла быть короче
            with open(os.path.join(cdir, 'optimized.xyz'), 'w') as f:
                f.write(_xyz_block(spec['symbols'], coords, spec['total_charge'], label))
            break
        except psi4.OptimizationConvergenceError as e:
            last = e.wfn.molecule()
            coords = last.geometry().np * psi4.constants.bohr2angstroms
            status['steps'] += qm['opt_chunk']
            with open(current, 'w') as f:
                f.write(_xyz_block(spec['symbols'], coords, spec['total_charge'], label))
            # очистка перед следующей порцией: без opt_clean optking даёт PSIO_ERROR,
            # а до первой оптимизации opt_clean вызывать нельзя (segfault psi4 1.6)
            psi4.core.clean()
            psi4.core.opt_clean()
        finally:
            with open(status_path, 'w') as f:
                json.dump(status, f)
    else:
        status['converged'] = False
    with open(status_path, 'w') as f:
        json.dump(status, f)
    shutil.rmtree(scratch, ignore_errors=True)
    return status


def _run_parallel(commands, n_parallel, labels, what):
    """
    Запускает команды не более n_parallel одновременно, печатает ход.
    commands - список (argv, рабочая папка, файл для вывода процесса).
    """
    env = dict(os.environ)
    env['PATH'] = os.path.dirname(sys.executable) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = _PACKAGE_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    pending = list(zip(labels, commands))
    running, failed = [], []
    start = time.time()
    while pending or running:
        while pending and len(running) < n_parallel:
            label, cmd = pending.pop(0)
            log = open(cmd[2], 'a')
            running.append((label, subprocess.Popen(cmd[0], cwd=cmd[1], env=env, stdout=log,
                                                    stderr=subprocess.STDOUT), log))
        time.sleep(2)
        for item in list(running):
            label, proc, log = item
            if proc.poll() is not None:
                running.remove(item)
                log.close()
                if proc.returncode != 0:
                    failed.append(label)
                print(f'  {what} {label}: {"ошибка" if proc.returncode else "готово"} '
                      f'({format_time(time.time() - start)} от начала)')
    return failed


@logged
def run_resp_job(folder, n_threads=4, n_parallel=1, memory='4 GB', n_processes=None):
    """
    Выполняет задачу RESP из папки (prepare_resp_job): оптимизация конформаций в psi4
    (n_parallel процессов по n_threads потоков), ESP и двухстадийный RESP в psiresp.
    Повторный запуск продолжает: готовые конформации и расчёты ESP не пересчитываются.
    Возвращает np.ndarray - заряды атомов остатка в порядке остатка шага 4.
    """
    import psiresp
    folder = os.path.abspath(folder)
    spec = json.load(open(os.path.join(folder, SPEC_FILE)))
    labels = [c['label'] for c in spec['conformers']]

    # 1. оптимизация
    todo = []
    for label in labels:
        st = os.path.join(folder, 'conformers', label, 'status.json')
        if not (os.path.exists(st) and json.load(open(st)).get('converged')):
            todo.append(label)
    if todo:
        print(f'Оптимизация HF/6-31G*: {len(todo)} конформаций, по {n_threads} потоков, '
              f'одновременно {n_parallel}')
        cmds = [([sys.executable, '-c', _WORKER, folder, '--optimize', label,
                  '--threads', str(n_threads), '--memory', memory], folder,
                 os.path.join(folder, 'conformers', label, 'worker.log')) for label in todo]
        failed = _run_parallel(cmds, n_parallel, todo, 'конформация')
        if failed:
            raise RuntimeError(f'Оптимизация завершилась ошибкой: {failed} (см. conformers/<метка>/opt.out)')
    statuses = {label: json.load(open(os.path.join(folder, 'conformers', label, 'status.json')))
                for label in labels}
    bad = [label for label, s in statuses.items() if not s.get('converged')]
    if bad:
        raise RuntimeError(f'Не сошлись за {spec["qm"]["max_opt_steps"]} шагов: {bad}. '
                           'Повторный запуск продолжит с последней геометрии (увеличьте max_opt_steps).')

    # 2. ESP и двухстадийный RESP (psiresp считает ESP и решает каждую стадию)
    q_all = _fit_two_stage(spec, folder, labels, n_threads, n_parallel, memory, n_processes)
    charges = q_all[spec['residue_atoms']]
    result = {'charges': charges.tolist(), 'charges_all': q_all.tolist(),
              'conformers': {label: statuses[label] for label in labels}}
    with open(os.path.join(folder, RESULT_FILE), 'w') as f:
        json.dump(result, f, indent=1)
    return _check_resp_charges(charges, spec, folder)


def _independent_constraints(n_atoms, fixed, sums, groups):
    """
    Убирает линейно зависимые ограничения (с ними матрица psiresp вырождена и фит
    даёт произвольный результат). Полный заряд молекулы psiresp задаёт сам - он учтён
    первым. Возвращает (fixed, sums, пары эквивалентности, число отброшенных).
    """
    rows = [np.ones(n_atoms)]
    rank = 1

    def try_add(row):
        nonlocal rank
        trial = np.linalg.matrix_rank(np.vstack(rows + [row]))
        if trial > rank:
            rows.append(row)
            rank = trial
            return True
        return False

    kept_fixed, kept_sums, kept_pairs, dropped = {}, [], [], 0
    for idx, q in fixed.items():
        row = np.zeros(n_atoms)
        row[idx] = 1
        if try_add(row):
            kept_fixed[idx] = q
        else:
            dropped += 1
    for indices, q in sums:
        row = np.zeros(n_atoms)
        row[list(indices)] = 1
        if try_add(row):
            kept_sums.append((indices, q))
        else:
            dropped += 1
    for group in groups:
        for a, b in zip(group, group[1:]):
            row = np.zeros(n_atoms)
            row[a], row[b] = 1, -1
            if try_add(row):
                kept_pairs.append([a, b])
            else:
                dropped += 1
    return kept_fixed, kept_sums, kept_pairs, dropped


def _stage2_atoms(rdmol, fixed):
    """sp3-углероды CH2/CH3 и их водороды (без фиксированных): {C: [H, ...]}."""
    groups = {}
    for atom in rdmol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetIdx() in fixed or atom.GetDegree() != 4:
            continue
        hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1 and n.GetIdx() not in fixed]
        if len(hs) >= 2 and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()):
            groups[atom.GetIdx()] = hs
    return groups


def _fit_two_stage(spec, folder, labels, n_threads, n_parallel, memory, n_processes):
    """
    ESP (psiresp, сетка MSK) и двухстадийный RESP (Bayly 1993, Cornell 1995):
    стадия 1 (ограничение 0.0005) - свободны все атомы, кроме фиксированных; эквивалентны
    симметричные атомы, кроме водородов CH2/CH3; стадия 2 (0.001) - заново подгоняются
    только CH2/CH3 (углерод и водороды, водороды при одном углероде эквивалентны),
    остальные атомы фиксируются на значениях стадии 1. Вторую стадию psiresp 0.4.2
    готовит с дублирующимися ограничениями (матрица вырождена), поэтому она собирается здесь.
    Все конформации входят в фит с равным весом.
    """
    import psiresp
    qm = spec['qm']
    rdmol = Chem.MolFromMolBlock(spec['molblock'], removeHs=False)
    n_atoms = rdmol.GetNumAtoms()
    fixed = {int(k): float(v) for k, v in spec['fixed'].items()}
    sums = [] if spec['context'] == 'capped' else [(spec['residue_atoms'], spec['residue_charge'])]
    stage2 = _stage2_atoms(rdmol, fixed)
    stage2_h = {h for hs in stage2.values() for h in hs}
    stage2_all = stage2_h | set(stage2)

    mol = psiresp.Molecule.from_rdkit(rdmol, optimize_geometry=False, charge=spec['total_charge'],
                                      conformer_generation_options=dict(n_max_conformers=0,
                                                                        keep_original_conformer=False))
    # psi4 не должен сдвигать и поворачивать молекулу: сетку ESP psiresp строит в системе
    # координат конформации, а потенциал считает по волновой функции psi4. Конформации RDKit
    # уже центрированы и повёрнуты по главным осям, а геометрии после оптимизации - нет.
    mol.qcmol = mol.qcmol.copy(update={'fix_com': True, 'fix_orientation': True})
    mol.conformers = []
    for label in labels:
        mol.add_conformer_with_coordinates(_read_xyz(os.path.join(folder, 'conformers', label, 'optimized.xyz')))
    esp_options = psiresp.QMEnergyOptions(method=qm['method'], basis=qm['basis'],
                                          keywords={'scf_type': qm['scf_type']})

    def solve(molecules, fixed_q, sum_q, groups, height):
        f, s_, pairs, dropped = _independent_constraints(n_atoms, fixed_q, sum_q, groups)
        cc = psiresp.ChargeConstraintOptions(symmetric_atoms_are_equivalent=False,
                                             symmetric_methyls=False, symmetric_methylenes=False)
        for idx, q in f.items():
            cc.add_charge_sum_constraint_for_molecule(molecules[0], charge=q, indices=[idx])
        for indices, q in s_:
            cc.add_charge_sum_constraint_for_molecule(molecules[0], charge=q, indices=list(indices))
        for a, b in pairs:
            cc.add_charge_equivalence_constraint_for_molecule(molecules[0], indices=[a, b])
        job = psiresp.Job(molecules=molecules, working_directory=os.path.join(folder, 'psiresp'),
                          charge_constraints=cc, n_processes=n_processes, qm_esp_options=esp_options,
                          resp_options=psiresp.RespOptions(stage_2=False, restraint_height_stage_1=height))
        _run_esp(job, folder, n_threads, n_parallel, memory)
        return job, np.array(job.charges[0], dtype=float), dropped

    groups1 = [g for g in spec['equivalent_groups'] if not set(g) & stage2_h]
    job1, q1, dropped1 = solve([mol], fixed, sums, groups1, 0.0005)
    q = q1
    dropped2 = 0
    if stage2_all:
        fixed2 = dict(fixed)
        fixed2.update({i: float(q1[i]) for i in range(n_atoms) if i not in stage2_all and i not in fixed})
        groups2 = [hs for hs in stage2.values() if len(hs) > 1]
        groups2 += [g for g in spec['equivalent_groups'] if set(g) <= set(stage2)]
        _, q, dropped2 = solve(job1.molecules, fixed2, sums, groups2, 0.001)
    if dropped1 or dropped2:
        print(f'Ограничения RESP: линейно зависимых отброшено - стадия 1: {dropped1}, стадия 2: {dropped2} '
              '(они следуют из остальных).')
    return q


def _run_esp(job, folder, n_threads, n_parallel, memory):
    """job.run() с выполнением расчётов ESP psi4 (параллельно); готовые не пересчитываются."""
    for _ in range(3):
        try:
            job.run()
            return
        except SystemExit:
            script = os.path.join(folder, 'psiresp', 'single_point', 'run_single_point.sh')
            files = [line.split()[-1] for line in open(script) if line.startswith('psi4 ')]
            print(f'ESP HF/6-31G*: {len(files)} расчётов psi4')
            cmds = [(['psi4', '-n', str(n_threads), '--memory', memory.replace(' ', ''),
                      '--qcschema', f], os.path.dirname(script),
                     os.path.join(os.path.dirname(script), f + '.log')) for f in files]
            failed = _run_parallel(cmds, n_parallel, files, 'ESP')
            os.rename(script, script + '.done')
            if failed:
                raise RuntimeError(f'Расчёт ESP завершился ошибкой: {failed}')
    raise RuntimeError('psiresp не завершил расчёт ESP за 3 раунда.')


def _check_resp_charges(charges, spec, folder):
    fixed_res = {spec['residue_atoms'].index(int(i)): q for i, q in spec['fixed'].items()
                 if int(i) in spec['residue_atoms']}
    total_ok = abs(charges.sum() - spec['residue_charge']) < 1e-3
    fixed_ok = all(abs(charges[i] - q) < 1e-4 for i, q in fixed_res.items())
    print(f'RESP {spec["name"]}: конформаций {len(spec["conformers"])}, сумма зарядов остатка '
          f'{charges.sum():.5f} (заряд {spec["residue_charge"]:+d}), ограничения соблюдены: {fixed_ok}')
    log_note('RESP: заряды', folder=folder, charges=charges.tolist())
    if not (total_ok and fixed_ok):
        raise ValueError('Заряды RESP не прошли проверку: см. сумму и ограничения выше.')
    return charges


@logged
def load_resp_result(folder):
    """Читает заряды готовой задачи RESP (например, посчитанной на кластере) и проверяет их."""
    folder = os.path.abspath(folder)
    path = os.path.join(folder, RESULT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f'Нет {path}: задача ещё не посчитана (run_resp_job или SLURM).')
    spec = json.load(open(os.path.join(folder, SPEC_FILE)))
    charges = np.array(json.load(open(path))['charges'])
    return _check_resp_charges(charges, spec, folder)


def write_slurm_script(folder, cpus=24, mem='32G', time_limit='08:00:00', n_parallel=None,
                       partition=None, env_setup=CLUSTER_ENV_SETUP):
    """
    Пишет <папка задачи>/run_resp.sbatch. Ресурсы делятся между конформациями:
    n_parallel одновременно (по умолчанию - число конформаций, но не больше cpus),
    потоков на каждую cpus // n_parallel, памяти - поровну. Отправка:
    cd <папка> && sbatch run_resp.sbatch. Если задачу оборвёт лимит времени, повторная
    отправка продолжит с сохранённых геометрий.
    """
    folder = os.path.abspath(folder)
    spec = json.load(open(os.path.join(folder, SPEC_FILE)))
    n_conf = len(spec['conformers'])
    n_parallel = n_parallel or min(n_conf, cpus)
    threads = max(1, cpus // n_parallel)
    mem_gb = int(''.join(ch for ch in mem if ch.isdigit()))
    per_job = f'{max(1, int(mem_gb * 0.9 / n_parallel))}GB'
    root = os.path.relpath(_PACKAGE_ROOT, folder)
    lines = ['#!/bin/bash',
             f'#SBATCH --job-name=resp_{spec["name"]}',
             '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
             f'#SBATCH --cpus-per-task={cpus}', f'#SBATCH --mem={mem}',
             f'#SBATCH --time={time_limit}',
             '#SBATCH --output=slurm_%j.out']
    if partition:
        lines.append(f'#SBATCH --partition={partition}')
    lines += ['', 'cd "$SLURM_SUBMIT_DIR"', env_setup,
              f'export PYTHONPATH="$SLURM_SUBMIT_DIR/{root}:$PYTHONPATH"',
              f'export OMP_NUM_THREADS={threads}',
              f'python -c "{_WORKER}" . --threads {threads} --parallel {n_parallel} '
              f'--memory {per_job}', '']
    path = os.path.join(folder, 'run_resp.sbatch')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'SLURM: {path} ({n_parallel} конформаций одновременно по {threads} потоков, '
          f'{per_job} памяти на каждую). Отправка: cd {folder} && sbatch run_resp.sbatch')
    return path


_BENCH_SCRIPT = """import json, sys, time, glob, os
import psi4
spec = json.load(open('cluster_test/spec.json'))
start = sorted(glob.glob('cluster_test/conformers/*/start.xyz'))[0]
lines = open(start).read().split('\\n')[2:2 + len(spec['symbols'])]
geom = '\\n'.join(lines)
out = open('bench.txt', 'w')
for n in [int(x) for x in sys.argv[1:]]:
    psi4.core.clean()
    psi4.set_num_threads(n)
    psi4.set_memory('4 GB')
    psi4.core.set_output_file(f'bench_{n}.out', False)
    mol = psi4.geometry(f"{spec['total_charge']} 1\\n{geom}\\nsymmetry c1\\nno_com\\nno_reorient")
    t = time.time()
    psi4.gradient('hf/6-31g*', molecule=mol)
    msg = f'градиент HF/6-31G*, {len(spec["symbols"])} атомов, {n} потоков: {time.time() - t:.1f} с'
    print(msg, flush=True)
    out.write(msg + '\\n'); out.flush()
"""


@logged
def prepare_cluster_check(residue_chem, constraints, folder='RESP_data/cluster_check', cpus=24, mem='32G',
                          time_limit='01:00:00', partition=None, env_setup=CLUSTER_ENV_SETUP):
    """
    Готовит проверку кластера: папка с run_check.sbatch, который на узле
    1) замеряет градиент psi4 HF/6-31G* для кэпированного остатка на 1, cpus/4, cpus/2 и cpus
       потоках (bench.txt) - видно, ускоряют ли ядра узла расчёт;
    2) выполняет маленькую задачу RESP целиком (HF/STO-3G, одна конформация) - проверка,
       что окружение, psi4, psiresp и пакет работают на кластере.
    Отправка: cd <папка> && sbatch run_check.sbatch. Результат: bench.txt, slurm_*.out,
    cluster_test/result.json.
    """
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    prepare_resp_job(residue_chem, constraints, 'cluster_test', working_dir=folder,
                     fix_backbone=[(-60.0, -40.0)], n_sidechain=1, conformer_pool=50, basis='sto-3g')
    with open(os.path.join(folder, 'bench.py'), 'w') as f:
        f.write(_BENCH_SCRIPT)
    threads = sorted({1, max(1, cpus // 4), max(1, cpus // 2), cpus})
    root = os.path.relpath(_PACKAGE_ROOT, folder)
    lines = ['#!/bin/bash', '#SBATCH --job-name=resp_check', '#SBATCH --nodes=1', '#SBATCH --ntasks=1',
             f'#SBATCH --cpus-per-task={cpus}', f'#SBATCH --mem={mem}', f'#SBATCH --time={time_limit}',
             '#SBATCH --output=slurm_%j.out']
    if partition:
        lines.append(f'#SBATCH --partition={partition}')
    lines += ['', 'cd "$SLURM_SUBMIT_DIR"', env_setup,
              f'export PYTHONPATH="$SLURM_SUBMIT_DIR/{root}:$PYTHONPATH"',
              'echo "узел: $(hostname), ядер: $(nproc), python: $(which python)"',
              f'python bench.py {" ".join(map(str, threads))}',
              f'python -c "{_WORKER}" cluster_test --threads {cpus} --parallel 1 --memory 8GB', '']
    path = os.path.join(folder, 'run_check.sbatch')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Проверка кластера: {path}. Отправка: cd {folder} && sbatch run_check.sbatch '
          f'(замер {threads} потоков и пробная задача RESP).')
    return path


def _resp_python(resp_python=None):
    """
    Python окружения с psi4 и psiresp. Текущий, если они в нём есть; иначе окружение
    darwin_resp рядом с текущим (anaconda3/envs/darwin_resp) - так ноутбук может работать
    в darwin_ec (Espaloma), а RESP считаться в darwin_resp.
    """
    if resp_python:
        return resp_python
    try:
        import psiresp  # noqa: F401
        import psi4  # noqa: F401
        return sys.executable
    except ImportError:
        pass
    envs = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
    candidate = os.path.join(envs, 'darwin_resp', 'bin', 'python')
    if os.path.exists(candidate):
        return candidate
    raise ImportError('Нет psi4/psiresp: создайте окружение darwin_resp '
                      '(conda env create -n darwin_resp -f psiresp_min.yml) или передайте resp_python.')


def _run_resp_job_external(folder, python, n_threads, n_parallel, memory):
    """run_resp_job в другом окружении (python): вывод процесса печатается по мере выполнения."""
    env = dict(os.environ)
    env['PATH'] = os.path.dirname(python) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = _PACKAGE_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    env['PYTHONUNBUFFERED'] = '1'
    print(f'RESP считается в окружении {os.path.basename(os.path.dirname(os.path.dirname(python)))} '
          f'({python})')
    cmd = [python, '-c', _WORKER, folder, '--threads', str(n_threads), '--parallel', str(n_parallel),
           '--memory', memory.replace(' ', '')]
    noise = ('Warning', 'warn(', 'it/s]', 'forrtl', 'Image ', 'Unknown', 'dgstrf')
    with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
        for line in proc.stdout:
            if not any(n in line for n in noise) and line.strip():
                print(line.rstrip())
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f'Расчёт RESP завершился ошибкой (код {proc.returncode}): см. вывод выше '
                           f'и файлы в {folder}. Повторный запуск продолжит с готовых конформаций.')
    return load_resp_result(folder)


@logged
def resp_charges(residue_chem, constraints, name, run='local', working_dir='RESP_data',
                 context='capped', context_chem=None, fix_backbone=True, n_sidechain=3,
                 conformer_pool=300, energy_window=10.0, exclude_contacts=True, random_seed=42,
                 n_threads=4, n_parallel=1, memory='4 GB', slurm=None, resp_python=None, **qm_options):
    """
    Заряды RESP для остатка шага 4 (шаг 6) - с теми же входными данными, что Espaloma.

    Аргументы:
        residue_chem (Chem.Mol) - остаток шага 4
        constraints (dict) - {индекс атома остатка: заряд} (charge_constraints_from_rtp)
        name (str) - имя задачи: папка working_dir/name (повторный запуск продолжает её)
        run (str) - 'local' (считать здесь), 'slurm' (подготовить задачу и run_resp.sbatch),
            'prepare' (только подготовить и проверить, без квантовой химии)
        context (str) - 'capped' (ACE-X-NME, по умолчанию) или 'trimer' (нужен context_chem)
        fix_backbone - True: (phi, psi) amber - alphaR (-60, -40) и beta (-120, 130);
            False: без фиксации; список пар [(phi, psi), ...] или словарь {метка: (phi, psi)}
        n_sidechain (int) - конформаций боковой цепи на каждую пару углов остова
        conformer_pool, energy_window, exclude_contacts, random_seed - отбор конформаций
            (см. resp_conformers)
        n_threads, n_parallel, memory - ресурсы для run='local': потоков на один psi4,
            одновременных расчётов, память на один psi4
        resp_python (str) - Python окружения с psi4/psiresp для run='local'; по умолчанию
            текущее, если в нём есть psiresp, иначе darwin_resp рядом с текущим окружением
        slurm (dict) - для run='slurm': cpus, mem, time_limit, partition, env_setup
            (по умолчанию 24 ядра, 32G, 8 часов, окружение darwin_resp - CLUSTER_ENV_SETUP)
        **qm_options - method='hf', basis='6-31g*', g_convergence='gau', opt_chunk=50,
            max_opt_steps=400, scf_type='df' (opt_chunk - шагов оптимизации между сохранениями геометрии)
    Возвращает:
        np.ndarray - заряды атомов остатка (run='local'), иначе путь к папке задачи
    """
    if run not in RESP_RUN_MODES:
        raise ValueError(f"run='{run}': допустимо {RESP_RUN_MODES}")
    folder = prepare_resp_job(residue_chem, constraints, name, working_dir=working_dir,
                              context=context, context_chem=context_chem, fix_backbone=fix_backbone,
                              n_sidechain=n_sidechain, conformer_pool=conformer_pool,
                              energy_window=energy_window, exclude_contacts=exclude_contacts,
                              random_seed=random_seed, **qm_options)
    if run == 'prepare':
        print("run='prepare': квантовая химия не запускалась.")
        return folder
    return run_resp(folder, run=run, n_threads=n_threads, n_parallel=n_parallel, memory=memory,
                    slurm=slurm, resp_python=resp_python)


@logged
def run_resp(folder, run='local', n_threads=4, n_parallel=1, memory='4 GB', slurm=None, resp_python=None):
    """
    Запуск подготовленной задачи RESP (prepare_resp_job).
        run='local' - считать на этой машине: в текущем окружении, если в нём есть psiresp,
            иначе в darwin_resp (отдельный процесс, ход расчёта печатается); возвращает заряды
        run='slurm' - написать run_resp.sbatch (параметры slurm: cpus, mem, time_limit,
            partition, env_setup); возвращает путь к папке, заряды потом - load_resp_result
    Повторный запуск продолжает с готовых конформаций и расчётов ESP.
    """
    if run == 'slurm':
        write_slurm_script(folder, **(slurm or {}))
        return folder
    if run != 'local':
        raise ValueError(f"run='{run}': допустимо 'local' или 'slurm'")
    python = _resp_python(resp_python)
    if python == sys.executable:
        return run_resp_job(folder, n_threads=n_threads, n_parallel=n_parallel, memory=memory)
    return _run_resp_job_external(os.path.abspath(folder), python, n_threads, n_parallel, memory)


def _main(argv=None):
    parser = argparse.ArgumentParser(description='Выполнить задачу RESP (папка с spec.json).')
    parser.add_argument('folder')
    parser.add_argument('--optimize', metavar='LABEL', help='оптимизировать одну конформацию')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--memory', default='4GB')
    args = parser.parse_args(argv)
    if args.optimize:
        status = _optimize_conformer(args.folder, args.optimize, n_threads=args.threads,
                                     memory=args.memory)
        sys.exit(0 if status.get('converged') is not None else 1)
    run_resp_job(args.folder, n_threads=args.threads, n_parallel=args.parallel, memory=args.memory)


# Импорт из других модулей пакета - в конце файла (см. CLAUDE.md, правило импортов).
from .charges import read_rtp_charges  # noqa: E402
from .residue import find_backbone_match  # noqa: E402

if __name__ == '__main__':
    _main()
