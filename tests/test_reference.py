"""
Тесты на эталонном наборе: Lysine_prop, Lysine_Malonyl, Lysine_Formyl, Lysine_3M, AF_546_cys.

Каждый тест повторяет шаги нового ноутбука 1_charge_calculation.ipynb (по параметрам из
make_charge_notebooks.PARAMS) и сравнивает результат с файлами, лежащими в git (они получены
этими же шагами и проверены). Тест падает, если изменение кода меняет имена атомов, порядок,
протонирование, ограничения зарядов, кэпированный мономер или подготовку задачи RESP.
Если поведение меняется намеренно - пересчитайте файлы ноутбуком и обновите их в git.

Квантовая химия (psi4) не запускается: RESP проверяется до этапа расчёта (prepare).
"""
import contextlib
import functools
import io
import json
import os

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from conftest import ROOT

RDLogger.DisableLog('rdApp.*')

import ResParTools as pt                      # noqa: E402
from make_charge_notebooks import PARAMS      # noqa: E402

REFERENCE = ['Lysine_prop', 'Lysine_Malonyl', 'Lysine_Formyl', 'Lysine_3M', 'AF_546_cys']
CASES = [pytest.param(f, marks=pytest.mark.slow) if f.startswith('AF_') else f for f in REFERENCE]


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _mon_name(folder):
    return os.path.splitext(os.path.basename(PARAMS[folder]['monomer_file']))[0]


def _pdb_names(path):
    """Имена атомов, имя и номер остатка из PDB."""
    mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
    infos = [a.GetPDBResidueInfo() for a in mol.GetAtoms()]
    return ([i.GetName().strip() for i in infos], {i.GetResidueName().strip() for i in infos},
            {i.GetResidueNumber() for i in infos})


@functools.lru_cache(maxsize=None)
def steps_1_to_4(folder, tmp_dir):
    """Шаги 1-4 ноутбука: мономер в тримере, сохранение, перенумерация, имена атомов."""
    p = PARAMS[folder]
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, folder))
    try:
        monomer = _quiet(pt.file_opener, p['monomer_file'])
        trimers = _quiet(pt.file_opener, [p['trimer_file']])
        mon = next(iter(monomer))
        match = _quiet(pt.match_mon_to_pol, monomer, trimers, only_heavy_mapping=True, timeout=5)
        sub = match['substructure'][mon]
        _quiet(pt.save_chem_to_pdb, sub, os.path.join(tmp_dir, f'3_{mon}'))
        pdb = _quiet(pt.file_opener, os.path.join(tmp_dir, f'3_{mon}.pdb'))[f'3_{mon}']
        renum = _quiet(pt.renumber_residue_atoms, pdb, ref_base_name=p['parent_residue'], timeout=5)
        parent = json.loads(renum.GetProp('ParentAtoms'))
        _quiet(pt.save_aa_chem_to_pdb, renum, os.path.join(tmp_dir, f'3_2_{mon}'), resname=p['base_name'],
               resid=2, segid='A', naming=p['naming'], parent_atoms=parent)
        residue = _quiet(pt.file_opener, os.path.join(tmp_dir, f'3_2_{mon}.pdb'))[f'3_2_{mon}']
        return mon, residue, tuple(parent), monomer[mon], sub.GetNumAtoms()
    finally:
        os.chdir(cwd)


@pytest.fixture(scope='session')
def tmp_dir(tmp_path_factory):
    return str(tmp_path_factory.mktemp('reference'))


@pytest.mark.parametrize('folder', CASES)
def test_atom_names_match_committed(folder, tmp_dir):
    """Имена атомов, их порядок, имя и номер остатка - как в 4_<имя>.pdb в git."""
    mon, residue, _, _, n_sub = steps_1_to_4(folder, tmp_dir)
    names = [a.GetProp('AtomName') for a in residue.GetAtoms()]
    ref_names, ref_resnames, ref_resids = _pdb_names(
        os.path.join(ROOT, folder, 'molecules', 'substructure', f'4_{mon}.pdb'))
    assert residue.GetNumAtoms() == n_sub
    assert names == ref_names
    assert ref_resnames == {PARAMS[folder]['base_name']} and ref_resids == {2}
    assert len(set(names)) == len(names), 'повторяющиеся имена атомов'
    assert names[:4] == ['N', 'H', 'CA', 'HA'] and names[-2:] == ['C', 'O'], 'остов не на своих местах'


@pytest.mark.parametrize('folder', CASES)
def test_general_atoms_match_committed(folder, tmp_dir):
    """general_atoms.json (атомы родительского остатка для 3_edd_topology) - как в git."""
    _, residue, parent, _, _ = steps_1_to_4(folder, tmp_dir)
    names = [residue.GetAtomWithIdx(i).GetProp('AtomName') for i in parent]
    with open(os.path.join(ROOT, folder, 'general_atoms.json')) as f:
        assert names == json.load(f)


@pytest.mark.parametrize('folder', CASES)
def test_protonation_hw(folder, tmp_dir):
    """Шаг 4.1: HW1 на N, HW2 на C, атомы остатка не меняются, заряд тот же."""
    _, residue, _, _, _ = steps_1_to_4(folder, tmp_dir)
    h_res = _quiet(pt.add_protons_and_renumber_H, residue, resname=PARAMS[folder]['base_name'], resid=2, segid='A')
    n = residue.GetNumAtoms()
    assert h_res.GetNumAtoms() == n + 2
    assert [h_res.GetAtomWithIdx(i).GetProp('AtomName') for i in range(n)] == \
           [a.GetProp('AtomName') for a in residue.GetAtoms()]
    added = {h_res.GetAtomWithIdx(i).GetProp('AtomName'):
             h_res.GetAtomWithIdx(i).GetNeighbors()[0].GetProp('AtomName') for i in range(n, n + 2)}
    assert added == {'HW1': 'N', 'HW2': 'C'}
    assert Chem.GetFormalCharge(h_res) == Chem.GetFormalCharge(residue)


@pytest.mark.parametrize('folder', CASES)
def test_charge_constraints_from_rtp(folder, tmp_dir):
    """Ограничения шагов 5/6: остов N H CA HA C O = заряды родительского остатка amber14sb."""
    _, residue, _, _, _ = steps_1_to_4(folder, tmp_dir)
    rtp_res = pt.AMINO_ACIDS[PARAMS[folder]['parent_residue']][0].upper()
    constraints = _quiet(pt.charge_constraints_from_rtp, residue, rtp_res)
    rtp = pt.read_rtp_charges(rtp_res)
    got = {residue.GetAtomWithIdx(i).GetProp('AtomName'): q for i, q in constraints.items()}
    assert got == {n: rtp[n] for n in ('N', 'H', 'CA', 'HA', 'C', 'O')}


def _ca_is_L(mol):
    probe = Chem.Mol(mol)
    assert AllChem.EmbedMolecule(probe, randomSeed=7) == 0
    ca, bb = pt.find_backbone_match(probe)
    n, c = bb[0], bb[2]
    cb = [a.GetIdx() for a in probe.GetAtomWithIdx(ca).GetNeighbors() if a.GetAtomicNum() > 1 and a.GetIdx() not in (n, c)][0]
    x = probe.GetConformer().GetPositions()
    return float(np.dot(x[n] - x[ca], np.cross(x[c] - x[ca], x[cb] - x[ca]))) > 0


@pytest.mark.parametrize('folder', CASES)
def test_capped_monomer(folder, tmp_dir):
    """ACE-X-NME: как <имя>_capped.smiles в git, CA - L, граф и заряд как у остатка шага 4."""
    mon, residue, _, monomer, _ = steps_1_to_4(folder, tmp_dir)
    capped = _quiet(pt.resp.capped_monomer, monomer)
    with open(os.path.join(ROOT, folder, 'molecules', f'{mon}_capped.smiles')) as f:
        committed = Chem.MolFromSmiles(f.read().strip())
    assert Chem.MolToSmiles(Chem.RemoveHs(capped)) == Chem.MolToSmiles(committed)
    assert _ca_is_L(capped)
    mol = pt.resp._with_reference_stereo(pt.resp.cap_residue(residue), capped)
    assert mol.GetNumAtoms() == residue.GetNumAtoms() + 12
    assert Chem.GetFormalCharge(mol) == Chem.GetFormalCharge(residue)


@pytest.mark.parametrize('folder', CASES)
def test_resp_prepare(folder, tmp_dir):
    """Подготовка RESP (без QM): кэпы и остов фиксированы, суммы, эквивалентность, стереохимия."""
    mon, residue, _, monomer, _ = steps_1_to_4(folder, tmp_dir)
    rtp_res = pt.AMINO_ACIDS[PARAMS[folder]['parent_residue']][0].upper()
    constraints = _quiet(pt.charge_constraints_from_rtp, residue, rtp_res)
    job = _quiet(pt.prepare_resp_job, residue, constraints, f'{mon}_test', working_dir=tmp_dir,
                 fix_backbone=[(-60.0, -40.0)], n_sidechain=1, conformer_pool=40, monomer=monomer)
    with open(os.path.join(job, pt.SPEC_FILE)) as f:
        spec = json.load(f)
    names, residues = spec['atom_names'], spec['atom_residues']
    fixed = {int(k): v for k, v in spec['fixed'].items()}
    by_res = {r: sorted(names[i] for i in fixed if residues[i] == r) for r in ('ACE', 'NME', 'остаток')}
    assert by_res['ACE'] == sorted(['C', 'O', 'CH3', 'HH31', 'HH32', 'HH33'])
    assert by_res['NME'] == sorted(['N', 'H', 'CH3', 'HH31', 'HH32', 'HH33'])
    assert by_res['остаток'] == sorted(['N', 'H', 'CA', 'HA', 'C', 'O'])
    for cap in ('ACE', 'NME'):
        assert abs(sum(fixed[i] for i in fixed if residues[i] == cap)) < 1e-4
    assert spec['total_charge'] == spec['residue_charge'] == Chem.GetFormalCharge(residue)
    assert spec['stereo'].get('CA') == ('R' if PARAMS[folder]['parent_residue'] == 'C' else 'S')
    assert len(spec['frozen']) == 2
    assert not any(c['contact'] for c in spec['conformers']), 'конформация с контактом боковой цепи'
    groups = [{names[i] for i in g} for g in spec['equivalent_groups']]
    if folder == 'Lysine_Malonyl':
        assert {'OK1', 'OK2'} in groups, 'кислороды карбоксилата должны быть эквивалентны'


def test_independent_constraints_drop_redundant():
    """Линейно зависимые ограничения отбрасываются (иначе матрица psiresp вырождена)."""
    # 3 атома, полный заряд задаёт psiresp; фиксация всех трёх - одно ограничение лишнее
    fixed, sums, pairs, dropped = pt.resp._independent_constraints(3, {0: 0.1, 1: -0.2, 2: 0.1}, [], [[1, 2]])
    assert len(fixed) == 2 and dropped == 2
