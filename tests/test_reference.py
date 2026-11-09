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
import glob
import io
import json
import os
import shutil

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from conftest import ROOT

RDLogger.DisableLog('rdApp.*')

import ResParTools as pt                      # noqa: E402
import make_charge_notebooks as notebook_builder  # noqa: E402
from make_charge_notebooks import PARAMS      # noqa: E402

REFERENCE = ['Lysine_prop', 'Lysine_Malonyl', 'Lysine_Formyl', 'Lysine_3M', 'AF_546_cys']
CASES = [pytest.param(f, marks=pytest.mark.slow) if f.startswith('AF_') else f for f in REFERENCE]


def test_capped_reference_library_and_labels():
    """Кэпированные PDB читаются проектным парсером и дают правильный код родителя."""
    library = os.path.join(ROOT, 'molecules', 'aminoacids_capped')
    rtp = pt.read_rtp(os.path.join(ROOT, 'amber14sb_parmbsc1_cufix.ff', 'aminoacids.rtp'))[1]
    paths = sorted(glob.glob(os.path.join(library, '*.pdb')))
    assert len(paths) == 29
    for path in paths:
        mol = _quiet(pt.pdb_to_chem, path, format_coord='3D')
        if isinstance(mol, dict):
            mol = mol[os.path.splitext(os.path.basename(path))[0]]
        names = [a.GetPDBResidueInfo() for a in mol.GetAtoms()]
        assert {(a.GetResidueName().strip(), a.GetResidueNumber()) for a in names} == {
            ('ACE', 1), (os.path.basename(path)[:3], 2), ('NME', 3)}
        center = _quiet(pt.extract_residue, mol, 2)
        assert [a.GetProp('AtomName') for a in center.GetAtoms()] == [
            a.GetProp('AtomName') for a in mol.GetAtoms()
            if a.GetPDBResidueInfo().GetResidueNumber() == 2]
        assert [a.GetProp('AtomName') for a in center.GetAtoms()] == [
            atom[0] for atom in rtp[os.path.basename(path)[:3]]['atoms']]
        index = {(a.GetPDBResidueInfo().GetResidueNumber(), a.GetProp('AtomName')): a.GetIdx()
                 for a in mol.GetAtoms()}
        assert mol.GetBondBetweenAtoms(index[(1, 'C')], index[(2, 'N')]) is not None
        assert mol.GetBondBetweenAtoms(index[(2, 'C')], index[(3, 'N')]) is not None
    assert pt.template_aa_letter('GKG_H') == 'K'
    assert pt.template_aa_letter('LYS') == 'K'
    assert pt.template_aa_letter('LYN') == 'K'
    assert pt.template_aa_letter('TRP') == 'W'


@pytest.mark.parametrize('folder', [name for name in PARAMS if name.startswith('Lysine_')])
@pytest.mark.slow
def test_automatic_capped_parent_for_lysine_modifications(folder, tmp_dir):
    """Автовыбор по мономеру сообщает кандидата и сохраняет код родителя для всех 12 входов."""
    candidates = os.path.join(tmp_dir, 'lysine_parent_candidates')
    os.makedirs(candidates, exist_ok=True)
    for name in ('LYS.pdb', 'LYN.pdb'):
        src = os.path.join(ROOT, 'molecules', 'aminoacids_capped', name)
        dst = os.path.join(candidates, name)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)

    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, folder))
    try:
        monomer = _quiet(pt.open_mol, PARAMS[folder]['monomer_file'])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            residue = pt.renumber_residue_atoms(
                monomer, ref_base_name=None, residue_type='protein_capped',
                path_to_ref_mol=candidates, timeout=3, strict_parent=False)
        assert residue.GetProp('Parent') == 'K'
        assert residue.GetProp('RefResidue') in {'LYS', 'LYN'}
        assert 'Родительский остаток не указан' in output.getvalue()
        assert sum(a.HasProp('ParentName') for a in residue.GetAtoms()) > 10
    finally:
        os.chdir(cwd)


@pytest.mark.parametrize('folder', [name for name in PARAMS if name.startswith('Lysine_')] + ['AF_546_cys'])
def test_open_peptide_boundaries_preserves_modification(folder):
    """Границы происходят из capped PDB; состав сверяется со старым тримерным эталоном."""
    p = PARAMS[folder]
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, folder))
    try:
        monomer = _quiet(pt.open_mol, p['monomer_file'])
        parent = 'C' if folder == 'AF_546_cys' else 'LYS'
        template = 'CYS' if parent == 'C' else parent
        ref = os.path.join(pt.REF_TEMPLATES['protein_capped']['dir'], f'{template}.pdb')
        capped_ref = _quiet(pt.pdb_to_chem, ref, format_coord='3D')
        if isinstance(capped_ref, dict):
            capped_ref = capped_ref[template]
        mapped = _quiet(pt.renumber_residue_atoms, monomer, ref_base_name=parent,
                        residue_type='protein_capped', path_to_ref_mol=ref,
                        timeout=3, strict_parent=False)
        opened = _quiet(pt.open_peptide_boundaries, mapped, capped_ref, residue_number=2)
        if folder == 'Lysine_Malonyl':
            broken_ref = Chem.RWMol(capped_ref)
            c_idx = next(a.GetIdx() for a in capped_ref.GetAtoms()
                         if a.GetPDBResidueInfo().GetResidueNumber() == 2
                         and a.GetPDBResidueInfo().GetName().strip() == 'C')
            nme_n_idx = next(a.GetIdx() for a in capped_ref.GetAtoms()
                             if a.GetPDBResidueInfo().GetResidueNumber() == 3
                             and a.GetPDBResidueInfo().GetName().strip() == 'N')
            broken_ref.RemoveBond(c_idx, nme_n_idx)
            with pytest.raises(ValueError, match='обе пептидные границы N/C'):
                _quiet(pt.open_peptide_boundaries, mapped, broken_ref.GetMol(), residue_number=2)

        def parent_atom(mol, name):
            matches = [a for a in mol.GetAtoms() if a.HasProp('ParentName')
                       and a.GetProp('ParentName') == name]
            assert len(matches) == 1
            return matches[0]

        n, c = parent_atom(opened, 'N'), parent_atom(opened, 'C')
        ca, o = parent_atom(opened, 'CA'), parent_atom(opened, 'O')
        assert opened.GetBondBetweenAtoms(n.GetIdx(), ca.GetIdx()) is not None
        assert opened.GetBondBetweenAtoms(c.GetIdx(), ca.GetIdx()) is not None
        assert opened.GetBondBetweenAtoms(c.GetIdx(), o.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
        assert [a.GetSymbol() for a in n.GetNeighbors()].count('H') == 1
        assert {a.GetIdx() for a in c.GetNeighbors()} == {ca.GetIdx(), o.GetIdx()}
        before = {a.GetIntProp('PrevIdx'): a.GetSymbol() for a in mapped.GetAtoms()
                  if a.GetAtomicNum() > 1}
        after = {a.GetIntProp('PrevIdx'): a.GetSymbol() for a in opened.GetAtoms()
                 if a.GetAtomicNum() > 1}
        assert set(after) <= set(before)
        assert json.loads(opened.GetProp('BoundarySubstituents'))
        # Независимый эталон: сохранённый остаток, вырезанный старым тримерным путём.
        mon_name = _mon_name(folder)
        expected_path = os.path.join(
            ROOT, folder, 'molecules', 'substructure', f'4_{mon_name}.pdb')
        expected, _, _ = _pdb_names(expected_path)
        expected_mol = _quiet(pt.pdb_to_chem, expected_path, format_coord='3D')
        if isinstance(expected_mol, dict):
            expected_mol = next(iter(expected_mol.values()))
        named = _quiet(pt.rdkit_pdb_modification, opened, resname=p['base_name'], resid=2,
                       segid='A', naming=p['naming'],
                       parent_atoms=json.loads(opened.GetProp('ParentAtoms')))
        assert [a.GetProp('AtomName') for a in named.GetAtoms()] == expected
        assert _named_graph(named) == _named_graph(expected_mol)
        assert opened.GetProp('OldIndices') == json.dumps(
            [a.GetIntProp('PrevIdx') for a in opened.GetAtoms()])
        assert len(json.loads(opened.GetProp('ParentAtoms'))) == sum(
            a.HasProp('ParentName') for a in opened.GetAtoms())
        assert named.GetProp('Parent') == ('K' if parent == 'LYS' else 'C')
        assert named.GetProp('Naming') == p['naming']

        # HW для открытых валентностей появятся только в подготовленной копии для acpype.
        with_h = _quiet(pt.add_protons_and_renumber_H, named, resname=p['base_name'], resid=2)
        added = json.loads(with_h.GetProp('AddedH'))
        assert len(added) == 2
        assert {with_h.GetAtomWithIdx(i).GetNeighbors()[0].GetProp('AtomName') for i in added} == {'N', 'C'}
    finally:
        os.chdir(cwd)


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


def _named_graph(mol):
    """Состав и связи по AtomName, независимо от порядка атомов."""
    atoms = tuple(sorted((atom.GetProp('AtomName'), atom.GetSymbol(), atom.GetFormalCharge())
                         for atom in mol.GetAtoms()))
    bonds = tuple(sorted((min(bond.GetBeginAtom().GetProp('AtomName'), bond.GetEndAtom().GetProp('AtomName')),
                          max(bond.GetBeginAtom().GetProp('AtomName'), bond.GetEndAtom().GetProp('AtomName')),
                          str(bond.GetBondType())) for bond in mol.GetBonds()))
    return atoms, bonds


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
        _quiet(pt.save_chem_to_pdb, renum, os.path.join(tmp_dir, f'3_2_{mon}'), resname=p['base_name'],
               resid=2, segid='A', naming=p['naming'], parent_atoms=parent, no_H=True)
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
    with open(os.path.join(ROOT, folder, 'data', 'general_atoms.json')) as f:
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
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        job = pt.prepare_resp_job(residue, constraints, f'{mon}_test', working_dir=tmp_dir,
                                  fix_backbone=[(-60.0, -40.0)], n_sidechain=1, conformer_pool=40, monomer=monomer)
    with open(os.path.join(job, pt.SPEC_FILE)) as f:
        spec = json.load(f)
    # сходимость по умолчанию - gau_tight (R.E.D.); большой молекуле предлагается gau (2026-10-22)
    assert spec['qm']['g_convergence'] == 'gau_tight'
    assert ("g_convergence='gau'" in buf.getvalue()) == (len(spec['symbols']) > pt.resp.TIGHT_MAX_ATOMS)
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
    assert spec['qm']['optimizer'] == 'geometric'
    assert not any(c['contact'] for c in spec['conformers']), 'конформация с контактом боковой цепи'
    groups = [{names[i] for i in g} for g in spec['equivalent_groups']]
    if folder == 'Lysine_Malonyl':
        assert {'OK1', 'OK2'} in groups, 'кислороды карбоксилата должны быть эквивалентны'


def test_resp_job_folder(tmp_path):
    """Повторный prepare: та же задача - та же папка (прогресс сохраняется), другая - новая папка."""
    spec = {'name': 'x', 'qm': {'max_opt_steps': 400}}
    starts = {'alphaR_1': np.zeros((2, 3))}
    folder = str(tmp_path / 'job')
    assert pt._job_folder(folder, spec, starts) == folder
    os.makedirs(os.path.join(folder, 'conformers', 'alphaR_1'))
    with open(os.path.join(folder, pt.SPEC_FILE), 'w') as f:
        json.dump(spec, f)
    with open(os.path.join(folder, 'conformers', 'alphaR_1', 'start.xyz'), 'w') as f:
        f.write(pt._xyz_block(['C', 'H'], starts['alphaR_1'], 0))
    assert pt._job_folder(folder, spec, starts) == folder
    moved = {'alphaR_1': starts['alphaR_1'] + 0.1}
    assert _quiet(pt._job_folder, folder, spec, moved) == folder + '_01'
    assert _quiet(pt._job_folder, folder, dict(spec, qm={'max_opt_steps': 100}), starts) == folder + '_01'


def test_broken_bonds():
    """Развал геометрии при оптимизации (optking, 2026-10-14): связь, растянутая в 1.3 раза, - брак."""
    mol = Chem.AddHs(Chem.MolFromSmiles('CCO'))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    xyz = mol.GetConformer().GetPositions()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    assert pt._broken_bonds(symbols, xyz, xyz) == []
    moved = xyz.copy()
    moved[2] += (xyz[2] - xyz[1]) * 0.5            # C-O длиннее в 1.5 раза, O-H уезжает вместе с O
    assert [(i, j) for i, j, _ in pt._broken_bonds(symbols, xyz, moved)] == [(1, 2)]


@pytest.mark.slow
def test_optimize_records_gradients(tmp_path):
    """
    status.json: в истории запуска - и лимит шагов, и сколько градиентов сделано (2026-10-22,
    status показывал только лимит). Вода HF/STO-3G из искажённой геометрии, лимит 2 шага:
    geomeTRIC считает градиент в старте и по одному на шаг - 3 градиента, не сошлось.
    """
    pytest.importorskip('psi4')
    pytest.importorskip('geometric')
    folder = str(tmp_path)
    spec = {'symbols': ['O', 'H', 'H'], 'total_charge': 0, 'multiplicity': 1, 'frozen': [],
            'qm': {'method': 'hf', 'basis': 'sto-3g', 'g_convergence': 'gau', 'scf_type': 'pk',
                   'max_opt_steps': 2, 'continue_steps': 2, 'optimizer': 'geometric'}}
    with open(os.path.join(folder, pt.SPEC_FILE), 'w') as f:
        json.dump(spec, f)
    cdir = os.path.join(folder, 'conformers', 'w')
    os.makedirs(cdir)
    with open(os.path.join(cdir, 'start.xyz'), 'w') as f:
        f.write(pt._xyz_block(spec['symbols'], np.array([[0, 0, 0], [1.1, 0, 0], [-0.2, 1.0, 0]]), 0))
    status = _quiet(pt.resp._optimize_conformer, folder, 'w', n_threads=1, memory='500 MB')
    assert status['converged'] is False
    assert status['history'][-1]['steps'] == 2
    assert status['history'][-1]['gradients'] == 3


def test_independent_constraints_drop_redundant():
    """Линейно зависимые ограничения отбрасываются (иначе матрица psiresp вырождена)."""
    # 3 атома, полный заряд задаёт psiresp; фиксация всех трёх - одно ограничение лишнее
    fixed, sums, pairs, dropped = pt.resp._independent_constraints(3, {0: 0.1, 1: -0.2, 2: 0.1}, [], [[1, 2]])
    assert len(fixed) == 2 and dropped == 2


def test_rtp_roundtrip(tmp_path):
    """read_rtp -> write_rtp -> read_rtp не меняет ни одного остатка базового поля (заряды без округления)."""
    path = os.path.join(ROOT, 'amber14sb_parmbsc1_cufix.ff', 'aminoacids.rtp')
    bondedtypes, residues = pt.read_rtp(path)
    assert bondedtypes == pt.RTP_BONDEDTYPES
    assert {'LYS', 'CYS', 'ACE', 'NME'} <= set(residues)
    out = pt.write_rtp(str(tmp_path / 'copy.rtp'), residues, bondedtypes)
    assert pt.read_rtp(out) == (bondedtypes, residues)
    lys = residues['LYS']
    assert lys['atoms'][0] == ('N', 'N', -0.3479, 1) and ['N', 'H'] in lys['bonds']
    assert pt.read_rtp_charges('LYS')['NZ'] == -0.3854


@functools.lru_cache(maxsize=None)
def props_steps_1_to_4(folder):
    """Шаги 1-4 без словарей (templates/1_charge_calculation_props.ipynb): всё в памяти, без PDB."""
    p = PARAMS[folder]
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, folder))
    try:
        monomer = _quiet(pt.open_mol, p['monomer_file'])
        trimer = _quiet(pt.open_mol, p['trimer_file'])
        residue = _quiet(pt.find_residue, monomer, trimer, timeout=5)
        residue = _quiet(pt.renumber_residue_atoms, residue, ref_base_name=p['parent_residue'], timeout=5)
        parent = [a.GetIdx() for a in residue.GetAtoms() if a.HasProp('ParentName')]
        residue = _quiet(pt.rdkit_pdb_modification, residue, resname=p['base_name'], resid=2, segid='A',
                         naming=p['naming'], parent_atoms=parent)
        return monomer, trimer, residue
    finally:
        os.chdir(cwd)


# Lysine_lac: SMILES с [NH2] - после вырезания N без неявного H (найдено 2026-10-10)
@pytest.mark.parametrize('folder', CASES + ['Lysine_lac'])
def test_props_path_matches_committed(folder):
    """Свойства молекулы вместо словарей: те же имена, порядок и атомы родителя, что в git."""
    monomer, trimer, residue = props_steps_1_to_4(folder)
    mon = _mon_name(folder)
    names, resnames, _ = _pdb_names(os.path.join(ROOT, folder, 'molecules', 'substructure', f'4_{mon}.pdb'))
    assert [a.GetProp('AtomName') for a in residue.GetAtoms()] == names
    assert resnames == {PARAMS[folder]['base_name']}
    general = json.load(open(os.path.join(ROOT, folder, 'data', 'general_atoms.json')))
    assert [a.GetProp('AtomName') for a in residue.GetAtoms() if a.HasProp('ParentName')] == general
    # сквозные индексы указывают на тот же элемент в мономере и тримере
    for a in residue.GetAtoms():
        assert monomer.GetAtomWithIdx(a.GetIntProp('SrcIdx')).GetSymbol() == a.GetSymbol()
        assert trimer.GetAtomWithIdx(a.GetIntProp('PolymerIdx')).GetSymbol() == a.GetSymbol()
    assert residue.GetProp('Parent') == PARAMS[folder]['parent_residue']
    assert residue.GetProp('Naming') in ('greek', 'index')


@pytest.mark.parametrize('folder', ['Lysine_Malonyl', 'Lysine_lac'])
def test_boundary_extraction_is_invariant_to_smiles_atom_order(folder):
    """Повторная сериализация того же графа не меняет residue из MCS-пути."""
    p = PARAMS[folder]
    cwd = os.getcwd()
    os.chdir(os.path.join(ROOT, folder))
    try:
        source = _quiet(pt.open_mol, p['monomer_file'])
        parent = 'LYS'
        ref_path = os.path.join(pt.REF_TEMPLATES['protein_capped']['dir'], 'LYS.pdb')
        capped_ref = _quiet(pt.pdb_to_chem, ref_path, format_coord='3D')
        if isinstance(capped_ref, dict):
            capped_ref = capped_ref['LYS']
        expected_path = os.path.join(
            ROOT, folder, 'molecules', 'substructure', f'4_{_mon_name(folder)}.pdb')
        expected_names, _, _ = _pdb_names(expected_path)
        expected_mol = _quiet(pt.pdb_to_chem, expected_path, format_coord='3D')
        if isinstance(expected_mol, dict):
            expected_mol = next(iter(expected_mol.values()))
        params = Chem.SmilesParserParams()
        params.removeHs = False

        for seed in range(3):
            smiles = Chem.MolToSmiles(source, canonical=False, isomericSmiles=True,
                                      doRandom=True, rootedAtAtom=seed)
            variant = Chem.MolFromSmiles(smiles, params)
            assert variant is not None
            mapped = _quiet(pt.renumber_residue_atoms, variant, ref_base_name=parent,
                            residue_type='protein_capped', path_to_ref_mol=ref_path,
                            timeout=3, strict_parent=False)
            residue = _quiet(pt.open_peptide_boundaries, mapped, capped_ref, residue_number=2)
            residue = _quiet(pt.rdkit_pdb_modification, residue, resname=p['base_name'], resid=2,
                             segid='A', naming=p['naming'],
                             parent_atoms=json.loads(residue.GetProp('ParentAtoms')))
            assert [atom.GetProp('AtomName') for atom in residue.GetAtoms()] == expected_names
            assert _named_graph(residue) == _named_graph(expected_mol)
    finally:
        os.chdir(cwd)


def test_generated_workflow_has_stage_specific_params():
    """Три генератора не подставляют параметры ноутбука 1 в этапы 2/3."""
    for folder, expected in notebook_builder.WORKFLOW_PARAMS.items():
        for notebook_name in ('1_charge_calculation.ipynb', '2_acpype.ipynb', '3_forcefield_md.ipynb'):
            path = os.path.join(ROOT, folder, notebook_name)
            nb = json.load(open(path))
            assert nb['metadata']['respartools']['workflow'] == 'properties_v1'
            params = next(''.join(c['source']) for c in nb['cells'] if c.get('id') == 'params')
            assert f'FOLDER = {folder!r}' in params
            assert f'base_name = {expected["base_name"]!r}' in params
            assert f'monomer_file = {expected["monomer_file"]!r}' in params
            if notebook_name == '2_acpype.ipynb':
                assert 'INPUT_BASE =' in params and 'INPUT_H =' in params
                assert 'FF_OUT =' not in params
            elif notebook_name == '3_forcefield_md.ipynb':
                assert 'INPUT =' in params and 'amber14sb_mod_smi.ff' in params
                assert 'md_from_smiles' in params
            else:
                parent = expected.get('parent_override')
                assert f'parent_override = {parent!r}' in params and 'trimer_file' not in params
                charges = next(''.join(c['source']) for c in nb['cells'] if c.get('id') == 's5')
                assert 'show_names=False, show_index=False' in charges
                assert '05_espaloma_charges.png' in charges


def test_notebook_archive_detection_and_collision_names(tmp_path):
    assert notebook_builder._has_outputs({
        'cells': [{'cell_type': 'code', 'outputs': [], 'execution_count': 3}]})
    assert not notebook_builder._has_outputs({
        'cells': [{'cell_type': 'code', 'outputs': [], 'execution_count': None}]})

    source = tmp_path / '1.ipynb'
    source.write_text('new')
    archive = tmp_path / 'archive' / 'notebooks'
    archive.mkdir(parents=True)
    (archive / '1.ipynb').write_text('original')
    suffix = notebook_builder.date.today().isoformat()
    (archive / f'1_{suffix}.ipynb').write_text('same-day archive')
    (archive / f'1_{suffix}_2.ipynb').write_text('second archive')
    archived = notebook_builder._archive_existing(str(source))
    assert archived.endswith(f'1_{suffix}_3.ipynb')
    assert open(archived).read() == 'new'


def test_extract_residue_requires_chain_when_residue_key_is_ambiguous():
    path = os.path.join(pt.REF_TEMPLATES['protein_capped']['dir'], 'LYS.pdb')
    capped = _quiet(pt.pdb_to_chem, path, format_coord='3D')
    if isinstance(capped, dict):
        capped = capped['LYS']
    center = _quiet(pt.extract_residue, capped, 2)
    combined = Chem.RWMol()
    for chain_id in ('A', 'B'):
        offset = combined.GetNumAtoms()
        for atom in center.GetAtoms():
            copied = Chem.Atom(atom)
            old_info = atom.GetPDBResidueInfo()
            info = Chem.AtomPDBResidueInfo(
                old_info.GetName(), old_info.GetSerialNumber(), old_info.GetAltLoc(),
                old_info.GetResidueName(), old_info.GetResidueNumber(), chain_id,
                old_info.GetInsertionCode(), old_info.GetOccupancy(), old_info.GetTempFactor(),
                old_info.GetIsHeteroAtom())
            copied.SetMonomerInfo(info)
            combined.AddAtom(copied)
        for bond in center.GetBonds():
            combined.AddBond(offset + bond.GetBeginAtomIdx(), offset + bond.GetEndAtomIdx(),
                             bond.GetBondType())
    combined = combined.GetMol()

    with pytest.raises(ValueError, match='задайте chain_id'):
        pt.extract_residue(combined, 2)
    selected = pt.extract_residue(combined, 2, chain_id='B')
    assert selected.GetNumAtoms() == center.GetNumAtoms()
    assert {a.GetPDBResidueInfo().GetChainId() for a in selected.GetAtoms()} == {'B'}


@pytest.mark.parametrize('folder', ['Lysine_Formyl', 'Lysine_Malonyl'])
def test_save_load_mol_roundtrip(folder, tmp_path):
    _, _, residue = props_steps_1_to_4(folder)
    residue = Chem.Mol(residue)
    q = [0.001 * i for i in range(residue.GetNumAtoms())]
    pt.set_charges(residue, q, 'Espaloma', {0: q[0]})
    back = pt.load_mol(_quiet(pt.save_mol, residue, str(tmp_path / 'r')))
    assert back.GetNumAtoms() == residue.GetNumAtoms()
    for a, b in zip(residue.GetAtoms(), back.GetAtoms()):
        assert (a.GetSymbol(), a.GetFormalCharge(), a.GetNumImplicitHs()) == \
               (b.GetSymbol(), b.GetFormalCharge(), b.GetNumImplicitHs())
        assert a.GetPropsAsDict(includePrivate=False) == b.GetPropsAsDict(includePrivate=False)
        assert b.GetPDBResidueInfo().GetResidueName() == PARAMS[folder]['base_name']
    assert back.GetProp('ChargeMethod') == 'Espaloma' and back.GetProp('Parent') == 'K'


@pytest.mark.parametrize('folder', ['Lysine_Formyl', 'Lysine_Malonyl', 'Lysine_3M'])
def test_run_acpype(folder, tmp_path):
    """Шаг 2: заряд для acpype - из молекулы (Malonyl -1, 3M +1), типы в атомах, порядок сверен."""
    _, _, residue = props_steps_1_to_4(folder)
    residue = Chem.Mol(residue)
    _quiet(pt.run_acpype, residue, str(tmp_path))
    types = {a.GetProp('AtomName'): a.GetProp('AcpypeType') for a in residue.GetAtoms()}
    assert len(types) == residue.GetNumAtoms()
    assert types['CA'] == 'c3' and types['O'] == 'o' and types['HB1'] == 'hc'
    itp = pt.read_itp(residue.GetProp('AcpypeItp'))
    assert len(itp['atoms']) == residue.GetNumAtoms() + 2          # + два водорода только для acpype
    assert abs(sum(float(r[6]) for r in itp['atoms']) - Chem.GetFormalCharge(residue)) < 1e-3
    if folder == 'Lysine_3M':
        assert types['NZ'] == 'n4'
    if folder == 'Lysine_Malonyl':
        assert types['OK1'] == types['OK2'] == 'o'


def _base_hdb():
    hdb, cur = {}, None
    for line in open(os.path.join(ROOT, 'amber14sb_parmbsc1_cufix.ff', 'aminoacids.hdb')):
        p = line.split()
        if len(p) == 2 and not p[0].isdigit():
            cur = hdb.setdefault(p[0], [])
        elif p:
            cur.append((int(p[0]), int(p[1]), *p[2:]))
    return hdb


def test_hdb_entries_match_base_hdb():
    """Генератор hdb по графу rtp воспроизводит aminoacids.hdb amber14sb (тип и управляющие атомы)."""
    _, residues = pt.read_rtp(os.path.join(ROOT, 'amber14sb_parmbsc1_cufix.ff', 'aminoacids.rtp'))
    base = _base_hdb()
    c_term_oxygen = (1, 1, 'OC1', 'C', 'CA', 'OC2')     # особая запись C-концов: O как «водород»
    checked = 0
    for name, res in residues.items():
        if name not in base or 'bonds' not in res or name in ('HOH', 'HO4', 'HO5', 'NME', 'NHE', 'NH2'):
            continue
        got = set(map(tuple, pt.hdb_entries(res['atoms'], res['bonds'])))
        want = set(base[name]) - {c_term_oxygen}
        if name == 'NPRO':          # тот же тип, управляющие атомы в другом порядке
            want = {(2, 6, 'H', 'N', 'CD', 'CA') if e == (2, 6, 'H', 'N', 'CA', 'CD') else e for e in want}
        assert got == want, name
        checked += 1
    assert checked > 70


def test_write_forcefield_formyl(tmp_path):
    """Шаг 3: типы по окружению, rtp и hdb остатка; повторная запись не дублирует строки."""
    _, _, residue = props_steps_1_to_4('Lysine_Formyl')
    residue = Chem.Mol(residue)
    _quiet(pt.run_acpype, residue, str(tmp_path / 'acpype'))
    q = json.load(open(os.path.join(ROOT, 'Lysine_Formyl', 'data', 'AI_chrges_K_form.json')))
    pt.set_charges(residue, q, 'Espaloma')
    ff = str(tmp_path / 'test.ff')
    for _ in range(2):
        rtp, explicit = _quiet(pt.write_forcefield, Chem.Mol(residue), ff)
    _quiet(pt.write_forcefield, residue, ff)
    types = {a.GetProp('AtomName'): a.GetProp('AtomType') for a in residue.GetAtoms()}
    lys = dict((n, t) for n, t, *_ in pt.read_rtp(os.path.join(ROOT, 'amber14sb_parmbsc1_cufix.ff', 'aminoacids.rtp'))[1]['LYS']['atoms'])
    for name in ('N', 'H', 'CA', 'HA', 'CB', 'HB1', 'CD', 'CE', 'C', 'O'):
        assert types[name] == lys[name], name          # окружение как в лизине - тип лизина
    assert types['HE1'] == 'h1' and types['NZ'] == 'n'  # изменённое окружение: типы GAFF
    res = pt.read_rtp(rtp)[1]['KFO']
    assert [a[0] for a in res['atoms']] == [a.GetProp('AtomName') for a in residue.GetAtoms()]
    assert abs(sum(a[2] for a in res['atoms'])) < 1e-4
    assert ('C8', 'n') in explicit['bonds']             # в amber14sb нет связи C8 с типом GAFF n
    assert open(os.path.join(ff, 'residuetypes.dat')).read().count('KFO\t') == 1
    assert open(os.path.join(ff, 'KFO.hdb')).read().startswith('KFO\t8\n')


def test_hybridization_for_types():
    """Гибридизация для assign_types: N и C остова - sp2 и в шаблоне, и в остатке (через ACE-X-NME)."""
    _, _, residue = props_steps_1_to_4('Lysine_Formyl')
    lys = pt._template_hybridization('LYS')
    assert lys['N'][0] == lys['C'][0] == 'SP2' and lys['NZ'][0] == 'SP3'
    hyb = dict(zip([a.GetProp('AtomName') for a in residue.GetAtoms()], pt._hybridization(residue)))
    assert hyb['N'][0] == hyb['C'][0] == 'SP2', 'N остова без кэпа выглядел бы как sp3'
    assert hyb['NZ'][0] == 'SP2' and hyb['CE'] == lys['CE']   # амид NZ - sp2, CE как в лизине


def test_charges_json_atom_order(tmp_path):
    """JSON зарядов с именами атомов: при чтении порядок сверяется; старый формат - с предупреждением."""
    names, q = ['N', 'H', 'CA'], [-0.3, 0.2, 0.1]
    _quiet(pt.save_charges_json, 'new', q, str(tmp_path), atom_names=names)
    assert pt.load_charges_json(str(tmp_path / 'new.json'), names) == q
    with pytest.raises(ValueError, match='порядок атомов'):
        pt.load_charges_json(str(tmp_path / 'new.json'), ['H', 'N', 'CA'])
    _quiet(pt.save_charges_json, 'old', q, str(tmp_path))
    assert json.load(open(tmp_path / 'old.json')) == q                       # старый формат - список
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert pt.load_charges_json(str(tmp_path / 'old.json'), names) == q
    assert 'порядок атомов не проверен' in buf.getvalue()
    with pytest.raises(ValueError, match='зарядов 3, атомов 2'):
        pt.load_charges_json(str(tmp_path / 'old.json'), names[:2])


def test_validate_vacuum_formyl(tmp_path):
    """Проверка в GROMACS: pdb2gmx -ignh (водороды по hdb), grompp без предупреждений, минимизация."""
    try:
        pt.gmx._gmx_bin()
    except FileNotFoundError:
        pytest.skip('нет GROMACS (окружение gmx)')
    monomer, _, residue = props_steps_1_to_4('Lysine_Formyl')
    residue = Chem.Mol(residue)
    _quiet(pt.run_acpype, residue, str(tmp_path / 'acpype'))
    pt.set_charges(residue, json.load(open(os.path.join(ROOT, 'Lysine_Formyl', 'data', 'AI_chrges_K_form.json'))), 'Espaloma')
    ff = str(tmp_path / 'amber14sb_test.ff')
    _quiet(pt.write_forcefield, residue, ff)
    work = _quiet(pt.write_validate_job, residue, monomer, ff, str(tmp_path / 'validate'))
    rep = _quiet(pt.run_validate, work, n_threads=1)
    assert rep['ok'], rep['problems']
    assert rep['stages'] == ['pdb2gmx', 'grompp', 'em_vac'] and rep['atoms_match']


@pytest.mark.parametrize('code, parent', [('4PQ', 'TRP'), ('ILX', 'ILE')])
def test_capped_parent_names_for_ring_and_branches(code, parent):
    params = notebook_builder.WORKFLOW_PARAMS[code]
    source = _quiet(pt.open_mol, os.path.join(ROOT, code, params['monomer_file']))
    ref_path = os.path.join(pt.REF_TEMPLATES['protein_capped']['dir'], f'{parent}.pdb')
    ref = _quiet(pt.pdb_to_chem, ref_path, format_coord='3D')
    if isinstance(ref, dict):
        ref = ref[parent]
    central = _quiet(pt.extract_residue, ref, 2)
    Chem.GetSymmSSSR(central)
    ring_bonds = {
        frozenset(a.GetProp('AtomName') for a in (b.GetBeginAtom(), b.GetEndAtom()))
        for b in central.GetBonds() if b.GetIsAromatic() and b.IsInRing()
    }
    parser = Chem.SmilesParserParams()
    parser.removeHs = False
    reversed_source = Chem.RenumberAtoms(source, list(reversed(range(source.GetNumAtoms()))))
    reordered = Chem.MolFromSmiles(Chem.MolToSmiles(reversed_source, canonical=False), parser)
    for monomer in (source, reordered):
        mapped = _quiet(pt.renumber_residue_atoms, monomer, ref_base_name=parent,
                        residue_type='protein_capped', path_to_ref_mol=ref_path)
        residue = _quiet(pt.open_peptide_boundaries, mapped, ref)
        _quiet(pt.rdkit_pdb_modification, residue, resname=code, parent_reference=central)
        names = {a.GetProp('AtomName'): a for a in residue.GetAtoms()}
        assert len(names) == residue.GetNumAtoms()
        if code == '4PQ':
            assert len(ring_bonds) == 10
            assert ring_bonds == {
                frozenset(a.GetProp('AtomName') for a in (b.GetBeginAtom(), b.GetEndAtom()))
                for b in residue.GetBonds() if b.GetIsAromatic() and b.IsInRing()
            }
            assert {a.GetProp('AtomName') for a in names['OH1'].GetNeighbors()} == {'CZ3', 'HH1'}
            assert {a.GetProp('AtomName') for a in names['HH2'].GetNeighbors()} == {'CH2'}
        else:
            assert names['CD2'].GetProp('ParentName') == 'CD'
            assert {a.GetProp('AtomName') for a in names['CG1'].GetNeighbors()} >= {'OD1', 'CD2'}
            assert names['CG2'].GetProp('ParentName') == 'CG2'


def test_manual_atom_names_by_index_are_atomic():
    mol = Chem.MolFromSmiles('CC')
    for atom, name in zip(mol.GetAtoms(), ('CA', 'CB')):
        atom.SetProp('AtomName', name)
        atom.SetProp('ParentName', name)
    for mapping, message in (({0: 'CB'}, 'CB.*0.*1'), ({2: 'CX'}, '2'), ({1: 'C BAD'}, 'C BAD')):
        with pytest.raises(ValueError, match=message):
            pt.modifie_residue_info(mol, mapping)
        assert [a.GetProp('AtomName') for a in mol.GetAtoms()] == ['CA', 'CB']
        assert all(a.GetPDBResidueInfo() is None for a in mol.GetAtoms())

    assert pt.modifie_residue_info(mol, {0: 'CB', 1: 'CA'}) is mol
    assert [a.GetProp('AtomName') for a in mol.GetAtoms()] == ['CB', 'CA']
    assert [a.GetPDBResidueInfo().GetName().strip() for a in mol.GetAtoms()] == ['CB', 'CA']
    assert [a.GetProp('ParentName') for a in mol.GetAtoms()] == ['CA', 'CB']

    mol.GetAtomWithIdx(1).SetProp('AtomName', 'CB')
    pt.modifie_residue_info(mol, {1: 'CX'})
    assert [a.GetProp('AtomName') for a in mol.GetAtoms()] == ['CB', 'CX']
    with pytest.warns(FutureWarning, match='устарел'):
        pt.modifie_residue_info(mol, {'CX': 'CY'})
    assert mol.GetAtomWithIdx(1).GetProp('AtomName') == 'CY'
