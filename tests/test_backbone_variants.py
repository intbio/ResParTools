"""
Изменённый остов: N-метил, тиоамид, депсипептид (2026-10-30).

Остатки строятся из SMILES (мономер и тример Gly-X-Gly, как в validation/), родитель - явно.
Ожидаемые имена выписаны вручную: заместитель на N - CN, HN1-3 (PDB CCD MLE, MVA, SAR),
тиокарбонил - S (CCD GL3, тиоглицин), O вместо N у гидроксикислоты - OHN (CCD 2OP, LAC);
остальные атомы - как у родителя в aminoacids.rtp amber14sb.
"""
import contextlib
import io
import json
import os

import numpy as np
import pytest
import ResParTools as pt  # путь к пакету - tests/conftest.py
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolTransforms

RDLogger.DisableLog('rdApp.*')

# имя: (родитель, rtp родителя, мономер, тример Gly-X-Gly, ожидаемые имена,
#       атомы остова с зарядом из rtp - атом и все его соседи из родителя)
CASES = {
    'NMe-Ala': ('A', 'ALA', 'CN[C@@H](C)C(=O)O', 'NCC(=O)N(C)[C@@H](C)C(=O)NCC(=O)O',
                'N CN HN1 HN2 HN3 CA HA CB HB1 HB2 HB3 C O', 'CA HA C O'),
    'NMe-Lys': ('K', 'LYS', 'CN[C@@H](CCCC[NH3+])C(=O)O', 'NCC(=O)N(C)[C@@H](CCCC[NH3+])C(=O)NCC(=O)O',
                'N CN HN1 HN2 HN3 CA HA CB HB1 HB2 CG HG1 HG2 CD HD1 HD2 CE HE1 HE2 NZ HZ1 HZ2 HZ3 C O',
                'CA HA C O'),
    'thio-Ala': ('A', 'ALA', 'N[C@@H](C)C(=S)O', 'NCC(=O)N[C@@H](C)C(=S)NCC(=O)O',
                 'N H CA HA CB HB1 HB2 HB3 C S', 'N H CA HA'),
    'Lac-depsi': ('A', 'ALA', 'O[C@@H](C)C(=O)O', 'NCC(=O)O[C@@H](C)C(=O)NCC(=O)O',
                  'OHN CA HA CB HB1 HB2 HB3 C O', 'HA C O'),
}


def _quiet(func, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def _named(case):
    parent, _, mon_smi, tri_smi, _, _ = CASES[case]
    mon = Chem.AddHs(Chem.MolFromSmiles(mon_smi))
    tri = Chem.AddHs(Chem.MolFromSmiles(tri_smi))
    mon.SetProp('_Name', 'monomer')
    tri.SetProp('_Name', 'trimer')
    res = _quiet(pt.find_residue, mon, tri, timeout=5)
    res = _quiet(pt.renumber_residue_atoms, res, ref_base_name=parent, timeout=5)
    parent_atoms = [a.GetIdx() for a in res.GetAtoms() if a.HasProp('ParentName')]
    named = _quiet(pt.rdkit_pdb_modification, Chem.Mol(res), resname='XXX', resid=2, naming='greek',
                   parent_atoms=parent_atoms)
    return mon, named


@pytest.mark.parametrize('case', CASES)
def test_residue_names(case):
    """Остаток выделен целиком (H заместителя на N не теряются) и назван по CCD/amber."""
    _, named = _named(case)
    assert ' '.join(a.GetProp('AtomName') for a in named.GetAtoms()) == CASES[case][4]


def test_modified_backbone_sdf_roundtrip(tmp_path):
    """Флаг ModifiedBackbone сохраняется в SDF как bool для следующих шагов пайплайна."""
    _, named = _named('NMe-Ala')
    saved = pt.save_mol(named, str(tmp_path / 'nme'))
    loaded = pt.load_mol(saved)
    assert loaded.HasProp('ModifiedBackbone') and loaded.GetBoolProp('ModifiedBackbone')


@pytest.mark.parametrize('case', CASES)
def test_backbone_charge_constraints(case, capsys):
    """Для остова, изменённого химически, стандартные заряды rtp не фиксируются."""
    _, named = _named(case)
    constraints = pt.charge_constraints_from_rtp(named, CASES[case][1])
    assert constraints == {}
    assert 'стандартные заряды из rtp для этого остатка неприменимы' in capsys.readouterr().out


def _omega(conf, a, b, c, d):
    return abs(rdMolTransforms.GetDihedralDeg(conf, a, b, c, d))


@pytest.mark.parametrize('case', CASES)
def test_resp_conformers_trans(case, tmp_path):
    """Кэпированный остаток для RESP: обе связи остатка с кэпами (амид, эфир, тиоамид) - транс."""
    mon, named = _named(case)
    constraints = _quiet(pt.charge_constraints_from_rtp, named, CASES[case][1])
    job = _quiet(pt.prepare_resp_job, named, constraints, case, working_dir=str(tmp_path),
                 fix_backbone=[(-60.0, -40.0)], n_sidechain=2, conformer_pool=40, monomer=mon)
    with open(os.path.join(job, pt.SPEC_FILE)) as f:
        spec = json.load(f)
    mol = Chem.MolFromMolBlock(spec['molblock'], removeHs=False)
    phi, psi = spec['dihedrals']['phi'], spec['dihedrals']['psi']
    # ω: атом кэпа - C кэпа - атом остова - CA и CA - C - N кэпа - атом кэпа
    def heavy_nbr(i, exclude):
        return next(n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()
                    if n.GetAtomicNum() == 6 and n.GetIdx() != exclude)
    omegas = [(heavy_nbr(phi[0], phi[1]), phi[0], phi[1], phi[2]),
              (psi[1], psi[2], psi[3], heavy_nbr(psi[3], psi[2]))]
    for c in spec['conformers']:
        xyz = np.loadtxt(os.path.join(job, 'conformers', c['label'], 'start.xyz'), skiprows=2, usecols=(1, 2, 3))
        conf = mol.GetConformer()
        for i, p in enumerate(xyz):
            conf.SetAtomPosition(i, p.tolist())
        assert all(_omega(conf, *w) > 150 for w in omegas), f"{c['label']}: цис-связь с кэпом"


@pytest.mark.parametrize('case, improper_n, first, typed_name, atom_type', [
    ('NMe-Ala', ['-C', 'CA', 'N', 'CN'], 'N', 'N', 'n'),
    ('NMe-Lys', ['-C', 'CA', 'N', 'CN'], 'N', 'N', 'n'),
    ('thio-Ala', ['-C', 'CA', 'N', 'H'], 'N', 'S', 's'),
    ('Lac-depsi', None, 'OHN', 'OHN', 'os'),
])
def test_capped_acpype_rtp_terms(case, improper_n, first, typed_name, atom_type, tmp_path):
    """rtp: связь с предыдущим остатком и импроперы остова - по атомам этого остатка (у N-метила
    при N - CN вместо H, у депсипептида при OHN импропера нет)."""
    _, named = _named(case)
    _quiet(pt.run_acpype, named, str(tmp_path / 'acpype'))
    pt.set_charges(named, [Chem.GetFormalCharge(named) / named.GetNumAtoms()] * named.GetNumAtoms(), 'Espaloma')
    parent = pt.AMINO_ACIDS[named.GetProp('Parent')][0].upper()
    parent_rtp = pt.read_rtp(os.path.join(pt.DEFAULT_FF, 'aminoacids.rtp'))[1][parent]
    _quiet(pt.assign_types, named, parent_rtp, parent)
    res, _ = _quiet(pt.residue_rtp, named, pt.DEFAULT_FF)
    assert any(set(b[:2]) == {'-C', first} for b in res['bonds'])
    if improper_n:
        assert any(set(i[:4]) == set(improper_n) for i in res['impropers'])
    typed = next(a for a in named.GetAtoms() if a.GetProp('AtomName') == typed_name)
    assert typed.GetProp('AtomType') == atom_type
