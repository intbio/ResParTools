"""
Стандартные аминокислоты через путь шагов 1-4 (мономер в тримере Gly-X-Gly, перенумерация по
шаблону родителя, имена атомов): имена и их порядок должны совпасть с aminoacids.rtp amber14sb.
Имена проверяются дважды: с атомами родителя (как у модификаций) и только по правилу имён
(parent_atoms=None) - правило должно само воспроизводить номенклатуру amber.

Не проверяются: Gly (нет шаблона GGG - родителем модификации не бывает), Pro (кольцо через N
остова - путь мономер-в-тримере для него не проверен).
"""
import contextlib
import io
import os

import pytest
from rdkit import Chem

import ResParTools as pt   # путь к пакету - validation/conftest.py

BASE_FF = os.path.join(os.path.dirname(pt.__file__), 'amber14sb_parmbsc1_cufix.ff')

# одна буква: SMILES боковой цепи (протонирование - как у остатка rtp)
SIDE_CHAINS = {
    'A': 'C', 'S': 'CO', 'C': 'CS', 'V': 'C(C)C', 'L': 'CC(C)C', 'I': '[C@@H](C)CC', 'T': '[C@@H](C)O',
    'M': 'CCSC', 'F': 'Cc1ccccc1', 'Y': 'Cc1ccc(O)cc1', 'W': 'Cc1c[nH]c2ccccc12', 'H': 'Cc1c[nH]cn1',
    'N': 'CC(N)=O', 'Q': 'CCC(N)=O', 'D': 'CC(=O)[O-]', 'E': 'CCC(=O)[O-]', 'K': 'CCCC[NH3+]',
    'R': 'CCCNC(=[NH2+])N',
}
RTP_NAME = {'H': 'HIE'}   # гистидин: протон на NE2 (HIE), как SMILES выше

# Известные расхождения (2026-10-22). strict=True: когда исправят - тест сообщит, пометку снять.
KNOWN = {
    'T': 'правило имён: ветви уровня G по порядку атомов -> CG1/OG2, в amber OG1/CG2 (гетероатом - 1)',
    'I': 'правило имён: водороды метила CG2 - HG13 вместо HG23',
    'W': 'именование Trp (CZ2/CZ3/CH2) отложено до модификаций Trp',
}
CASES = [pytest.param(a, marks=pytest.mark.xfail(reason=KNOWN[a], strict=True)) if a in KNOWN else a
         for a in SIDE_CHAINS]


def _rtp_names(resname):
    _, residues = pt.read_rtp(os.path.join(BASE_FF, 'aminoacids.rtp'))
    return [a[0] for a in residues[resname]['atoms']]


def _residue(one):
    side = SIDE_CHAINS[one]
    mon = Chem.AddHs(Chem.MolFromSmiles(f'N[C@@H]({side})C(=O)O'))
    tri = Chem.AddHs(Chem.MolFromSmiles(f'NCC(=O)N[C@@H]({side})C(=O)NCC(=O)O'))
    mon.SetProp('_Name', 'monomer')
    tri.SetProp('_Name', 'trimer')
    with contextlib.redirect_stdout(io.StringIO()):
        res = pt.find_residue(mon, tri, timeout=5)
        return pt.renumber_residue_atoms(res, ref_base_name=one, timeout=5)


@pytest.mark.parametrize('one', CASES)
@pytest.mark.parametrize('with_parent', [True, False], ids=['parent', 'rule_only'])
def test_standard_residue_names(one, with_parent):
    resname = RTP_NAME.get(one, pt.AMINO_ACIDS[one][0].upper())
    res = _residue(one)
    parent = [a.GetIdx() for a in res.GetAtoms() if a.HasProp('ParentName')] if with_parent else None
    with contextlib.redirect_stdout(io.StringIO()):
        named = pt.rdkit_pdb_modification(Chem.Mol(res), resname=resname, resid=2, naming='greek',
                                          parent_atoms=parent)
    assert [a.GetProp('AtomName') for a in named.GetAtoms()] == _rtp_names(resname)
