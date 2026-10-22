"""
Аминокислоты базового поля через путь шагов 1-4 (мономер в тримере Gly-X-Gly, перенумерация по
шаблону родителя, имена атомов): имена и их порядок должны совпасть с aminoacids.rtp amber14sb.
Имена проверяются дважды: с атомами родителя (как у модификаций) и только по правилу имён
(parent_atoms=None) - правило должно само воспроизводить номенклатуру amber.

Мономер и тример строятся здесь из SMILES боковой цепи (не из файлов папок модификаций);
родитель задаётся явно (ref_base_name) - шаблон берётся из molecules/aminoacids_template/.
Проверяются все остатки aminoacids.rtp с атомом CA; те, что путь пока не поддерживает, помечены
xfail с причиной (концевые остатки, нет шаблона родителя, дисульфид).
"""
import contextlib
import io
import os

import pytest
from rdkit import Chem

import ResParTools as pt   # путь к пакету - validation/conftest.py

BASE_FF = os.path.join(os.path.dirname(pt.__file__), 'amber14sb_parmbsc1_cufix.ff')

# имя в rtp: (родитель, SMILES боковой цепи; протонирование - как у остатка rtp)
RESIDUES = {
    'ALA': ('A', 'C'), 'SER': ('S', 'CO'), 'CYS': ('C', 'CS'), 'CYM': ('C', 'C[S-]'),
    'VAL': ('V', 'C(C)C'), 'LEU': ('L', 'CC(C)C'), 'ILE': ('I', '[C@@H](C)CC'), 'THR': ('T', '[C@@H](C)O'),
    'MET': ('M', 'CCSC'), 'PHE': ('F', 'Cc1ccccc1'), 'TYR': ('Y', 'Cc1ccc(O)cc1'),
    'TRP': ('W', 'Cc1c[nH]c2ccccc12'),
    'HIE': ('H', 'Cc1c[nH]cn1'), 'HID': ('H', 'Cc1cnc[nH]1'), 'HIP': ('H', 'Cc1c[nH]c[nH+]1'),
    'ASN': ('N', 'CC(N)=O'), 'GLN': ('Q', 'CCC(N)=O'),
    'ASP': ('D', 'CC(=O)[O-]'), 'ASH': ('D', 'CC(=O)O'), 'GLU': ('E', 'CCC(=O)[O-]'), 'GLH': ('E', 'CCC(=O)O'),
    'LYS': ('K', 'CCCC[NH3+]'), 'LYN': ('K', 'CCCCN'), 'ARG': ('R', 'CCCNC(=[NH2+])N'),
}
# Путь пока не поддерживает (2026-10-22): xfail, причина видна в выводе
UNSUPPORTED = {
    'GLY': 'нет шаблона родителя GGG (Gly родителем модификации не бывает)',
    'PRO': 'кольцо через N остова: путь мономер-в-тримере для Pro не сделан',
    'HYP': 'гидроксипролин: кольцо через N остова, нет шаблона',
    'ORN': 'орнитин: нет шаблона родителя',
    'DAB': '2,4-диаминомасляная кислота: нет шаблона родителя',
    'CYX': 'цистеин в дисульфиде: нужен партнёр по связи S-S',
}
TERMINAL = 'концевые остатки (N*/C*): параметризация N- и C-концов - отдельная ветка'
# Известные расхождения правила имён (2026-10-22). strict=True: когда исправят - тест сообщит.
KNOWN = {
    'TRP': 'кольцо: номера CZ2/CZ3/CH2 в amber по ветви, а не по уровню - именование отложено',
    'CYM': 'тиолат: SG с зарядом -1 не сопоставляется с шаблоном цистеина (ASP с O- - сопоставляется)',
}


def _rtp():
    _, residues = pt.read_rtp(os.path.join(BASE_FF, 'aminoacids.rtp'))
    return {name: [a[0] for a in res['atoms']] for name, res in residues.items()
            if {'N', 'CA', 'C'} <= {a[0] for a in res['atoms']}}


RTP = _rtp()


def _case(name):
    if name in KNOWN:
        return pytest.param(name, marks=pytest.mark.xfail(reason=KNOWN[name], strict=True))
    if name in UNSUPPORTED or name not in RESIDUES:
        reason = UNSUPPORTED.get(name, TERMINAL if name[0] in 'NC' and name[1:] in RTP else 'нет SMILES в тесте')
        return pytest.param(name, marks=pytest.mark.xfail(reason=reason, strict=True, run=False))
    return name


def test_all_rtp_amino_acids_covered():
    """Каждый аминокислотный остаток rtp либо проверяется, либо помечен с причиной."""
    missing = [n for n in RTP if n not in RESIDUES and n not in UNSUPPORTED
               and not (n[0] in 'NC' and n[1:] in RTP)]
    assert not missing, f'нет ни SMILES, ни причины: {missing}'


def _residue(parent, side):
    mon = Chem.AddHs(Chem.MolFromSmiles(f'N[C@@H]({side})C(=O)O'))
    tri = Chem.AddHs(Chem.MolFromSmiles(f'NCC(=O)N[C@@H]({side})C(=O)NCC(=O)O'))
    mon.SetProp('_Name', 'monomer')
    tri.SetProp('_Name', 'trimer')
    with contextlib.redirect_stdout(io.StringIO()):
        res = pt.find_residue(mon, tri, timeout=5)
        return pt.renumber_residue_atoms(res, ref_base_name=parent, timeout=5)


@pytest.mark.parametrize('resname', [_case(n) for n in RTP])
@pytest.mark.parametrize('with_parent', [True, False], ids=['parent', 'rule_only'])
def test_rtp_residue_names(resname, with_parent):
    parent, side = RESIDUES[resname]
    res = _residue(parent, side)
    parent_atoms = [a.GetIdx() for a in res.GetAtoms() if a.HasProp('ParentName')] if with_parent else None
    with contextlib.redirect_stdout(io.StringIO()):
        named = pt.rdkit_pdb_modification(Chem.Mol(res), resname=resname, resid=2, naming='greek',
                                          parent_atoms=parent_atoms)
    assert [a.GetProp('AtomName') for a in named.GetAtoms()] == RTP[resname]
