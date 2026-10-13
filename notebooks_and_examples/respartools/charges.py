"""Заряды: ограничения из rtp amber14sb, работа со списками зарядов."""
import os
from collections import defaultdict
import numpy as np





from .log import logged
from .utils import _PACKAGE_ROOT
        
DEFAULT_RTP = os.path.join(_PACKAGE_ROOT,
                           'amber14sb_parmbsc1_cufix.ff', 'aminoacids.rtp')
BACKBONE_CONSTRAINT_ATOMS = ('N', 'H', 'CA', 'HA', 'C', 'O')


def read_rtp_charges(resname, rtp_path=DEFAULT_RTP):
    """
    Заряды атомов остатка из секции [ atoms ] файла .rtp (GROMACS).
    Возвращает {имя атома: заряд} в порядке файла.
    """
    residues = read_rtp(rtp_path)[1]
    if resname not in residues or not residues[resname].get('atoms'):
        raise ValueError(f'Остаток {resname} не найден в {rtp_path}')
    return {name: q for name, _, q, _ in residues[resname]['atoms']}


@logged
def charge_constraints_from_rtp(mol, rtp_residue, atom_names=BACKBONE_CONSTRAINT_ATOMS,
                                rtp_path=DEFAULT_RTP):
    """
    Ограничения зарядов для Espaloma (charge(mol, constraints=...)): атомы с именами
    atom_names получают заряды одноимённых атомов остатка rtp_residue из силового поля.
    По умолчанию - атомы остова N H CA HA C O (как во всех ноутбуках лизинов): остов
    модифицированного остатка сохраняет заряды amber14sb, боковая цепь считается Espaloma.

    Аргументы:
        mol (Chem.Mol) - остаток с именами атомов (свойство 'AtomName', шаг 4)
        rtp_residue (str) - остаток в .rtp, чьи заряды берутся (например 'CYS', 'LYS')
        atom_names (list) - имена атомов с фиксированным зарядом
        rtp_path (str) - файл .rtp; по умолчанию aminoacids.rtp amber14sb из репозитория
    Возвращает:
        dict {индекс атома: заряд}
    """
    rtp = read_rtp_charges(rtp_residue, rtp_path)
    index = defaultdict(list)
    for atom in mol.GetAtoms():
        if atom.HasProp('AtomName'):
            index[atom.GetProp('AtomName').strip()].append(atom.GetIdx())
    not_in_rtp = [n for n in atom_names if n not in rtp]
    if not_in_rtp:
        raise ValueError(f'В {rtp_residue} ({rtp_path}) нет атомов {not_in_rtp}; есть: {list(rtp)}')
    problems = {n: index.get(n, []) for n in atom_names if len(index.get(n, [])) != 1}
    if problems:
        raise ValueError(f'Имена атомов должны встречаться в молекуле ровно один раз '
                         f'(имя: найденные индексы): {problems}')
    constraints = {index[n][0]: rtp[n] for n in atom_names}
    print(f'Заряды из {rtp_residue} (имя, индекс, заряд): '
          + ', '.join(f'{n} {index[n][0]} {rtp[n]:+.4f}' for n in atom_names))
    return constraints



## Работа с листом зарядов

@logged
def make_substructure_charge_list(pol_name, charge_array, match_dict, ):
    monomer_charge = [0] * (len(match_dict['substructure'][pol_name].GetAtoms()))
    # charge_array = charge_array.round(5)
    for monomer_index, polymer_index in enumerate(match_dict['mon_pol_matches'][pol_name].values()):
        monomer_charge[monomer_index] = charge_array[polymer_index]
    monomer_charge = np.array(monomer_charge).round(4)
    return monomer_charge

def group_charges(mol, charges):
    """
    Заряды групп: тяжёлый атом вместе со своими водородами (CH2, NH3 и т.п.). По ним удобно
    сравнивать схемы зарядов: AM1-BCC и RESP по-разному делят заряд внутри группы.
    Возвращает {имя тяжёлого атома (или индекс): заряд группы} в порядке атомов.
    """
    if len(charges) != mol.GetNumAtoms():
        raise ValueError(f'Зарядов {len(charges)}, атомов {mol.GetNumAtoms()}')
    groups = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        q = float(charges[atom.GetIdx()]) + sum(float(charges[n.GetIdx()]) for n in atom.GetNeighbors()
                                                 if n.GetAtomicNum() == 1)
        name = atom.GetProp('AtomName') if atom.HasProp('AtomName') else str(atom.GetIdx())
        groups[name] = q
    return groups


@logged
def compare_charges(mol, charge_sets, top=10):
    """
    Сравнивает наборы зарядов одного остатка ({'Espaloma': q1, 'RESP': q2, ...}) по группам
    (group_charges): печатает суммы, среднее и наибольшие расхождения групп между первым
    и остальными наборами. Возвращает {набор: {группа: заряд}}.
    """
    names = list(charge_sets)
    tables = {n: group_charges(mol, q) for n, q in charge_sets.items()}
    ref = names[0]
    for n in names:
        print(f'{n}: сумма {sum(float(x) for x in charge_sets[n]):+.4f}')
    for n in names[1:]:
        diff = {g: tables[n][g] - tables[ref][g] for g in tables[ref]}
        mean = sum(abs(d) for d in diff.values()) / len(diff)
        worst = sorted(diff, key=lambda g: -abs(diff[g]))[:top]
        print(f'{n} - {ref}: среднее |разница групп| {mean:.3f} e; наибольшие: '
              + ', '.join(f'{g} {tables[ref][g]:+.2f} -> {tables[n][g]:+.2f}' for g in worst))
    return tables


def set_charges(mol, charges, method, constraints=None):
    """
    Записывает заряды в атомы: свойство Charge<method> (например ChargeEspaloma, ChargeRESP;
    методов может быть больше), свойство молекулы ChargeMethod = method (заряды, которые пойдут
    в силовое поле), ChargeFixed = True у атомов из constraints ({индекс: заряд}).
    Меняет и возвращает mol.
    """
    if len(charges) != mol.GetNumAtoms():
        raise ValueError(f'Зарядов {len(charges)}, атомов {mol.GetNumAtoms()}')
    key = f'Charge{method}'
    for atom, q in zip(mol.GetAtoms(), charges):
        atom.SetDoubleProp(key, float(q))
        if constraints is not None:
            atom.SetBoolProp('ChargeFixed', atom.GetIdx() in constraints)
    mol.SetProp('ChargeMethod', method)
    return mol


from .forcefield import read_rtp  # noqa: E402
