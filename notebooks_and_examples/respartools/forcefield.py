"""Файлы силового поля GROMACS: rtp (read_rtp, write_rtp), itp (read_itp), acpype (run_acpype), hdb, atp, r2b."""
import os
import sys
import re
import glob
from collections import defaultdict

import subprocess
import shutil


from rdkit import Chem


from .log import logged
from .utils import _PACKAGE_ROOT, print_red


# Дубликат: ниже в модуле есть второе определение check_duplicate_atom_names, оно и действует
# (в Python работает последнее определение). Этот вариант закомментирован, чтобы не путаться.
# def check_duplicate_atom_names(mol):
#     """Проверяет уникальность имен атомов"""
#     name_to_indices = {}
#     for atom in mol.GetAtoms():
#         if atom.HasProp("AtomName"):
#             name = atom.GetProp("AtomName")
#             idx = atom.GetIdx()
#             if name not in name_to_indices:
#                 name_to_indices[name] = []
#             name_to_indices[name].append(idx)
#
#     duplicates = {name: indices for name, indices in name_to_indices.items() if len(indices) > 1}
#
#     if duplicates:
#         print("⚠️ Обнаружены дубликаты имен:")
#         for name, indices in duplicates.items():
#             print(f"  {name}: индексы {indices}")
#         return False
#
#     print("✅ Все имена атомов уникальны")
#     return True

def remove_extra_H(top, extra_pattern_H = 'HW'):
    atom_to_remove = [atom for atom in top.atoms if extra_pattern_H in atom.name]
    print(f"Найденные атомы для удаления: {atom_to_remove}")
    # Удаление атомов из списка атомов
    for atom in atom_to_remove[::-1]:
        top.atoms.remove(atom)

    # Ручное удаление связей, углов и диэдральных углов
    bonds_to_remove = [
        bond for bond in top.bonds if any(atom in atom_to_remove for atom in (bond.atom1, bond.atom2))
    ]
    angles_to_remove = [
        angle for angle in top.angles if any(atom in atom_to_remove for atom in (angle.atom1, angle.atom2, angle.atom3))
    ]
    dihedrals_to_remove = [
        dihedral for dihedral in top.dihedrals if any(atom in atom_to_remove for atom in (dihedral.atom1, dihedral.atom2, dihedral.atom3, dihedral.atom4))
    ]

    # Удаление связей
    for bond in bonds_to_remove:
        top.bonds.remove(bond)

    # Удаление углов
    for angle in angles_to_remove:
        top.angles.remove(angle)

    # Удаление диэдральных углов
    for dihedral in dihedrals_to_remove:
        top.dihedrals.remove(dihedral)

# Сохранение измененного файла топологии
# top.write(f"Acpype_data/{acpype_name}.acpype/{acpype_name}_GMX_cleaned.itp")
@logged
def hdb_generator(mol_chem, resname='MOD', resid=1, segid='A'):
    """
    Принимает на вход:
     - mol путь к файлу или rdkit.Chem объект
    Модифицирует имена атомов в молекуле по правилам аминокислот:
    - N, H, CA, C, O имеют стандартные имена
    - Боковые атомы получают буквенные метки по греческому алфавиту
    - Протоны наследуют имя родительского атома и получают числовой суффикс
    """
    def find_common_prefix(strings):
        if not strings:
            return ""

        first = strings[0]
        for i in range(len(first), 0, -1):
            prefix = first[:i]
            if all(s.startswith(prefix) for s in strings[1:]):
                return prefix
        return ""

    hdb = ['1	1	H	N	-C	CA	\n',
           '1	5	HA	CA	N	CB	C\n']
    
    for atom in mol_chem.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        current_idx = atom.GetIdx()
        current_atom_name = atom.GetProp('AtomName')
        protons = []
        visited_atoms = {current_atom_name: current_idx}
        for n_atom in atom.GetNeighbors():
            atom_name = n_atom.GetProp('AtomName')
            atom_idx = n_atom.GetIdx()
            if n_atom.GetSymbol() != "H":
                visited_atoms[atom_name] = atom_idx
            elif n_atom.GetProp('AtomName') not in ['H','HA', 'HN']:
                protons.append(atom_name)
        if not protons:  # У атома нет протонов идем к слежующему атому
            continue
        # Если H на атоме с валентностью 2
        if len(visited_atoms) == 2:  # Будет только 1 тяжелый сосед
            atom_2 = mol_chem.GetAtomWithIdx(min(visited_atoms.values()))
            # # Берем индекс соседа и находим его соседа с наименьшим индексом
            atom_3 = min(filter(lambda n: n.GetSymbol() != "H" and
                                n.GetIdx() not in visited_atoms.values(), atom_2.GetNeighbors()),
                         key=lambda n: n.GetIdx(), default=None)
            if atom_3:
                atom_3_name = atom_3.GetProp('AtomName')
                atom_3_idx = atom_3.GetIdx()
                visited_atoms[atom_3_name] = atom_3_idx

        hyb = atom.GetHybridization()
        n_hydrogens = len(protons)
        n_heavy_neighbors = int(atom.GetDegree()) - n_hydrogens
        if hyb == Chem.HybridizationType.SP3:
            if n_hydrogens == 1 and n_heavy_neighbors == 3:
                geom_n = 5  # sp3 углерод с 1 протоном
            elif n_hydrogens == 2 and n_heavy_neighbors == 2:
                geom_n = 6  # sp3 углерод с 2 протоном
            elif n_hydrogens == 3 and n_heavy_neighbors == 1:
                geom_n = 4  # sp3 углерод с 3 протоном
            elif n_hydrogens == 1 and n_heavy_neighbors == 1:
                geom_n = 2  # sp3 углерод с 1 протоном
            else:
                print(f"Не учтенный вариант {current_atom_name}:",
                     f"{hyb}, {n_hydrogens}, {n_heavy_neighbors}", sep='\n')
                geom_n = '-'
        elif hyb == Chem.HybridizationType.SP2:
            if n_hydrogens == 1 and n_heavy_neighbors == 2:
                geom_n = 1
            elif n_hydrogens == 2 and n_heavy_neighbors == 1:
                geom_n = 3
            elif n_hydrogens == 1 and n_heavy_neighbors == 1:
                geom_n = 2
            else:
                print(f"Не учтенный вариант {current_atom_name}:",
                     f"{hyb}, {n_hydrogens}, {n_heavy_neighbors}", sep='\n')
                geom_n = '-'
        else:
            if protons:
                print("Существуют не описанные протоны:")
                print(atom.GetProp("AtomName"), hyb)
                print(protons)
                geom_n = '-'
            
        if protons: # ['H'] ['HB1', 'HB2'] ['HC1', 'HC2', 'HC3']
            # print(protons)
            n_H = str(len(protons))
            H_name = find_common_prefix(protons)
            if H_name not in ['H', 'HA', 'HN']:
                hdb.append("{}\t{}\t{}\t{}\n".format(n_H, geom_n, H_name, "\t".join(visited_atoms)))
                # ['CB', 'CA', 'CG'] ['CK', 'CI'] ['CA', 'N', 'C', 'CB']
    # print(heavy_atoms)
    hdb = [f'{resname}\t{len(hdb)}\n'] + hdb
    return hdb
    #     print(neighbors, sep = '\n')
    # print(heavy_atoms)

@logged
def check_atomtypes(top, path_to_atp = '', param_folder = ''):
    exist_types = []
    with open(path_to_atp, 'r') as atomtypes:
        file = atomtypes.readlines()
    for line in file:
        exist_types.append(line.split()[0])

    atom_types = {str(atom.atom_type) : atom.mass for atom in top}
    add_atom_types = []       
    for atom_type, atom_mass in atom_types.items():
        if atom_type not in exist_types:
            add_atom_types.append('%-2s%24.5f\n' % (atom_type, atom_mass))
    
    if param_folder:
        os.makedirs(param_folder, exist_ok=True)
    save_path = f'{param_folder}/atomtypes.atp'
    
    if add_atom_types:
        add_atom_types.extend(file)
        # print(f'In {path_to_atp}\nAdd line(-s):\n{add_atom_types}')
        
        with open (save_path, 'w') as atomtypes:
            atomtypes.writelines(add_atom_types)
        print(f'Добавлены недостающие типы атомов в {save_path}')
    else:
        with open (save_path, 'w') as atomtypes:
            atomtypes.writelines(file)
        print(f'Все используемые типы атомов указаны в {path_to_atp}')

        
@logged
def make_r2b(path_to_r2b = '', reference_aa = '', add_aa_name = '', param_folder = '', out=False):
    with open (path_to_r2b, 'r') as r2b:
        r2b_list = r2b.readlines()
    for i, r2b_line in enumerate(r2b_list):
        rtp_str = r2b_line.upper()
        if reference_aa.upper() == rtp_str.split()[0]:
            print(f'Replese old srt:\n{rtp_str}')
            add_rtp_line = rtp_str.replace('-', add_aa_name.upper(),1)
            print(f'To new str:\n{add_rtp_line}')
            r2b_list[i] = add_rtp_line

    save_path = f'{param_folder}/aminoacids.r2b'
    with open(save_path, 'w') as f:
        f.writelines(r2b_list)
    print(f'Save in  {save_path}')


# -----------------------------------------------------------------------------
# rtp: один парсер и один писатель на весь пакет
# -----------------------------------------------------------------------------
RTP_SECTIONS = ('atoms', 'bonds', 'angles', 'dihedrals', 'impropers', 'exclusions', 'cmap')
RTP_BONDEDTYPES = [1, 1, 9, 4, 1, 3, 1, 0]   # amber14sb: bonds angles dihedrals impropers all_dihedrals nrexcl HH14 RemoveDih


def read_rtp(path):
    """
    Файл .rtp GROMACS. Комментарии (после ;) отбрасываются.
    Возвращает (bondedtypes - список чисел или None,
                {остаток: {'atoms': [(имя, тип, заряд, группа)], 'bonds': [[поля]], 'angles': ..., ...}})
    В секциях кроме atoms строка хранится списком полей как есть (имена атомов, затем параметры).
    """
    bondedtypes, residues, current, section = None, {}, None, None
    with open(path) as file:
        for line in file:
            text = line.split(';')[0].strip()
            if not text:
                continue
            header = re.fullmatch(r'\[\s*(\S+)\s*\]', text)
            if header:
                name = header.group(1)
                if name == 'bondedtypes':
                    current, section = None, 'bondedtypes'
                elif name in RTP_SECTIONS:
                    if current is None:
                        raise ValueError(f'{path}: секция [ {name} ] вне остатка')
                    section = name
                    current.setdefault(name, [])
                else:
                    current, section = residues.setdefault(name, {}), None
                continue
            parts = text.split()
            if section == 'bondedtypes':
                bondedtypes = [int(x) for x in parts]
            elif section == 'atoms':
                current['atoms'].append((parts[0], parts[1], float(parts[2]), int(parts[3])))
            elif section is not None:
                current[section].append(parts)
    return bondedtypes, residues


def write_rtp(path, residues, bondedtypes=RTP_BONDEDTYPES):
    """
    Пишет .rtp: заголовок [ bondedtypes ] и остатки {имя: {'atoms': [(имя, тип, заряд, группа)], ...}}
    (формат read_rtp). Заряды пишутся без округления.
    """
    lines = ['[ bondedtypes ]\n',
             '; bonds  angles  dihedrals  impropers all_dihedrals nrexcl HH14 RemoveDih\n',
             ' ' + ' '.join(f'{x:6d}' for x in bondedtypes) + '\n']
    for resname, res in residues.items():
        lines.append(f'\n[ {resname} ]\n')
        for section in RTP_SECTIONS:
            if section not in res:
                continue
            lines.append(f' [ {section} ]\n')
            if section == 'atoms':
                lines += [f'{n:>6} {t:>6} {q!r:>12} {g:5d}\n' for n, t, q, g in res['atoms']]
            else:
                lines += [''.join(f'{x:>6} ' for x in row).rstrip() + '\n' for row in res[section]]
    with open(path, 'w') as file:
        file.writelines(lines)
    return path


# -----------------------------------------------------------------------------
# acpype (шаг 2) и чтение itp
# -----------------------------------------------------------------------------
ITP_SECTIONS = ('atoms', 'bonds', 'pairs', 'angles', 'dihedrals')


def read_itp(path):
    """
    Секции молекулы из .itp GROMACS (acpype): {'atoms': [[поля]], 'bonds': ..., 'dihedrals': ...}.
    Строка - список полей без комментария; номера атомов - как в файле (с 1). Секций
    [ dihedrals ] в acpype две (собственные функция 9, несобственные функция 4) - они идут
    в один список, функция - в 5-м поле.
    """
    sections, current = {}, None
    with open(path) as file:
        for line in file:
            text = line.split(';')[0].strip()
            if not text:
                continue
            header = re.fullmatch(r'\[\s*(\S+)\s*\]', text)
            if header:
                current = header.group(1) if header.group(1) in ITP_SECTIONS else None
                if current:
                    sections.setdefault(current, [])
                continue
            if current:
                sections[current].append(text.split())
    return sections


def _acpype_bin(acpype_bin=None):
    """acpype: переданный путь, из PATH или из окружения topmol2 рядом с текущим."""
    if acpype_bin:
        return acpype_bin
    found = shutil.which('acpype')
    if found:
        return found
    envs = os.path.dirname(os.path.dirname(os.path.dirname(sys.executable)))
    candidate = os.path.join(envs, 'topmol2', 'bin', 'acpype')
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError('Нет acpype: окружение topmol2 (acpype, ambertools) или передайте acpype_bin.')


@logged
def run_acpype(mol, workdir, atom_type='amber', charge_type='gas', acpype_bin=None, random_seed=42):
    """
    Шаг 2: топология acpype (типы атомов и параметры связей, углов, диэдралов) для остатка.
    acpype не принимает атомы с оборванной валентностью, поэтому, если в mol ещё нет
    водородов шага 4.1 (свойство атома AddedH), они добавляются здесь
    (add_protons_and_renumber_H) - только для acpype, в mol их нет. Вход acpype - MOL-файл
    прямо из RDKit (порядки связей и 3D-координаты, без obabel), заряд - формальный заряд
    молекулы (не руками). Порядок атомов itp сверяется с молекулой по элементам (по массе).

    Записывает в атомы mol 'AcpypeType', в молекулу 'AcpypeItp' (путь к itp; атомы itp с
    номером больше числа атомов остатка - добавленные водороды). Заряды acpype (charge_type)
    не используются: заряды - шаг 5/6. Меняет и возвращает mol.
    """
    name = mol.GetProp('ResName') if mol.HasProp('ResName') else \
        mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName().strip()
    os.makedirs(workdir, exist_ok=True)
    has_added = any(a.HasProp('AddedH') and a.GetBoolProp('AddedH') for a in mol.GetAtoms())
    if has_added:
        work = Chem.Mol(mol)
    else:
        info = mol.GetAtomWithIdx(0).GetPDBResidueInfo()
        work = add_protons_and_renumber_H(mol, resname=name, resid=info.GetResidueNumber() if info else 1,
                                          segid=info.GetChainId() if info else 'A')
    if not work.GetNumConformers() or not work.GetConformer().Is3D():
        work = set_mol_coords(work, coords='3D', random_seed=random_seed)
    mol_path = os.path.join(workdir, f'{name}.mol')
    Chem.MolToMolFile(work, mol_path)
    net = Chem.GetFormalCharge(mol)
    cmd = [_acpype_bin(acpype_bin), '-i', f'{name}.mol', '-c', charge_type, '-a', atom_type,
           '-n', str(net), '-b', name, '-o', 'gmx']
    run = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    with open(os.path.join(workdir, 'acpype.log'), 'w') as log:
        log.write(run.stdout + run.stderr)
    itp = os.path.join(workdir, f'{name}.acpype', f'{name}_GMX.itp')
    if run.returncode != 0 or not os.path.exists(itp):
        tail = (run.stdout + run.stderr).strip().splitlines()[-3:]
        raise RuntimeError(f'acpype не построил топологию ({os.path.join(workdir, "acpype.log")}): '
                           + ' | '.join(tail))
    atoms = read_itp(itp)['atoms']
    table = Chem.GetPeriodicTable()
    if len(atoms) != work.GetNumAtoms():
        raise ValueError(f'В itp {len(atoms)} атомов, во входе acpype {work.GetNumAtoms()}')
    wrong = [(a.GetIdx(), a.GetSymbol(), row[7]) for a, row in zip(work.GetAtoms(), atoms)
             if abs(table.GetAtomicWeight(a.GetSymbol()) - float(row[7])) > 0.1]
    if wrong:
        raise ValueError(f'Порядок атомов itp не совпадает с молекулой (индекс, элемент, масса в itp): {wrong[:5]}')
    for atom, row in zip(mol.GetAtoms(), atoms):
        atom.SetProp('AcpypeType', row[1])
    mol.SetProp('AcpypeItp', os.path.abspath(itp))
    print(f'acpype ({atom_type}, заряд {net:+d}): {len(atoms)} атомов'
          + ('' if has_added else f' (из них {len(atoms) - mol.GetNumAtoms()} водорода только для acpype)')
          + f', типы записаны в AcpypeType; {itp}')
    return mol


# -----------------------------------------------------------------------------
# Шаг 3: остаток для силового поля (типы, rtp, hdb) и запись поля
# -----------------------------------------------------------------------------
def _element(name):
    return 'H' if name[0] == 'H' else name[0]

def hdb_entries(atoms, bonds):
    """
    Записи .hdb остатка по его графу (формат read_rtp): atoms [(имя, ...)], bonds [[a, b, ...]],
    внешние связи - '-C', '+N'. Тип добавления H - по числу водородов и числу соседей атома
    (правила GROMACS: 1 плоский, 2 OH/SH, 3 плоская NH2, 4 метил, 5 CH sp3, 6 CH2 sp3).
    Управляющие атомы - соседи в порядке атомов остатка (внешние - первыми).
    """
    order = {a[0]: i for i, a in enumerate(atoms)}
    nbrs = {a[0]: [] for a in atoms}
    for b in bonds:
        x, y = b[0], b[1]
        if x in nbrs:
            nbrs[x].append(y)
        if y in nbrs:
            nbrs[y].append(x)
    key = lambda n: (0, 0) if n[0] in '+-' else (1, order[n])
    entries = []
    for name, *_ in atoms:
        if _element(name) == 'H':
            continue
        hs = [n for n in nbrs[name] if n[0] not in '+-' and _element(n) == 'H']
        if not hs:
            continue
        heavy = sorted([n for n in nbrs[name] if n not in hs], key=key)
        total = len(hs) + len(heavy)
        n_h = len(hs)
        if n_h == 1:
            hname = hs[0]
        else:
            prefixes = {h[:-1] for h in hs}
            digits = sorted(h[-1] for h in hs)
            if len(prefixes) != 1 or digits != [str(i) for i in range(1, n_h + 1)]:
                raise ValueError(f'{name}: водороды {hs} не записать в hdb (нужны имена <префикс>1..{n_h})')
            hname = prefixes.pop()
        def second(j):   # сосед соседа j (не сам атом), первый по порядку
            cand = sorted([n for n in nbrs.get(j, []) if n != name and n[0] not in '+-' and _element(n) != 'H'], key=key)
            if not cand:
                raise ValueError(f'{name}: у {j} нет второго соседа для hdb')
            return cand[0]
        if n_h == 1 and total == 3:
            kind, ctrl = 1, heavy[:2]
        elif n_h == 1 and total == 4:
            kind, ctrl = 5, heavy[:3]
        elif n_h == 1 and total == 2:
            kind, ctrl = 2, [heavy[0], second(heavy[0])]
        elif n_h == 2 and total == 4:
            kind, ctrl = 6, heavy[:2]
        elif n_h == 2 and total == 3:
            kind, ctrl = 3, [heavy[0], second(heavy[0])]
        elif n_h == 3 and total == 4:
            kind, ctrl = 4, [heavy[0], second(heavy[0])]
        else:
            raise ValueError(f'{name}: {n_h} H при {total} соседях - нет типа hdb')
        entries.append([n_h, kind, hname, name] + ctrl)
    return entries


def read_ffbonded(path):
    """
    Какие связи, углы и диэдралы описаны в ffbonded.itp по типам атомов:
    {'bonds': {(t1, t2)}, 'angles': {(t1, t2, t3)}, 'propers': [(t1..t4)], 'impropers': [(t1..t4)]}
    (связи и углы - в обоих направлениях; в диэдралах возможен шаблон X).
    """
    out = {'bonds': set(), 'angles': set(), 'propers': [], 'impropers': []}
    section = None
    with open(path) as file:
        for line in file:
            text = line.split(';')[0].strip()
            if not text or text.startswith('#'):
                continue
            header = re.fullmatch(r'\[\s*(\S+)\s*\]', text)
            if header:
                section = header.group(1)
                continue
            p = text.split()
            if section in ('bondtypes', 'constrainttypes'):
                out['bonds'].update({tuple(p[:2]), tuple(p[1::-1])})
            elif section == 'angletypes':
                out['angles'].update({tuple(p[:3]), tuple(p[2::-1])})
            elif section == 'dihedraltypes':
                (out['impropers'] if p[4] == '4' else out['propers']).append(tuple(p[:4]))
    return out


def _dihedral_known(types, table):
    """Есть ли диэдрал с такими типами (прямо или в обратном порядке, X - любой тип)."""
    return any(all(x == 'X' or x == y for x, y in zip(t, cand))
               for t in table for cand in (types, types[::-1]))


def _env(atom):
    """Окружение атома молекулы: (элемент, соседей вместе с неявными H, явных H)."""
    return (atom.GetSymbol(), atom.GetTotalDegree(), sum(n.GetAtomicNum() == 1 for n in atom.GetNeighbors()))


def _rtp_env(res, name):
    """То же для атома остатка rtp (связь '-C N' - это и связь C со следующим остатком)."""
    nb = [b[1] if b[0] == name else b[0] for b in res['bonds'] if name in b[:2]]
    nb += ['+' + b[1] for b in res['bonds'] if b[0][0] == '-' and b[0][1:] == name]
    return (_element(name), len(nb), sum(_element(n) == 'H' for n in nb if n[0] not in '+-'))


@logged
def assign_types(mol, parent_res):
    """
    Шаг 3: тип атома для силового поля (свойство AtomType, источник - TypeSource).
    Атом родителя (ParentName) сохраняет тип родителя, если его окружение то же
    (элемент, число соседей, число H); водород - если окружение то же и у его тяжёлого
    атома, и у всех соседей того атома (иначе это H у изменённой группы, например HE1/HE2
    при ацилировании NZ: в лизине они HP рядом с N+). Остальные атомы - тип acpype
    (AcpypeType, run_acpype). Вместо списков исключений и родителя, вписанных руками.
        parent_res - остаток родителя из read_rtp (например read_rtp(...)[1]['LYS'])
    Меняет и возвращает mol.
    """
    ptype = {n: t for n, t, *_ in parent_res['atoms']}
    same = {}
    for a in mol.GetAtoms():
        pn = a.GetProp('ParentName') if a.HasProp('ParentName') else None
        same[a.GetIdx()] = pn in ptype and _env(a) == _rtp_env(parent_res, pn)
    for a in mol.GetAtoms():
        keep = same[a.GetIdx()]
        if keep and a.GetAtomicNum() == 1:
            heavy = a.GetNeighbors()[0]
            keep = same[heavy.GetIdx()] and all(same[n.GetIdx()] for n in heavy.GetNeighbors())
        a.SetProp('AtomType', ptype[a.GetProp('ParentName')] if keep else a.GetProp('AcpypeType'))
        a.SetProp('TypeSource', 'parent' if keep else 'acpype')
    changed = [f"{a.GetProp('AtomName')} {ptype[a.GetProp('ParentName')]}->{a.GetProp('AtomType')}"
               for a in mol.GetAtoms() if a.HasProp('ParentName') and a.GetProp('ParentName') in ptype
               and a.GetProp('TypeSource') == 'acpype']
    if changed:
        print('Атомы родителя с изменённым окружением - тип acpype: ' + ', '.join(changed))
    return mol


def residue_rtp(mol, ff_dir):
    """
    Остаток для rtp из молекулы (AtomName, AtomType, Charge<ChargeMethod>, AcpypeItp).
    Связи - все связи остатка и '-C N'. Явные параметры (из itp acpype, уже в единицах
    GROMACS) пишутся только для связей, углов и диэдралов, которых нет в ffbonded.itp
    поля по типам атомов; остальное GROMACS берёт из поля (углы и собственные диэдралы
    pdb2gmx строит сам). Несобственные диэдралы - acpype (центр - третий атом) и два
    стандартных у остова. Члены с водородами, добавленными только для acpype, отбрасываются.
    Возвращает (остаток в формате read_rtp, {секция: [типы с явными параметрами]}).
    """
    method = mol.GetProp('ChargeMethod')
    n = mol.GetNumAtoms()
    names = [a.GetProp('AtomName') for a in mol.GetAtoms()]
    types = [a.GetProp('AtomType') for a in mol.GetAtoms()]
    ff = read_ffbonded(os.path.join(ff_dir, 'ffbonded.itp'))
    itp = read_itp(mol.GetProp('AcpypeItp'))
    res = {'atoms': [(names[i], types[i], round(a.GetDoubleProp(f'Charge{method}'), 6), i + 1)
                     for i, a in enumerate(mol.GetAtoms())],
           'bonds': [], 'angles': [], 'dihedrals': [], 'impropers': []}
    explicit = defaultdict(set)
    for row in itp.get('bonds', []) + itp.get('angles', []) + itp.get('dihedrals', []):
        size = 2 if len(row) == 5 else 3 if len(row) == 6 else 4
        idx = [int(x) - 1 for x in row[:size]]
        if any(i >= n for i in idx):
            continue                     # водород, добавленный только для acpype
        t = tuple(types[i] for i in idx)
        nm = [names[i] for i in idx]
        if size == 2:
            known = t in ff['bonds']
            res['bonds'].append(nm if known else nm + row[3:5])
            section = 'bonds'
        elif size == 3:
            known = t in ff['angles']
            if not known:
                res['angles'].append(nm + row[4:6])
            section = 'angles'
        elif row[4] == '9':
            known = _dihedral_known(t, ff['propers'])
            if not known:
                res['dihedrals'].append(nm + row[5:8])
            section = 'dihedrals'
        else:
            known = _dihedral_known(t, ff['impropers'])
            res['impropers'].append(nm if known else nm + row[5:8])
            section = 'impropers'
        if not known:
            explicit[section].add(t)
    res['bonds'].append(['-C', 'N'])
    res['impropers'] += [['-C', 'CA', 'N', 'H'], ['CA', '+N', 'C', 'O']]
    res = {k: v for k, v in res.items() if v}
    return res, {k: sorted(v) for k, v in explicit.items()}


DEFAULT_FF = os.path.join(_PACKAGE_ROOT, 'amber14sb_parmbsc1_cufix.ff')


def _copy_forcefield(ff_base, ff_out):
    """
    Копия поля без файлов, которые не дают pdb2gmx работать: rtp с типами атомов, которых
    нет в atomtypes.atp (pdb2gmx читает все .rtp папки и останавливается на первом таком),
    блокноты и копии (*-Copy*). Пропущенные rtp печатаются.
    """
    atp = {line.split()[0] for line in open(os.path.join(ff_base, 'atomtypes.atp'))
           if line.split() and not line.startswith(';')}
    skipped = {}
    for path in glob.glob(os.path.join(ff_base, '*.rtp')):
        unknown = {t for res in read_rtp(path)[1].values() for _, t, _, _ in res.get('atoms', [])} - atp
        if unknown:
            skipped[os.path.basename(path)] = sorted(unknown)[:5]
    shutil.copytree(ff_base, ff_out, ignore=shutil.ignore_patterns('*.ipynb', '*-Copy*', *skipped))
    if skipped:
        print_red(f'⚠ Из {ff_base} не скопированы rtp с типами не из atomtypes.atp: {skipped}')


def _hdb_line(entry):
    return '\t'.join(str(x) for x in entry) + '\n'


@logged
def write_forcefield(mol, ff_out, ff_base=DEFAULT_FF):
    """
    Шаг 3: добавляет остаток в силовое поле ff_out (если папки нет - копия ff_base).
    Пишет <ResName>.rtp и <ResName>.hdb отдельными файлами (GROMACS читает все .rtp и .hdb
    папки поля), строку в residuetypes.dat и недостающие типы в atomtypes.atp.
    Повторный запуск заменяет запись остатка, а не дописывает её.
    mol - остаток после шагов 2, 5/6 (AtomName, AcpypeType, AcpypeItp, Charge<Метод>,
    ChargeMethod, Parent). Типы атомов назначаются здесь (assign_types).
    Возвращает (путь к rtp, {секция: типы с явными параметрами}).
    """
    if not os.path.isdir(ff_out):
        _copy_forcefield(ff_base, ff_out)
    resname = mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetResidueName().strip()
    parent = AMINO_ACIDS[mol.GetProp('Parent')][0].upper()
    parent_res = read_rtp(os.path.join(ff_base, 'aminoacids.rtp'))[1][parent]
    assign_types(mol, parent_res)
    res, explicit = residue_rtp(mol, ff_out)
    total = sum(q for _, _, q, _ in res['atoms'])
    if abs(total - round(total)) > 1e-3:
        print_red(f'⚠ Сумма зарядов {resname} {total:.5f} - не целое число.')
    rtp_path = write_rtp(os.path.join(ff_out, f'{resname}.rtp'), {resname: res},
                         read_rtp(os.path.join(ff_base, 'aminoacids.rtp'))[0])
    hdb = hdb_entries(res['atoms'], res['bonds'])
    with open(os.path.join(ff_out, f'{resname}.hdb'), 'w') as file:
        file.write(f'{resname}\t{len(hdb)}\n' + ''.join(_hdb_line(e) for e in hdb))
    # residuetypes.dat: строка остатка заменяется
    dat = os.path.join(ff_out, 'residuetypes.dat')
    lines = [line for line in open(dat) if line.split()[:1] != [resname]] if os.path.exists(dat) else []
    with open(dat, 'w') as file:
        file.writelines(lines + [f'{resname}\tProtein\n'])
    # типы: atomtypes.atp (масса) и наличие параметров Леннард-Джонса в ffnonbonded.itp
    atp = os.path.join(ff_out, 'atomtypes.atp')
    known = {line.split()[0] for line in open(atp) if line.split() and not line.startswith(';')}
    table = Chem.GetPeriodicTable()
    new = sorted({(a.GetProp('AtomType'), table.GetAtomicWeight(a.GetSymbol())) for a in mol.GetAtoms()
                  if a.GetProp('AtomType') not in known})
    if new:
        with open(atp, 'a') as file:
            file.writelines(f'{t:<6}{m:16.5f}\n' for t, m in new)
    nonbonded = {line.split()[0] for line in open(os.path.join(ff_out, 'ffnonbonded.itp'))
                 if line.split() and line.split()[0][0] not in ';[#'}
    missing = sorted({a.GetProp('AtomType') for a in mol.GetAtoms()} - nonbonded)
    if missing:
        print_red(f'⚠ Нет параметров Леннард-Джонса в ffnonbonded.itp для типов {missing}: grompp не сработает.')
    print(f'{resname}: {rtp_path}, {resname}.hdb ({len(hdb)} групп H); явные параметры: '
          + (', '.join(f'{k} {len(v)}' for k, v in explicit.items()) or 'нет'))
    return rtp_path, explicit


from .residue import set_mol_coords, add_protons_and_renumber_H  # noqa: E402
from .matching import AMINO_ACIDS  # noqa: E402
