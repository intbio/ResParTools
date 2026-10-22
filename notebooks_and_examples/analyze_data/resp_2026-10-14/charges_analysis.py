"""
Сравнение зарядов: новый RESP (кластер, psi4 1.6.1 + geomeTRIC, 2026-10-14) против старого RESP
(data/partial_charges.json) и против Espaloma (data/AI_chrges_*.json).
Атомы старого файла сопоставляются с остатком шага 4 по графу (элементы + связи, без порядков),
среди эквивалентных сопоставлений берётся ближайшее по зарядам (CH2/CH3 водороды, метилы).
Запуск: conda run -n darwin_ec python charges_analysis.py <папка вывода>
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.algorithms import isomorphism
from rdkit import Chem

ROOT = '/home/nikolay/Documents/VS_code/ResParTools/notebooks_and_examples'
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

RU = {'Lysine_1M': 'Монометилирование', 'Lysine_2M': 'Диметилирование', 'Lysine_3M': 'Триметилирование',
      'Lysine_1M_ACC': 'Метилацилирование', 'Lysine_ACC': 'Ацетилирование', 'Lysine_lac': 'Лактилирование',
      'Lysine_Cro': 'Кротонилирование', 'Lysine_prop': 'Пропионилирование', 'Lysine_Butyryl': 'Бутирилирование',
      'Lysine_Formyl': 'Формилирование', 'Lysine_Malonyl': 'Малонилирование', 'Lysine_Benzoyl': 'Бензоилирование'}
ORDER = list(RU)
BACKBONE = {'N', 'H', 'CA', 'HA', 'C', 'O'}
GREEK = {'B': 'β (CB)', 'G': 'γ (CG)', 'D': 'δ (CD)', 'E': 'ε (CE)', 'Z': 'ζ (NZ)'}
GROUPS = ['остов', 'β (CB)', 'γ (CG)', 'δ (CD)', 'ε (CE)', 'ζ (NZ)', 'модификация']


def graph(mol):
    g = nx.Graph()
    for a in mol.GetAtoms():
        g.add_node(a.GetIdx(), el=a.GetSymbol())
    for b in mol.GetBonds():
        g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return g


def best_mapping(g_new, g_old, q_new, q_old, limit=200000):
    """new idx -> old idx: изоморфизм графов с наименьшей суммой |q_new - q_old|."""
    gm = isomorphism.GraphMatcher(g_new, g_old, node_match=lambda a, b: a['el'] == b['el'])
    best, best_cost = None, np.inf
    for k, m in enumerate(gm.isomorphisms_iter()):
        cost = sum(abs(q_new[i] - q_old[j]) for i, j in m.items())
        if cost < best_cost:
            best, best_cost = m, cost
        if k >= limit:
            print(f'  ⚠ перебор сопоставлений остановлен на {limit}')
            break
    return best


def group_of(mol, names):
    """Группа атома: остов / β..ζ по греческой букве тяжёлого атома / модификация; H - по своему тяжёлому."""
    heavy_group = {}
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'H':
            continue
        n = names[a.GetIdx()]
        heavy_group[a.GetIdx()] = 'остов' if n in BACKBONE else GREEK.get(n[1] if len(n) > 1 else '', 'модификация')
    out = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == 'H':
            nb = a.GetNeighbors()[0].GetIdx()
            out.append('остов' if names[a.GetIdx()] in BACKBONE else heavy_group[nb])
        else:
            out.append(heavy_group[a.GetIdx()])
    return out


rows = []
for resp_file in sorted(glob.glob(f'{ROOT}/*/data/RESP_chrges_*.json')):
    folder = resp_file.split('/')[-3]
    mon = os.path.basename(resp_file)[len('RESP_chrges_'):-5]
    resp = json.load(open(resp_file))
    names, q_new = resp['atom_names'], np.array(resp['charges'])
    ai = json.load(open(glob.glob(f'{ROOT}/{folder}/data/AI_chrges_*.json')[0]))
    q_ai = np.array(ai['charges'] if isinstance(ai, dict) else ai)
    if isinstance(ai, dict):
        assert ai['atom_names'] == names, folder
    mol = Chem.MolFromPDBFile(f'{ROOT}/{folder}/molecules/substructure/4_{mon}.pdb', removeHs=False,
                              sanitize=False, proximityBonding=False)
    pdb_names = [a.GetPDBResidueInfo().GetName().strip() for a in mol.GetAtoms()]
    assert pdb_names == names, f'{folder}: порядок 4_{mon}.pdb не совпадает с RESP'
    elems = [a.GetSymbol() for a in mol.GetAtoms()]
    groups = group_of(mol, names)
    q_old = np.full(len(names), np.nan)
    old_file = f'{ROOT}/{folder}/data/partial_charges.json'
    if os.path.exists(old_file):
        old = json.load(open(old_file))
        smi = [f for f in glob.glob(f'{ROOT}/{folder}/molecules/substructure/*.smiles')
               if not os.path.basename(f)[:2] in ('3_', '4_')][0]
        p = Chem.SmilesParserParams()
        p.removeHs, p.sanitize = False, False
        old_mol = Chem.MolFromSmiles(open(smi).read().strip(), p)
        assert old_mol.GetNumAtoms() == len(old), f'{folder}: {old_mol.GetNumAtoms()} атомов в SMILES, {len(old)} зарядов'
        m = best_mapping(graph(mol), graph(old_mol), q_new, old)
        assert m is not None, f'{folder}: графы старого и нового остатка не совпадают'
        q_old = np.array([old[m[i]] for i in range(len(names))])
    for i, n in enumerate(names):
        rows.append({'Residue': folder, 'Residue_RU': RU[folder], 'atom': n, 'element': elems[i], 'group': groups[i],
                     'RESP_new': q_new[i], 'RESP_old': q_old[i], 'Espaloma': q_ai[i]})

df = pd.DataFrame(rows)
df['new_minus_old'] = df.RESP_new - df.RESP_old
df['espaloma_minus_new'] = df.Espaloma - df.RESP_new
df['espaloma_minus_old'] = df.Espaloma - df.RESP_old
df.round(4).to_csv(f'{OUT}/charges_per_atom.csv', index=False)

# суммы: проверка
tot = df.groupby('Residue')[['RESP_new', 'RESP_old', 'Espaloma']].sum(min_count=1).round(4).reindex(ORDER)
print('Суммарный заряд остатка:\n', tot.to_string(), '\n')

# по атомам: старый vs новый RESP
side = df[df.group != 'остов']
stat = side.groupby('Residue').apply(lambda x: pd.Series({
    'атомов (без остова)': len(x),
    'MAE new-old': x.new_minus_old.abs().mean(), 'max |new-old|': x.new_minus_old.abs().max(),
    'атом max': x.loc[x.new_minus_old.abs().idxmax(), 'atom'] if x.new_minus_old.notna().any() else '',
    'MAE Esp-new': x.espaloma_minus_new.abs().mean(), 'MAE Esp-old': x.espaloma_minus_old.abs().mean()})).reindex(ORDER)
stat.round(3).to_csv(f'{OUT}/stat_per_residue.csv')
print('Боковая цепь (остов фиксирован и одинаков во всех трёх):\n', stat.round(3).to_string(), '\n')

# по группам: суммарный заряд группы
grp = df.groupby(['Residue', 'group'])[['RESP_new', 'RESP_old', 'Espaloma']].sum(min_count=1).reset_index()
grp['group'] = pd.Categorical(grp.group, GROUPS, ordered=True)
grp = grp.sort_values(['Residue', 'group'])
grp.round(3).to_csv(f'{OUT}/charges_per_group.csv', index=False)
wide = grp.pivot(index='Residue', columns='group', values='RESP_new').reindex(ORDER)[GROUPS]
wide_old = grp.pivot(index='Residue', columns='group', values='RESP_old').reindex(ORDER)[GROUPS]
print('Заряд групп, новый RESP:\n', wide.round(3).to_string(), '\n')
print('Заряд групп, новый - старый RESP:\n', (wide - wide_old).round(3).to_string(), '\n')
wide_ai = grp.pivot(index='Residue', columns='group', values='Espaloma').reindex(ORDER)[GROUPS]
print('Заряд групп, Espaloma - новый RESP:\n', (wide_ai - wide).round(3).to_string(), '\n')
head = grp[grp.group.isin(['ζ (NZ)', 'модификация'])].groupby('Residue')[['RESP_old', 'RESP_new', 'Espaloma']].sum(min_count=1).reindex(ORDER)
print('Головная группа (NZ + модификация):\n', head.round(3).to_string(), '\n')

# по элементам (по всем атомам остатка - как в старой диаграмме base_analysis.ipynb)
sns.set_style('whitegrid')
colors = {'O': '#e41a1c', 'N': '#377eb8', 'C': '#4D4D4D', 'H': '#A6A6A6'}
names_ru = {'O': 'Кислород', 'N': 'Азот', 'C': 'Углерод', 'H': 'Водород'}
order_ru = [RU[r] for r in ORDER]


def swarm(col, title, fname, subset=None):
    d = df if subset is None else df[subset]
    mae = d.dropna(subset=[col]).groupby(['Residue_RU', 'element'])[col].apply(lambda x: x.abs().mean()).reset_index()
    mae.columns = ['Residue_RU', 'Atom', 'MAE']
    fig, ax = plt.subplots(figsize=(12, 9), constrained_layout=True)
    sns.swarmplot(data=mae, x='MAE', y='Residue_RU', hue='Atom', hue_order=['O', 'N', 'C', 'H'], palette=colors,
                  size=14, edgecolor='black', linewidth=0.8,
                  order=[o for o in order_ru if o in set(mae.Residue_RU)], ax=ax)
    ax.axvspan(0, 0.12, alpha=0.15, color='limegreen', zorder=0)
    ax.axvspan(0.12, 1, alpha=0.15, color='crimson', zorder=0)
    ax.axvline(0.12, color='black', linestyle='--', linewidth=2.5, alpha=0.8)
    ax.set_xlim(-0.02, 0.65)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel('MAE, атомные единицы заряда ($e$)', fontsize=13)
    ax.set_ylabel('Тип модификации лизина', fontsize=13)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[a], markersize=14,
                          markeredgecolor='black', label=names_ru[a]) for a in ['O', 'N', 'C', 'H']]
    ax.legend(handles=handles, title='Тип атома', loc='upper right', frameon=True)
    fig.savefig(f'{OUT}/{fname}', dpi=200, facecolor='white')
    plt.close(fig)
    return mae


m1 = swarm('espaloma_minus_new', 'MAE зарядов Espaloma Charge относительно RESP (новый, geomeTRIC)', 'MAE_Espaloma_vs_RESP_new.png')
m2 = swarm('espaloma_minus_old', 'MAE зарядов Espaloma Charge относительно старого RESP', 'MAE_Espaloma_vs_RESP_old.png')
m3 = swarm('new_minus_old', 'MAE: новый RESP относительно старого RESP', 'MAE_RESP_new_vs_old.png')
m4 = swarm('espaloma_minus_new', 'MAE Espaloma относительно нового RESP, только боковая цепь (без остова)',
           'MAE_Espaloma_vs_RESP_new_sidechain.png', subset=df.group != 'остов')
for name, m in [('Espaloma - RESP новый', m1), ('Espaloma - RESP старый', m2), ('RESP новый - старый', m3),
                ('Espaloma - RESP новый, без остова', m4)]:
    print(f'MAE по элементам, {name}:\n', m.pivot(index='Residue_RU', columns='Atom', values='MAE')
          .reindex([o for o in order_ru if o in set(m.Residue_RU)])[['O', 'N', 'C', 'H']].round(3).to_string(), '\n')

# парный график по атомам: старый против нового RESP
fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
d = side.dropna(subset=['RESP_old'])
for el in ['C', 'H', 'N', 'O']:
    x = d[d.element == el]
    ax.scatter(x.RESP_old, x.RESP_new, c=colors[el], edgecolor='black', s=40, label=names_ru[el])
lim = [-1.1, 1.1]
ax.plot(lim, lim, 'k--', lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel('старый RESP, e'); ax.set_ylabel('новый RESP, e')
ax.set_title('Заряды атомов боковой цепи: старый и новый RESP')
ax.legend()
fig.savefig(f'{OUT}/RESP_new_vs_old_atoms.png', dpi=200, facecolor='white')
plt.close(fig)

# крупнейшие расхождения по атомам
top = side.dropna(subset=['new_minus_old']).reindex(side.new_minus_old.abs().sort_values(ascending=False).index).head(15)
print('Крупнейшие |новый - старый RESP|:\n',
      top[['Residue', 'atom', 'group', 'RESP_old', 'RESP_new', 'Espaloma', 'new_minus_old']].round(3).to_string(index=False))
