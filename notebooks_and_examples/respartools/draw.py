"""Рисование молекул: draw_molecule, draw_mol_grid, draw_mon_pol_match."""
import os
import sys
import time
import re
import glob
import json
from pathlib import Path
from collections import Counter
from collections import defaultdict, deque
import numpy as np

import itertools
import threading
import subprocess

from decimal import Decimal as D

from rdkit.Chem.Draw import IPythonConsole
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Chem import rdDepictor
from rdkit.Geometry import Point3D
from typing import Callable

import datetime
import functools
import hashlib
import inspect
import platform
import traceback
import warnings
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

from .utils import print_red


_DRAW_BACKBONE = Chem.MolFromSmarts('[NX3][CX4]C=O')  # N-CA-C=O по тяжёлым атомам


def _orient_backbone_down(mol, prefer_atoms=None):
    """
    Поворачивает 2D-координаты молекулы (только для рисования): остов N-CA-C внизу, остальная
    молекула над ним, N слева, C справа - этапы и разные остатки на картинках ориентированы
    одинаково. Остов ищется по тяжёлым атомам (N-CA-C=O); если остовов несколько (тример),
    берётся тот, что целиком в prefer_atoms (подсвеченные атомы). Если остов не найден или
    выбор неоднозначен, координаты не меняются. Возвращает True, если поворот сделан.
    """
    matches = mol.GetSubstructMatches(_DRAW_BACKBONE)
    if prefer_atoms and len(matches) > 1:
        inside = [m for m in matches if set(m[:3]) <= set(prefer_atoms)]
        matches = inside or matches
    if len(matches) != 1 or mol.GetNumConformers() == 0:
        return False
    n_idx, ca_idx, c_idx = matches[0][:3]
    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions())[:, :2]
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    direction = pos[heavy].mean(axis=0) - pos[[n_idx, ca_idx, c_idx]].mean(axis=0)
    if np.linalg.norm(direction) < 1e-6:
        return False
    # поворот: направление «остов -> центр молекулы» смотрит вверх (+y на картинке RDKit)
    angle = np.pi / 2 - np.arctan2(direction[1], direction[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    new = (pos - pos[ca_idx]) @ rot.T
    if new[n_idx, 0] > new[c_idx, 0]:  # N слева, C справа
        new[:, 0] *= -1
    for idx, (x, y) in enumerate(new):
        conf.SetAtomPosition(idx, Point3D(float(x), float(y), 0.0))
    return True


def _prepare_draw_mol(mol, charge_list=None, prefer_coord_gfen=True, highlight_atoms=None,
                      show_names=False, show_index=True, orient_backbone=True):
    """
    Копия молекулы для рисования: подписи атомов «индекс:имя: заряд» (atomNote), 2D-раскладка,
    проверенные индексы подсветки. Общая часть draw_molecule и draw_mol_grid.
    Возвращает (молекула, атомы для подсветки, связи для подсветки).
    """
    draw_mol = Chem.Mol(mol)
    draw_mol.RemoveAllConformers()
    with_charges = charge_list is not None and len(charge_list) == draw_mol.GetNumAtoms()
    if charge_list is not None and not with_charges:
        print_red(f'⚠ Зарядов {len(charge_list)}, атомов {draw_mol.GetNumAtoms()}: '
                  'молекула нарисована без зарядов.')
    for atom in draw_mol.GetAtoms():
        atom.SetAtomMapNum(0)  # номера из SMILES не должны подменять индексы на картинке
        idx = atom.GetIdx()
        parts = [str(idx)] if show_index else []
        if show_names and atom.HasProp('AtomName') and atom.GetProp('AtomName').strip():
            parts.append(atom.GetProp('AtomName').strip())
        note = ':'.join(parts)
        if with_charges:
            note = f'{note}: {charge_list[idx]:.4f}' if note else f'{charge_list[idx]:.4f}'
        if note:
            atom.SetProp('atomNote', note)

    # CoordGen вызывается напрямую, без глобальной настройки RDKit (SetPreferCoordGen),
    # чтобы не менять раскладку других картинок в сессии
    if prefer_coord_gfen:
        from rdkit.Chem import rdCoordGen
        try:
            rdCoordGen.AddCoords(draw_mol)
        except Exception:
            rdDepictor.Compute2DCoords(draw_mol)
    else:
        rdDepictor.Compute2DCoords(draw_mol)

    hl_atoms = sorted(set(int(i) for i in highlight_atoms)) if highlight_atoms else []
    bad = [i for i in hl_atoms if not 0 <= i < draw_mol.GetNumAtoms()]
    if bad:
        raise ValueError(f'Индексы для подсветки вне молекулы ({draw_mol.GetNumAtoms()} атомов): {bad}')
    hl_set = set(hl_atoms)
    hl_bonds = [b.GetIdx() for b in draw_mol.GetBonds()
                if b.GetBeginAtomIdx() in hl_set and b.GetEndAtomIdx() in hl_set]
    if orient_backbone:
        _orient_backbone_down(draw_mol, hl_set)
    return draw_mol, hl_atoms, hl_bonds


def _auto_draw_size(n_atoms):
    width = int(min(1600, max(600, 110 * n_atoms ** 0.5)))
    return (width, int(width * 0.7))


def draw_molecule(mol, charge_list = None, size=None, prefer_coord_gfen = True,
                  highlight_atoms=None, show_names=False, show_index=True, orient_backbone=True):
    """
    Основная отрисовка молекулы. Рисуется копия: координаты, номера атомов и другие
    свойства исходной молекулы не меняются. Старое имя функции - draw_mol_with_atom_index
    (работает так же). Несколько молекул рядом - draw_mol_grid.

    Подпись атома собирается из включённых частей: «индекс:имя: заряд».

    Аргументы:
        mol (Chem.Mol) - молекула
        charge_list (list) - заряды атомов; добавляются к подписи, если длина совпадает
            с числом атомов (иначе функция предупреждает и рисует без зарядов)
        size (tuple) - размер картинки; по умолчанию подбирается по числу атомов
        prefer_coord_gfen (bool) - 2D-раскладка CoordGen (аккуратнее для больших молекул
            и молекул из PDB); False - стандартная раскладка RDKit
        highlight_atoms (list) - индексы атомов для подсветки (например, атомы родительского
            остатка: свойство 'ParentAtoms' после renumber_residue_atoms); связи между
            подсвеченными атомами тоже подсвечиваются
        show_names (bool) - добавить к подписи имя атома (свойство 'AtomName', например
            из PDB); у атомов без имени имя не пишется
        show_index (bool) - показывать настоящие индексы атомов (включая 0), по умолчанию True
        orient_backbone (bool) - повернуть картинку: остов аминокислоты (N-CA-C) внизу,
            N слева, C справа (см. _orient_backbone_down); молекулы без остова не поворачиваются
    Возвращает:
        PIL.Image
    """
    import io as _io
    from PIL import Image

    draw_mol, hl_atoms, hl_bonds = _prepare_draw_mol(mol, charge_list, prefer_coord_gfen,
                                                     highlight_atoms, show_names, show_index,
                                                     orient_backbone)
    if size is None:
        size = _auto_draw_size(draw_mol.GetNumAtoms())
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().annotationFontScale = 0.6
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, draw_mol, highlightAtoms=hl_atoms,
                                           highlightBonds=hl_bonds)
    except Exception:
        drawer = rdMolDraw2D.MolDraw2DCairo(*size)
        drawer.drawOptions().annotationFontScale = 0.6
        drawer.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(draw_mol, kekulize=False),
                            highlightAtoms=hl_atoms, highlightBonds=hl_bonds)
    drawer.FinishDrawing()
    return Image.open(_io.BytesIO(drawer.GetDrawingText()))


# старое имя: во всех ноутбуках до 2026-09 вызывается draw_mol_with_atom_index
draw_mol_with_atom_index = draw_molecule


def draw_mol_grid(mols, legends=None, highlight_atoms=None, show_index=True, show_names=False,
                  charge_lists=None, mols_per_row=3, sub_img_size=None, prefer_coord_gfen=True,
                  orient_backbone=True):
    """
    Несколько молекул на одной картинке (сетка), с теми же подписями и подсветкой, что
    у draw_molecule. Например, этапы подготовки остатка рядом: исходная молекула,
    перенумерованная, с именами атомов. Молекулы не меняются.

    Параметры highlight_atoms, show_index, show_names, charge_lists задаются одним значением
    для всех молекул или списком - по значению на каждую молекулу.

    Аргументы:
        mols (list или dict) - молекулы; dict {подпись: молекула} задаёт и подписи
        legends (list) - подписи под молекулами
        highlight_atoms (list) - список индексов для каждой молекулы (или None)
        show_index, show_names (bool или list) - см. draw_molecule
        charge_lists (list) - заряды для каждой молекулы (или None)
        mols_per_row (int) - молекул в строке
        sub_img_size (tuple) - размер одной ячейки; по умолчанию - по самой большой молекуле
        prefer_coord_gfen, orient_backbone (bool) - см. draw_molecule; при orient_backbone
            все молекулы с остовом ориентированы одинаково: остов внизу, N слева
    Возвращает:
        PIL.Image
    """
    import io as _io
    from PIL import Image

    if isinstance(mols, dict):
        if legends is None:
            legends = list(mols)
        mols = list(mols.values())
    mols = list(mols)
    n = len(mols)
    if n == 0:
        raise ValueError('draw_mol_grid: список молекул пуст')

    def per_mol(value, name, is_list_value=False):
        # одно значение на все молекулы или список по молекулам
        if value is None:
            return [None] * n
        if is_list_value:
            # список списков - по молекулам; иначе один список на все
            per = isinstance(value, (list, tuple)) and len(value) == n and all(
                v is None or isinstance(v, (list, tuple, np.ndarray)) for v in value)
            return list(value) if per else [value] * n
        if isinstance(value, (list, tuple)):
            if len(value) != n:
                raise ValueError(f'{name}: {len(value)} значений на {n} молекул')
            return list(value)
        return [value] * n

    highlights = per_mol(highlight_atoms, 'highlight_atoms', is_list_value=True)
    charges = per_mol(charge_lists, 'charge_lists', is_list_value=True)
    indices = per_mol(show_index, 'show_index')
    names = per_mol(show_names, 'show_names')
    if legends is not None and len(legends) != n:
        raise ValueError(f'legends: {len(legends)} подписей на {n} молекул')

    prepared = [_prepare_draw_mol(m, charges[i], prefer_coord_gfen, highlights[i],
                                  names[i], indices[i], orient_backbone) for i, m in enumerate(mols)]
    if sub_img_size is None:
        sub_img_size = _auto_draw_size(max(m.GetNumAtoms() for m in mols))
    n_col = max(1, min(mols_per_row, n))
    n_row = (n + n_col - 1) // n_col
    w, h = sub_img_size
    drawer = rdMolDraw2D.MolDraw2DCairo(w * n_col, h * n_row, w, h)
    drawer.drawOptions().annotationFontScale = 0.6
    drawer.DrawMolecules([p[0] for p in prepared],
                         highlightAtoms=[p[1] for p in prepared],
                         highlightBonds=[p[2] for p in prepared],
                         # место под подписи оставляет RDKit (заполнитель '_'), сами подписи рисует PIL
                         legends=['_'] * n if legends is not None else None)
    drawer.FinishDrawing()
    image = Image.open(_io.BytesIO(drawer.GetDrawingText())).convert('RGB')
    if legends is not None:
        # подписи рисует PIL: RDKit выводит кириллицу в подписях квадратами
        _draw_legends(image, [str(x) for x in legends], n_col, sub_img_size)
    return image


def _legend_font(size):
    """Шрифт с кириллицей для подписей: DejaVuSans из matplotlib, иначе шрифт PIL по умолчанию."""
    from PIL import ImageFont
    try:
        import matplotlib
        path = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf')
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _draw_legends(image, legends, n_col, sub_img_size):
    """Подписи по центру внизу каждой ячейки сетки."""
    from PIL import ImageDraw
    w, h = sub_img_size
    font = _legend_font(max(16, h // 25))
    draw = ImageDraw.Draw(image)
    for i, text in enumerate(legends):
        x0, y0 = (i % n_col) * w, (i // n_col) * h
        box = draw.textbbox((0, 0), text, font=font)
        # белый прямоугольник закрывает метку-заполнитель RDKit
        draw.rectangle([x0, y0 + h - h // 10, x0 + w - 1, y0 + h - 1], fill='white')
        draw.text((x0 + (w - (box[2] - box[0])) / 2, y0 + h - h // 20 - (box[3] - box[1]) / 2),
                  text, fill='black', font=font)


def draw_mon_pol_match(monomer_chem_dict, polymer_chem_dict={}, 
                       match_data = False, prefer_coord_gfen = False, 
                       add_atom_index = False, add_small_atom_index = False, show_any_monomer_matches = False, 
                       n_row = 1, useSVG = True, img_size = (500,300)):
    """
    Отображает сопоставление мономеров и полимеров в виде сетки изображений молекул.
    
    Аргументы:
        monomer_chem_dict (dict): Словарь мономеров.
        polymer_chem_dict (dict): Словарь полимеров.
        match_data (dict): Словарь данных сопоставления мономера с полимерами. По умолчанию False.
        add_atom_index (bool, optional): Добавлять ли номера атомов. По умолчанию False.
        add_small_atom_index (bool, optional): Использовать ли маленькие номера атомов. По умолчанию False.
        show_any_monomer_matches (bool, optional): Показать мономер с подструктурой для каждого полимера. По умолчанию False.
        n_row (int, optional): Количество молекул в строке. По умолчанию = кол-ву полимеров
        useSVG (bool, optional): Использовать ли SVG для изображения. По умолчанию True.
        img_size (tuple, optional): Размер изображения. По умолчанию (500, 300).
    
    Возвращает:
        Drawing: Изображение молекул в виде сетки.
    """
    n_mon, n_pol = len(monomer_chem_dict), len(polymer_chem_dict)
    if n_mon > 1 and n_pol > 1 and n_mon != n_pol:
        raise ValueError("Невозможно отобразить сопоставление для N:M (N,M > 1)")

    # Инициализация базовых структур
    mon_items = list(monomer_chem_dict.items())
    pol_items = list(polymer_chem_dict.items())
    
    
    chem_list = []
    legend_list = []
    highlight = []

    # Обработка случаев с сопоставлением
    if match_data:
        # 1:1 или N:N
        if n_mon == n_pol:
            pairs = zip(monomer_chem_dict.items(), polymer_chem_dict.items())
        # 1:N
        elif n_mon == 1:
            pairs = itertools.product(monomer_chem_dict.items(), polymer_chem_dict.items())
        # N:1
        elif n_pol == 1:
            pairs = itertools.product(monomer_chem_dict.items(), polymer_chem_dict.items())
        
        
        for (m_name, m_chem), (p_name, p_chem) in pairs:
            chem_list.extend([m_chem, p_chem])
            legend_list.extend([f"Monomer: {m_name}", f"Polymer: {p_name}"])
            
            # Обработка highlight атомов
            if match_data['mon_pol_matches'].get(m_name):
                match_info = match_data['mon_pol_matches'][m_name]
                if isinstance(match_info, dict):  # Для случая 1:1
                    highlight.extend([list(match_info.keys()), list(match_info.values())])
                else:  # Для случая 1:N или N:1
                    highlight.extend([[], []])
            elif match_data['mon_pol_matches'].get(p_name):
                match_info = match_data['mon_pol_matches'][p_name]
                highlight.extend([list(match_info.keys()), list(match_info.values())])
            else:
                raise ValueError(f"The match_data does not contain information about {m_name} or {p_name} matching.")
                
    # Без сопоставления
    else:
        for m_name, m_chem in monomer_chem_dict.items():
            chem_list.append(m_chem)
            legend_list.append(f"Monomer: {m_name}")
        for p_name, p_chem in polymer_chem_dict.items():
            chem_list.append(p_chem)
            legend_list.append(f"Polymer: {p_name}")
        
        # Балансировка списков для сетки
    # max_pairs_per_row = n_row if n_row > 0 else 4
    # mols_per_row = min(2 * max_pairs_per_row, len(chem_list))

    # Конфигурация отображения
    IPythonConsole.drawOptions.addAtomIndices = add_small_atom_index
    rdDepictor.SetPreferCoordGen(prefer_coord_gfen)
    
    # Генерация изображения
    drawing = Draw.MolsToGridImage(
        mols=chem_list,
        legends=legend_list,
        highlightAtomLists=highlight if match_data else None,
        molsPerRow=n_row,
        useSVG=useSVG,
        subImgSize=img_size
    )

    IPythonConsole.drawOptions.addAtomIndices = False
    rdDepictor.SetPreferCoordGen(False)
    return drawing
