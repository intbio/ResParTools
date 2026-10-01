"""Функции для совместимости со старыми ноутбуками."""
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


# =============================================================================
# УДАЛЁННЫЕ ФУНКЦИИ
# Вызываются в старых ноутбуках, но их кода нет ни в модуле, ни в истории git.
# Повторить старое поведение нельзя, поэтому функции сообщают, чем их заменить.
# =============================================================================

def renumber_amino_acid_atoms(*args, **kwargs):
    """Удалена. Замена: renumber_residue_atoms(mol, ref_base_name=<родительский остаток>)."""
    raise NotImplementedError(
        'renumber_amino_acid_atoms удалена (кода нет в репозитории). Замена: '
        "pt.renumber_residue_atoms(mol, ref_base_name='K') - перенумерация по шаблону "
        'родительского остатка, см. шаг 3.1 нового 1_charge_calculation.ipynb.')


def save_substructure_mol(*args, **kwargs):
    """Удалена. Замена: save_chem_to_pdb и save_chem_to_smiles для match_dict['substructure'][имя]."""
    raise NotImplementedError(
        'save_substructure_mol удалена (кода нет в репозитории). Замена: '
        "pt.save_chem_to_pdb(match_dict['substructure'][mon_name], путь) и "
        "pt.save_chem_to_smiles(..., atom_map=True), см. шаг 2.2 нового 1_charge_calculation.ipynb.")
