"""Лог для отладки: start_log / stop_log / log_note и декоратор @logged."""
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


## Лог для отладки
# Включается в ячейке ноутбука: pt.start_log(), выключается: pt.stop_log().
# Каждый запуск пишется в отдельную папку logs/<дата_время>/:
#   log.txt      - читаемый журнал: вызовы функций с параметрами, вложенные вызовы
#                  с отступом, сообщения функций, результаты, записанные файлы, время
#   calls.jsonl  - то же в машинном виде (одна JSON-запись на строку), для сравнения запусков
#   files/       - молекулы из вызовов, сделанных прямо из ноутбука: PDB, SDF (с именами
#                  атомов), mol2 (типы SYBYL приблизительные) и SVG с подписями «индекс:имя»
# Пока лог выключен, декоратор @logged ничего не делает и не замедляет функции.

_LOG = {'dir': None, 'txt': None, 'jsonl': None, 'depth': 0, 'counter': 0, 'stack': [], 'saved': {}}
_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')
_LOG_MAX_ITEMS = 500
_ATOM_COLUMNS = ['index', 'element', 'AtomName', 'resname', 'resnum', 'formal_charge']


def _now():
    return datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]


def _log_write_txt(text):
    _LOG['txt'].write(text + '\n')
    _LOG['txt'].flush()


def _log_write_json(record):
    _LOG['jsonl'].write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
    _LOG['jsonl'].flush()


def _file_sha(path, length=12):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()[:length]


def _git_state(path):
    def git(*args):
        try:
            return subprocess.run(['git', '-C', path] + list(args), capture_output=True,
                                  text=True, timeout=10).stdout
        except Exception:
            return ''
    commit = git('rev-parse', '--short', 'HEAD').strip()
    if not commit:
        return 'не git-репозиторий'
    # формат porcelain: 'XY путь', поэтому strip() до разбора строк нельзя
    changed = [line[3:] for line in git('status', '--porcelain', '--untracked-files=no').splitlines() if line]
    return {'branch': git('branch', '--show-current').strip(), 'commit': commit,
            'uncommitted_files': changed}


def _package_versions():
    versions = {}
    try:
        from importlib import metadata
        for name in ('numpy', 'torch', 'dgl', 'espaloma_charge', 'openff-toolkit',
                     'parmed', 'acpype', 'openbabel'):
            try:
                versions[name] = metadata.version(name)
            except Exception:
                pass
    except ImportError:
        pass
    from rdkit import rdBase
    versions['rdkit'] = rdBase.rdkitVersion
    return versions


def _package_sha(package_dir):
    """sha256 всех .py пакета (по именам и содержимому): версия кода в заголовке лога."""
    digest = hashlib.sha256()
    for name in sorted(f for f in os.listdir(package_dir) if f.endswith('.py')):
        digest.update(name.encode())
        with open(os.path.join(package_dir, name), 'rb') as f:
            digest.update(f.read())
    return digest.hexdigest()


def _log_environment(notebook=None):
    module_path = os.path.dirname(os.path.realpath(__file__))  # папка пакета respartools/
    return {
        'notebook': notebook or os.environ.get('JPY_SESSION_NAME') or 'не определён',
        'cwd': os.getcwd(),
        'python': f'{platform.python_version()} ({sys.executable})',
        # окружение, в котором реально работает ядро (CONDA_DEFAULT_ENV - это окружение терминала)
        'conda_env': os.path.basename(sys.prefix),
        'module': module_path,
        'module_sha256': _package_sha(module_path),
        'git': _git_state(os.path.dirname(module_path)),
        'packages': _package_versions(),
    }


def start_log(folder='logs', notebook=None):
    """
    Включает подробный лог для отладки (по умолчанию выключен).

    Аргументы:
        folder (str) - папка для логов; внутри создаётся папка запуска <дата_время>
        notebook (str) - имя ноутбука, если Jupyter не сообщает его сам
    Возвращает:
        str - путь к папке запуска
    """
    if _LOG['dir'] is not None:
        stop_log()
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.abspath(os.path.join(folder, stamp))
    suffix = 1
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = os.path.abspath(os.path.join(folder, f'{stamp}_{suffix}'))
    os.makedirs(os.path.join(run_dir, 'files'))
    _LOG.update(dir=run_dir, depth=0, counter=0, stack=[], saved={},
                txt=open(os.path.join(run_dir, 'log.txt'), 'w', encoding='utf-8'),
                jsonl=open(os.path.join(run_dir, 'calls.jsonl'), 'w', encoding='utf-8'))

    env = _log_environment(notebook)
    record = {'event': 'start', 'time': datetime.datetime.now().isoformat()}
    record.update(env)
    _log_write_json(record)
    lines = [f'ResParTools: лог запуска {stamp}']
    lines += [f'{key}: {json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value}'
              for key, value in env.items()]
    _log_write_txt('\n'.join(lines) + '\n' + '=' * 80)
    print(f'Лог включён: {run_dir}')
    return run_dir


def stop_log():
    """Выключает лог и закрывает файлы. Возвращает путь к папке запуска."""
    if _LOG['dir'] is None:
        return None
    run_dir = _LOG['dir']
    record = {'event': 'stop', 'time': datetime.datetime.now().isoformat(), 'calls': _LOG['counter']}
    espaloma = sys.modules.get('espaloma_charge')
    if espaloma is not None:
        # важно: в darwin_ec копия из site-packages может перекрывать espaloma-charge_mod
        record['espaloma_charge_file'] = getattr(espaloma, '__file__', None)
    _log_write_json(record)
    _log_write_txt('=' * 80 + f"\nконец: {_now()}; записей: {_LOG['counter']}"
                   + (f"\nespaloma_charge загружен из: {record['espaloma_charge_file']}"
                      if 'espaloma_charge_file' in record else ''))
    _LOG['txt'].close()
    _LOG['jsonl'].close()
    _LOG.update(dir=None, txt=None, jsonl=None, stack=[], depth=0)
    print(f'Лог сохранён: {run_dir}')
    return run_dir


def log_note(text, **data):
    """
    Добавляет в лог запись из ноутбука: текст и любые данные.
    Молекулы сохраняются в files/, словари и списки записываются целиком.
    Пример: pt.log_note('заряды Espaloma', charges=charges, mol=mol)
    """
    if _LOG['dir'] is None:
        return
    _LOG['counter'] += 1
    note_id = _LOG['counter']
    described = {key: _describe(value, note_id, f'note.{key}', save_files=True)
                 for key, value in data.items()}
    _log_write_json({'event': 'note', 'id': note_id, 'time': _now(), 'text': text, 'data': described})
    _log_write_txt(f'[{note_id:04d}] {_now()}  ЗАМЕТКА: {text}')
    for key, value in described.items():
        _log_write_txt(f'    {key} = {_short(value)}')


def _safe(getter):
    try:
        return getter()
    except Exception:
        return None


def _describe_file(path):
    return {'file': os.path.abspath(path), 'size': os.path.getsize(path), 'sha256': _file_sha(path)}


def _describe(value, call_id, label, save_files, level=0):
    """Превращает значение в JSON-совместимое описание для лога."""
    if isinstance(value, Chem.Mol):
        return _describe_mol(value, call_id, label, save_files)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, str):
        if len(value) < 1024 and os.path.isfile(value):
            return _describe_file(value)
        return value if len(value) <= 2000 else value[:2000] + '…'
    if isinstance(value, np.ndarray):
        return {'type': 'ndarray', 'shape': list(value.shape),
                'values': value.tolist() if value.size <= _LOG_MAX_ITEMS else 'слишком большой'}
    if level > 5:
        return repr(value)[:300]
    if isinstance(value, dict):
        return {str(key): _describe(item, call_id, f'{label}.{key}', save_files, level + 1)
                for key, item in list(value.items())[:_LOG_MAX_ITEMS]}
    if isinstance(value, (list, tuple, set)):
        return [_describe(item, call_id, f'{label}[{i}]', save_files, level + 1)
                for i, item in enumerate(list(value)[:_LOG_MAX_ITEMS])]
    return repr(value)[:300]


def _describe_mol(mol, call_id, label, save_files):
    atoms = []
    for atom in mol.GetAtoms():
        info = atom.GetPDBResidueInfo()
        atoms.append([atom.GetIdx(), atom.GetSymbol(),
                      atom.GetProp('AtomName') if atom.HasProp('AtomName') else '',
                      info.GetResidueName().strip() if info else '',
                      info.GetResidueNumber() if info else None,
                      atom.GetFormalCharge()])
    summary = {'type': 'Mol',
               'name': mol.GetProp('_Name') if mol.HasProp('_Name') else '',
               'n_atoms': mol.GetNumAtoms(),
               'n_heavy': sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1),
               'formula': _safe(lambda: rdMolDescriptors.CalcMolFormula(mol)),
               'formal_charge': sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
               'smiles': _safe(lambda: Chem.MolToSmiles(mol)),
               'atom_columns': _ATOM_COLUMNS,
               'atoms': atoms}
    if save_files:
        summary['files'] = _save_mol_files(mol, call_id, label, atoms)
    return summary


def _save_mol_files(mol, call_id, label, atoms):
    """Сохраняет молекулу в PDB, SDF, mol2 и SVG. Одинаковые молекулы сохраняются один раз."""
    key = hashlib.sha256((str(_safe(lambda: Chem.MolToMolBlock(mol, kekulize=False)))
                          + json.dumps(atoms)).encode()).hexdigest()
    if key in _LOG['saved']:
        return _LOG['saved'][key]
    safe_label = re.sub(r'[^\w.-]+', '_', label)[:80]
    base = os.path.join(_LOG['dir'], 'files', f'{call_id:04d}_{safe_label}')
    written = []
    for ext, writer in (('pdb', _write_pdb), ('sdf', _write_sdf), ('mol2', _write_mol2), ('svg', _write_svg)):
        path = f'{base}.{ext}'
        try:
            writer(mol, path)
            written.append(os.path.relpath(path, _LOG['dir']))
        except Exception as e:
            written.append(f'{ext}: не записан ({type(e).__name__}: {e})')
    _LOG['saved'][key] = written
    return written


def _write_pdb(mol, path):
    Chem.MolToPDBFile(mol, path)


def _write_sdf(mol, path):
    mol_copy = Chem.Mol(mol)
    if any(atom.HasProp('AtomName') for atom in mol_copy.GetAtoms()):
        for atom in mol_copy.GetAtoms():
            if not atom.HasProp('AtomName'):
                atom.SetProp('AtomName', '')
        Chem.CreateAtomStringPropertyList(mol_copy, 'AtomName')
    writer = Chem.SDWriter(path)
    try:
        writer.write(mol_copy)
    except Exception:
        writer.SetKekulize(False)
        writer.write(mol_copy)
    finally:
        writer.close()


def _sybyl_type(atom):
    """Приблизительный тип SYBYL по элементу, гибридизации и ароматичности."""
    symbol = atom.GetSymbol()
    hyb = atom.GetHybridization()
    SP, SP2 = Chem.HybridizationType.SP, Chem.HybridizationType.SP2
    double_to = [b.GetOtherAtom(atom).GetSymbol() for b in atom.GetBonds()
                 if b.GetBondType() == Chem.BondType.DOUBLE]
    if symbol == 'C':
        return 'C.ar' if atom.GetIsAromatic() else {SP: 'C.1', SP2: 'C.2'}.get(hyb, 'C.3')
    if symbol == 'N':
        if atom.GetIsAromatic():
            return 'N.ar'
        if hyb == SP:
            return 'N.1'
        if hyb == SP2:
            return 'N.2' if double_to else 'N.pl3'
        return 'N.4' if atom.GetFormalCharge() == 1 else 'N.3'
    if symbol == 'O':
        return 'O.2' if double_to else 'O.3'
    if symbol == 'S':
        n_double_o = double_to.count('O')
        return 'S.o2' if n_double_o >= 2 else 'S.o' if n_double_o == 1 else 'S.2' if double_to else 'S.3'
    if symbol == 'P':
        return 'P.3'
    return symbol


def _write_mol2(mol, path):
    bond_types = {Chem.BondType.SINGLE: '1', Chem.BondType.DOUBLE: '2',
                  Chem.BondType.TRIPLE: '3', Chem.BondType.AROMATIC: 'ar'}
    has_charges = any(atom.HasProp('PartialCharge') for atom in mol.GetAtoms())
    conf = mol.GetConformer() if mol.GetNumConformers() else None
    lines = ['@<TRIPOS>MOLECULE', mol.GetProp('_Name') if mol.HasProp('_Name') else 'MOL',
             f'{mol.GetNumAtoms()} {mol.GetNumBonds()} 1 0 0', 'SMALL',
             'USER_CHARGES' if has_charges else 'NO_CHARGES', '', '@<TRIPOS>ATOM']
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        pos = conf.GetAtomPosition(idx) if conf else None
        x, y, z = (pos.x, pos.y, pos.z) if pos else (0.0, 0.0, 0.0)
        info = atom.GetPDBResidueInfo()
        resnum = info.GetResidueNumber() if info else 1
        resname = info.GetResidueName().strip() if info else 'UNL'
        name = atom.GetProp('AtomName') if atom.HasProp('AtomName') else f'{atom.GetSymbol()}{idx + 1}'
        atom_type = atom.GetProp('Mol2AtomType') if atom.HasProp('Mol2AtomType') else _sybyl_type(atom)
        charge = atom.GetDoubleProp('PartialCharge') if atom.HasProp('PartialCharge') else 0.0
        lines.append(f'{idx + 1:7d} {name:<6s} {x:10.4f} {y:10.4f} {z:10.4f} {atom_type:<6s} '
                     f'{resnum:4d} {resname}{resnum} {charge:9.4f}')
    lines.append('@<TRIPOS>BOND')
    for bond in mol.GetBonds():
        lines.append(f'{bond.GetIdx() + 1:6d} {bond.GetBeginAtomIdx() + 1:5d} '
                     f'{bond.GetEndAtomIdx() + 1:5d} {bond_types.get(bond.GetBondType(), "1")}')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _write_svg(mol, path):
    mol_copy = Chem.Mol(mol)
    for atom in mol_copy.GetAtoms():
        name = atom.GetProp('AtomName') if atom.HasProp('AtomName') else ''
        atom.SetProp('atomNote', f'{atom.GetIdx()}:{name}' if name else str(atom.GetIdx()))
    rdDepictor.Compute2DCoords(mol_copy)
    size = 600 if mol_copy.GetNumAtoms() <= 40 else 1200
    drawer = rdMolDraw2D.MolDraw2DSVG(size, int(size * 0.75))
    try:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol_copy)
    except Exception:
        drawer = rdMolDraw2D.MolDraw2DSVG(size, int(size * 0.75))
        drawer.DrawMolecule(rdMolDraw2D.PrepareMolForDrawing(mol_copy, kekulize=False))
    drawer.FinishDrawing()
    with open(path, 'w') as f:
        f.write(drawer.GetDrawingText())


def _short(value, indent='    ', limit=3000):
    """
    Представление описания для log.txt (полное - в calls.jsonl). Короткие значения - одной
    строкой, длинные словари и списки - по одному ключу на строку, чтобы словари
    сопоставления атомов читались глазами.
    """
    if isinstance(value, dict) and value.get('type') == 'Mol':
        names = ' '.join(f'{a[0]}:{a[2] or a[1]}' for a in value['atoms'])
        text = (f"Mol {value['name']} {value['formula']}, атомов {value['n_atoms']} "
                f"(тяжёлых {value['n_heavy']}), формальный заряд {value['formal_charge']}")
        if value.get('files'):
            text += f"; файлы: {', '.join(value['files'])}"
        return text + f'\n{indent}    атомы: {names}'
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= 150 or not isinstance(value, (dict, list)) or not value:
        return text if len(text) <= limit else text[:limit] + f'… (ещё {len(text) - limit} символов в calls.jsonl)'
    inner = indent + '    '
    if isinstance(value, dict):
        return ''.join(f'\n{inner}{key}: {_short(item, inner, limit)}' for key, item in value.items())
    return ''.join(f'\n{inner}[{i}] {_short(item, inner, limit)}' for i, item in enumerate(value))


class _StdoutTee:
    """Дублирует вывод функций в консоль и в log.txt (цвета и анимация '\\r' убираются)."""
    def __init__(self, original):
        self.original = original
        self.buffer = ''

    def write(self, text):
        self.original.write(text)
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            self._log_line(line)
        return len(text)

    def _log_line(self, line):
        line = _ANSI_RE.sub('', line.split('\r')[-1]).rstrip()
        if line and _LOG['txt'] is not None:
            _log_write_txt('    ' * _LOG['depth'] + '» ' + line)

    def flush(self):
        self.original.flush()

    def flush_rest(self):
        if self.buffer:
            self._log_line(self.buffer)
            self.buffer = ''

    def __getattr__(self, name):
        return getattr(self.original, name)


def logged(func):
    """
    Декоратор: записывает вызов функции в лог (если он включён через start_log).
    Для функций с @data_to_dict ставится под ним, чтобы в лог попадал каждый файл отдельно.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        signature = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if _LOG['dir'] is None:
            return func(*args, **kwargs)
        return _logged_call(func, signature, args, kwargs)
    return wrapper


def _snapshot_output_files(params):
    """
    Файлы, которые функция могла записать: для каждого строкового параметра-пути - файлы
    в его папке, чьё имя начинается с имени из пути (функции сохранения сами дописывают
    расширение и суффиксы вроде _no_H). Возвращает {путь: время изменения}.
    """
    snapshot = {}
    for value in params.values():
        if not isinstance(value, str) or len(value) >= 1024 or '\n' in value:
            continue
        folder, prefix = os.path.split(os.path.abspath(value))
        if not prefix or not os.path.isdir(folder):
            continue
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            path = os.path.join(folder, name)
            if name.startswith(prefix) and os.path.isfile(path):
                snapshot[path] = os.path.getmtime(path)
    return snapshot


def _logged_call(func, signature, args, kwargs):
    _LOG['counter'] += 1
    call_id = _LOG['counter']
    depth = _LOG['depth']
    top_level = depth == 0
    parent = _LOG['stack'][-1] if _LOG['stack'] else None
    indent = '    ' * depth
    name = func.__name__
    key = _LOG.pop('key', None)  # имя молекулы из @data_to_dict
    label = f'{name}_{key}' if key is not None else name

    try:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        params = dict(bound.arguments)
    except Exception:
        params = {'args': args, 'kwargs': kwargs}
    described = {param: _describe(value, call_id, f'{label}.{param}', save_files=top_level)
                 for param, value in params.items()}

    _log_write_txt(f'{indent}[{call_id:04d}] {_now()}  {name}'
                   + (f"  для '{key}'" if key is not None else '')
                   + (f'   (вызвана из [{parent:04d}])' if parent else ''))
    for param, value in described.items():
        _log_write_txt(f'{indent}    {param} = {_short(value, indent + "    ")}')
    _log_write_json({'event': 'call', 'id': call_id, 'parent': parent, 'depth': depth,
                     'function': name, 'key': key, 'time': _now(), 'params': described})

    _LOG['depth'] += 1
    _LOG['stack'].append(call_id)
    tee, original_showwarning = None, None
    if top_level:
        tee = _StdoutTee(sys.stdout)
        sys.stdout = tee
        original_showwarning = warnings.showwarning

        def showwarning(message, category, filename, lineno, file=None, line=None):
            _log_write_txt('    ' * _LOG['depth'] + f'» {category.__name__}: {message}')
            original_showwarning(message, category, filename, lineno, file, line)
        warnings.showwarning = showwarning

    files_before = _snapshot_output_files(params)
    started = time.time()
    try:
        result = func(*args, **kwargs)
    except Exception as e:
        _log_write_txt(f'{indent}    ОШИБКА {type(e).__name__}: {e}')
        _log_write_json({'event': 'error', 'id': call_id, 'function': name,
                         'error': f'{type(e).__name__}: {e}', 'traceback': traceback.format_exc(),
                         'seconds': round(time.time() - started, 3)})
        raise
    finally:
        if tee is not None:
            tee.flush_rest()
            sys.stdout = tee.original
            warnings.showwarning = original_showwarning
        _LOG['depth'] -= 1
        _LOG['stack'].pop()

    seconds = round(time.time() - started, 3)
    files_after = _snapshot_output_files(params)
    written = [_describe_file(path) for path, mtime in sorted(files_after.items())
               if files_before.get(path) != mtime]
    result_description = _describe(result, call_id, f'{label}.result', save_files=top_level)
    _log_write_txt(f'{indent}    результат = {_short(result_description, indent + "    ")}')
    for info in written:
        _log_write_txt(f"{indent}    записан файл: {info['file']} (sha256 {info['sha256']})")
    _log_write_txt(f'{indent}    время: {seconds} с')
    _log_write_json({'event': 'return', 'id': call_id, 'function': name, 'seconds': seconds,
                     'result': result_description, 'written_files': written})
    return result
