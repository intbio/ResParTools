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


def _log_environment(notebook=None):
    module_path = os.path.realpath(__file__)  # настоящий файл, даже если модуль подключён ссылкой
    return {
        'notebook': notebook or os.environ.get('JPY_SESSION_NAME') or 'не определён',
        'cwd': os.getcwd(),
        'python': f'{platform.python_version()} ({sys.executable})',
        # окружение, в котором реально работает ядро (CONDA_DEFAULT_ENV - это окружение терминала)
        'conda_env': os.path.basename(sys.prefix),
        'module': module_path,
        'module_sha256': _file_sha(module_path),
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


## Вспомогательные мини функции
def get_name(path: str, index=0):
    """
    Аргумент: 
        path - строка с путем к файлу
    Возвращает:
        Имя для ключа в словаре 
    """
    # Сохраняем текущий stderr
    original_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        mol = Chem.MolFromSmiles(path)
        if mol:
            return index
        else:
            name_of_file = os.path.basename(path).split('.')[0]
            return name_of_file
    except Exception as e:
        print(f"Ошибка при обработке '{path}': {e}")
        return None
    finally:
        # Возвращаем stderr на исходное место
        sys.stderr = original_stderr
        
        
## Дектораторы
def data_to_dict(funсtion: Callable):
    '''
    Декоратор для обработки данных различного типа (строка, список, словарь) перед вызовом функции.
    '''
    def call(key, item, kwarg):
        # имя элемента словаря передаётся в лог, чтобы файлы назывались по молекуле
        if _LOG['dir'] is not None:
            _LOG['key'] = key
        return funсtion(item, **kwarg)

    def wrapper(data: str or list[str] or dict[str, str], **kwarg):
        if isinstance(data, str):
            key = get_name(data)
            return {key: call(key, data, kwarg)}
        elif isinstance(data, list):
            keys = [get_name(str_data, i) for i, str_data in enumerate(data)]
            return {key: call(key, str_data, kwarg) for key, str_data in zip(keys, data)}
        elif isinstance(data, dict):
            return {legend: call(legend, str_data, kwarg) for legend, str_data in data.items()}
        else:
            print('Некорректный тип данных ввода', file=sys.stderr)
            return None
    return wrapper

def do_fun_for_2d_list(function: Callable):
    '''
    Декоратор для применения функции к каждому элементу двумерного списка (матрицы).
    '''
    def wrapper(lst: list):
        result = []
        for element in lst:
            if isinstance(element, list):
                # Рекурсивно применяем декоратор к вложенному списку
                result.append(wrapper(element))
            else:
                # Применяем функцию к элементу
                result.append(function(element))
        return result
    return wrapper


## 

@data_to_dict
@logged
def read_file(path):
    '''
    Читает первую строку из файла и возвращает её.
    '''
    with open (path, 'r') as f:
        return f.readline().strip()

def _mol_from_mapped_smiles(str_smi, sanitize=True):
    """
    Читает SMILES, записанный с atom_map=True (номер у каждого атома, 1..n).
    Водороды берутся из файла как есть (открытые валентности остатка в цепи не
    «залечиваются»), атомы расставляются по номерам, номера снимаются.
    Если номера есть не у всех атомов, возвращает None - SMILES читается как обычно.
    """
    if not re.search(r'\[[^\]]*:\d+\]', str_smi):
        return None
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = sanitize
    mol = Chem.MolFromSmiles(str_smi, params)
    if mol is None:
        return None
    maps = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
    if sorted(maps) != list(range(1, mol.GetNumAtoms() + 1)):
        return None
    order = [0] * mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        order[atom.GetAtomMapNum() - 1] = atom.GetIdx()
    mol = Chem.RenumberAtoms(mol, order)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


@data_to_dict
@logged
def smi_to_chem(str_smi: str, sanitize=True, addH=True, make_N_root=False,format_coord='2D'):
    """
    Преобразует SMILES-строку в объект молекулы RDKit и добавляет атомы водорода.
    """
    rdkit_mol = _mol_from_mapped_smiles(str_smi, sanitize)
    if rdkit_mol is None:
        rdkit_mol = Chem.MolFromSmiles(str_smi, sanitize) # переводим во внутренний формат chem
        if addH:
            rdkit_mol = Chem.AddHs(rdkit_mol) # протонируем
        
    if format_coord == '2D':
        # Рассчитываем 2D координаты
        AllChem.Compute2DCoords(rdkit_mol)
    elif format_coord == '3D':
        # Рассчитываем 3D координаты (если еще не рассчитаны)
        Chem.SanitizeMol(rdkit_mol)
        AllChem.EmbedMolecule(rdkit_mol)
        AllChem.UFFOptimizeMolecule(rdkit_mol)
    else:
        raise ValueError("Недопустимый формат. Выберите '2D' или '3D'.")
    # Перенумерация атомов, деля аминогруппу началом молекулы
    if make_N_root:
        # Находим индекс азота из аминогруппы
        root_idx = find_amino_nitrogen(rdkit_mol)
        if root_idx != -1:
            # Получаем SMILES с корневым атомом
            smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, allHsExplicit=True, rootedAtAtom=root_idx)
            # Создаем новую молекулу из SMILES
            rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
            if rdkit_mol is None:
                raise ValueError("Ошибка при создании молекулы после перенумерации атомов")
        else:
            print("Азот аминогруппы не найден. Корень не изменён.")
        # Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE, catchErrors=True)
        Chem.SanitizeMol(rdkit_mol)
        
    # Сортируем атомы, чтобы тяжелые атомы шли первыми, а водороды позже
        atom_order = [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() != 'H']  # тяжелые атомы
        atom_order += [atom.GetIdx() for atom in rdkit_mol.GetAtoms() if atom.GetSymbol() == 'H']  # водороды
    
    # Перенумерация атомов согласно новому порядку
        new_rdkit_mol = Chem.RenumberAtoms(rdkit_mol, atom_order)
        # Пересчитываем валентности
        return new_rdkit_mol
    else:
        return rdkit_mol
    # if make_N_root:
    #     root_idx = find_amino_nitrogen(rdkit_mol)
    #     if root_idx != -1:
        #     smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, 
        #                                      allHsExplicit=True, rootedAtAtom=root_idx)
        #     rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=True)
        # return rdkit_mol


def _finalize_read_mol(rdkit_mol, path, sanitize, removeHs, format_coord):
    """
    Общий хвост для pdb_to_chem и mol2_to_chem: sanitize, удаление H, координаты.
    Свойства атомов (AtomName, PDB residue info, заряды) переживают RemoveHs,
    поэтому имена остаются привязаны к своим атомам.
    """
    if sanitize:
        try:
            Chem.SanitizeMol(rdkit_mol)
        except Exception as e:
            raise ValueError(
                f"{path}: структура не прошла sanitize ({e}). "
                "Обычно это лишняя связь или не указан формальный заряд. "
                "Исправьте файл или вызовите с sanitize=False (без ароматичности и проверки валентностей)."
            )
    if removeHs:
        rdkit_mol = Chem.RemoveHs(rdkit_mol, sanitize=sanitize)

    if format_coord == '2D':
        AllChem.Compute2DCoords(rdkit_mol)
    elif format_coord != '3D':
        raise ValueError("Недопустимый формат. Выберите '2D' или '3D'.")

    rdkit_mol.SetProp('AtomNames', str(rdkit_mol.GetNumAtoms()))
    return rdkit_mol


@data_to_dict
@logged
def pdb_to_chem(path_to_pdb, removeHs=False, make_N_root=False, sanitize=True, format_coord='2D'):
    """
    Открывает PDB файл при помощи RDKit и сохраняет имена атомов как свойства RDKit-атомов.

    Аргументы:
        path_to_pdb (str): Путь к файлу PDB.
        removeHs (bool): Удалять ли водороды. По умолчанию False.
        make_N_root (bool): Не используется, оставлен для совместимости вызовов.
        sanitize (bool): Проверка валентностей и распознавание ароматичности.
            Нужна, чтобы ароматические связи отличались от двойных так же, как в SMILES.
        format_coord (str): '2D' - пересчитать координаты для рисования,
            '3D' - оставить координаты из файла.

    Связи берутся из CONECT, если он есть в файле; иначе RDKit определяет их по расстояниям
    (proximity bonding). Вместе с CONECT proximity bonding не используется: на 2D и
    неоптимизированных структурах он добавляет фантомные связи.

    Возвращает:
        Chem.Mol: Молекула RDKit с сохраненными именами атомов (AtomName).
    """
    with open(path_to_pdb, "r") as pdb_file:
        has_conect = any(line.startswith("CONECT") for line in pdb_file)

    rdkit_mol = Chem.MolFromPDBFile(path_to_pdb, sanitize=False, removeHs=False,
                                    proximityBonding=not has_conect)
    if not rdkit_mol:
        raise ValueError(f"Не удалось загрузить PDB файл: {path_to_pdb}")

    # Имена атомов RDKit читает из фиксированных колонок PDB (13-16)
    for atom in rdkit_mol.GetAtoms():
        atom.SetProp("AtomName", atom.GetPDBResidueInfo().GetName().strip())

    return _finalize_read_mol(rdkit_mol, path_to_pdb, sanitize, removeHs, format_coord)


def _read_mol2_atom_block(path_to_mol2):
    """
    Читает секции @<TRIPOS>MOLECULE и @<TRIPOS>ATOM (первой молекулы в файле).
    Возвращает (имя молекулы, тип зарядов, список словарей по атомам в порядке файла).
    """
    with open(path_to_mol2, "r") as f:
        lines = f.read().splitlines()

    mol_name, charge_type, atoms = "", "", []
    section = None
    mol_line = 0
    for line in lines:
        if line.startswith("@<TRIPOS>"):
            if section == "ATOM":
                break
            section = line[len("@<TRIPOS>"):].strip()
            mol_line = 0
            continue
        if section == "MOLECULE":
            mol_line += 1
            if mol_line == 1:
                mol_name = line.strip()
            elif mol_line == 4:
                charge_type = line.strip()
        elif section == "ATOM" and line.strip():
            fields = line.split()
            atom = {"id": int(fields[0]), "name": fields[1], "type": fields[5]}
            atom["subst_id"] = int(fields[6]) if len(fields) > 6 else 1
            atom["subst_name"] = fields[7] if len(fields) > 7 else "UNL"
            atom["charge"] = float(fields[8]) if len(fields) > 8 else 0.0
            atoms.append(atom)
    return mol_name, charge_type, atoms


@data_to_dict
@logged
def mol2_to_chem(path_to_mol2, sanitize=True, removeHs=False, format_coord='2D'):
    """
    Открывает mol2 файл при помощи RDKit и переносит в молекулу всё, что есть в секции ATOM.

    Порядки связей (включая ароматические 'ar') берутся из секции BOND.
    RDKit сам сохраняет имя, тип SYBYL и заряд атома, но теряет номер и имя остатка,
    поэтому секция ATOM дополнительно разбирается вручную.

    Аргументы:
        path_to_mol2 (str): Путь к файлу mol2 (типы атомов SYBYL; файлы с типами GAFF RDKit не читает).
        sanitize (bool): Проверка валентностей и распознавание ароматичности.
        removeHs (bool): Удалять ли водороды.
        format_coord (str): '2D' - пересчитать координаты для рисования, '3D' - оставить из файла.

    Свойства атомов:
        AtomName (str), Mol2AtomType (str), PartialCharge (float),
        PDB residue info: имя атома, имя и номер остатка.
        Имя остатка: subst_name без хвоста с номером остатка (конвенция Tripos: 'KMA2' -> 'KMA').
    Свойства молекулы:
        AtomNames, Mol2Name, Mol2ChargeType.

    Возвращает:
        Chem.Mol
    """
    rdkit_mol = Chem.MolFromMol2File(path_to_mol2, sanitize=False, removeHs=False)
    if not rdkit_mol:
        raise ValueError(f"Не удалось загрузить mol2 файл: {path_to_mol2} "
                         "(RDKit понимает только типы атомов SYBYL, не GAFF)")

    mol_name, charge_type, atoms = _read_mol2_atom_block(path_to_mol2)
    if len(atoms) != rdkit_mol.GetNumAtoms():
        raise ValueError(f"{path_to_mol2}: в секции ATOM {len(atoms)} атомов, "
                         f"а RDKit прочитал {rdkit_mol.GetNumAtoms()}")

    for atom, info in zip(rdkit_mol.GetAtoms(), atoms):
        # RDKit сохраняет порядок атомов из файла; проверяем, что сопоставление верное
        if atom.HasProp("_TriposAtomName") and atom.GetProp("_TriposAtomName") != info["name"]:
            raise ValueError(f"{path_to_mol2}: порядок атомов RDKit не совпадает с файлом "
                             f"(атом {info['id']}: {atom.GetProp('_TriposAtomName')} != {info['name']})")
        resname = info["subst_name"]
        if resname.endswith(str(info["subst_id"])) and len(resname) > len(str(info["subst_id"])):
            resname = resname[:-len(str(info["subst_id"]))]

        atom.SetProp("AtomName", info["name"])
        atom.SetProp("Mol2AtomType", info["type"])
        atom.SetDoubleProp("PartialCharge", info["charge"])
        set_PDB_residue_info(atom, info["name"], resname=resname, resid=info["subst_id"], segid="A")

    # В mol2 нет поля формального заряда, и RDKit превращает заряженные атомы в радикалы
    # или отвергает их. Восстанавливаем заряды по валентности:
    #   N с 4 связями (аммоний, иминий)                          -> +1
    #   O с одной одинарной связью без H (сульфонат, карбоксилат) -> -1,
    #   только если водороды в файле явные: иначе так же выглядит обычная OH-группа
    rdkit_mol.UpdatePropertyCache(strict=False)
    has_explicit_H = any(atom.GetAtomicNum() == 1 for atom in rdkit_mol.GetAtoms())
    charged = []
    for atom in rdkit_mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            continue
        valence = atom.GetExplicitValence()
        if atom.GetAtomicNum() == 7 and valence == 4:
            atom.SetFormalCharge(1)
            charged.append(atom.GetProp("AtomName") + "(+1)")
        elif atom.GetAtomicNum() == 8 and valence == 1 and has_explicit_H:
            atom.SetFormalCharge(-1)
            atom.SetNumRadicalElectrons(0)
            charged.append(atom.GetProp("AtomName") + "(-1)")
    if charged:
        print(f"{path_to_mol2}: формальные заряды восстановлены по валентности: {', '.join(charged)}")

    rdkit_mol.SetProp("Mol2Name", mol_name)
    rdkit_mol.SetProp("Mol2ChargeType", charge_type)
    return _finalize_read_mol(rdkit_mol, path_to_mol2, sanitize, removeHs, format_coord)

@logged
def file_opener(path, **kwargs):
    """
    Открывает файл с молекулой, автоматически определяя формат (.smi/.smiles, .mol2, .pdb).

    Parameters:
    -----------
    path : str или list[str]
        Путь к файлу или список путей (форматы в списке могут быть разными)
    **kwargs : dict
        Дополнительные параметры для функций загрузки
    Возвращает:
        dict {имя файла без расширения: Chem.Mol}, как pdb_to_chem и smi_to_chem
    """
    if isinstance(path, (list, tuple)):
        result = {}
        for one_path in path:
            opened = file_opener(one_path, **dict(kwargs))
            duplicates = set(result) & set(opened)
            if duplicates:
                print_red(f'⚠ Молекулы с одинаковыми именами {sorted(duplicates)}: '
                          f'{one_path} заменяет открытую ранее.')
            result.update(opened)
        return result

    path_obj = Path(path)
    name = path_obj.stem
    extension = path_obj.suffix.lower()
    
    print(f"Загрузка файла: {name} (формат: {extension})")
    
    if extension in (".smi", ".smiles"):
        smi_str = read_file(path)
        rdkit_mol = smi_to_chem(smi_str, **kwargs)
    elif extension in (".mol2", ".pdb"):
        # Для файлов с геометрией по умолчанию сохраняем координаты из файла
        kwargs.setdefault('format_coord', '3D')
        reader = mol2_to_chem if extension == ".mol2" else pdb_to_chem
        rdkit_mol = reader(path, **kwargs)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {extension}. "
                       f"Поддерживаемые: .smi, .smiles, .mol2, .pdb")
    
    return rdkit_mol


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


# def modify_atoms_for_charge(mol):
#     """
#     Модифицирует атомные номера на основе заряда.
#     Формальный заряд добавляется к атомному номеру.
#     """
#     for atom in mol.GetAtoms():
#         charge = atom.GetFormalCharge()
#         atom.SetIntProp("OriginalAtomicNum", atom.GetAtomicNum())  # Сохраняем оригинальный номер
#         atom.SetAtomicNum(atom.GetAtomicNum() + charge * 100)  # Смещаем атомный номер

# def restore_original_atomic_numbers(mol):
#     """
#     Восстанавливает исходные атомные номера после модификации.
#     """
#     for atom in mol.GetAtoms():
#         if atom.HasProp("OriginalAtomicNum"):
#             atom.SetAtomicNum(atom.GetIntProp("OriginalAtomicNum"))
#             atom.ClearProp("OriginalAtomicNum")

def match_chem(mol_chem_1, mol_chem_2, compare_any_bond=False, match_residue_number=None):
    """
    Находит максимальную общую подструктуру (MCS) между двумя молекулами,
    ограничивая поиск атомами, принадлежащими заданному остатку во второй молекуле.
    
    Args:
        mol_chem_1 (Chem.Mol): Первая молекула (мономер).
        mol_chem_2 (Chem.Mol): Вторая молекула (полимер).
        compare_any_bond (bool): Игнорировать порядок связей.
        match_residue_number (int, optional): Номер остатка для фильтрации атомов второй молекулы.
    
    Returns:
        Tuple[Chem.Mol, Dict[int, int]]: Общая подструктура и соответствие атомов.
    """
    
    class CompareElements(rdFMCS.MCSAtomCompare):
        def __call__(self, p, mol1, atom1, mol2, atom2):
            a1 = mol1.GetAtomWithIdx(atom1)
            a2 = mol2.GetAtomWithIdx(atom2)

            # Проверки атомов
            if a1.GetAtomicNum() != a2.GetAtomicNum():
                return False
            if p.MatchValences and a1.GetTotalValence() != a2.GetTotalValence():
                return False
            if p.MatchChiralTag and not self.CheckAtomChirality(p, mol1, atom1, mol2, atom2):
                print(a1.GetProp('AtomName'), a2.GetProp('AtomName'))
                return False
            if p.MatchFormalCharge and not self.CheckAtomCharge(p, mol1, atom1, mol2, atom2):
                print(a1.GetProp('AtomName'), a2.GetProp('AtomName'))
                return False
            if p.RingMatchesRingOnly:
                return self.CheckAtomRingMatch(p, mol1, atom1, mol2, atom2)
            return True
        
    # Функция для фильтрации атомов по номеру остатка
    def filter_atoms_by_residue_number(mol, residue_number):
        """Возвращает новую молекулу, содержащую только атомы с заданным номером остатка."""
        indices_to_keep = []
        for atom in mol.GetAtoms():
            monomer_info = atom.GetMonomerInfo()
            if monomer_info and monomer_info.GetResidueNumber() == residue_number:
                indices_to_keep.append(atom.GetIdx())
        return Chem.PathToSubmol(mol, indices_to_keep)

    # Применяем фильтр к второй молекуле, если задан номер остатка
    if match_residue_number is not None:
        print(f'Matched only with residue namber {match_residue_number}')
        filtered_indices = []  # Список индексов атомов, которые принадлежат нужному остатку
        for atom in mol_chem_2.GetAtoms():
            monomer_info = atom.GetMonomerInfo()
            if monomer_info and monomer_info.GetResidueNumber() == match_residue_number:
                filtered_indices.append(atom.GetIdx())

        # Создаем отфильтрованную молекулу для поиска MCS
        filtered_mol_chem_2 = Chem.PathToSubmol(mol_chem_2, filtered_indices)
        # print(f"Filtered molecule contains {filtered_mol_chem_2.GetNumAtoms()} atoms.")
    else:
        filtered_mol_chem_2 = mol_chem_2

    # Настройки поиска MCS
    params = rdFMCS.MCSParameters()
    params.AtomCompareParameters.MatchValences = True
    params.AtomTyper = CompareElements()
    params.BondCompare = rdFMCS.BondCompare.CompareAny if compare_any_bond else rdFMCS.BondCompare.CompareOrder

    # Поиск MCS
    res = rdFMCS.FindMCS([mol_chem_1, filtered_mol_chem_2], params)
    # print(f'FindMCS result: {res}')
    substructure = Chem.MolFromSmarts(res.smartsString)

    # Сопоставление атомов    
    match_mol1 = mol_chem_1.GetSubstructMatch(substructure)
    match_polymer = mol_chem_2.GetSubstructMatch(substructure)
    # print(match_mol1, match_polymer, sep='\n')
    dict_matches = {}
    # if type_match_dict == 'Indexes':
    dict_matches['Indexes'] = dict(zip(match_mol1, match_polymer))
    # elif type_match_dict == 'Names':
    if mol_chem_1.HasProp('AtomNames') and mol_chem_2.HasProp('AtomNames'):
        mol1_atoms_name = [ mol_chem_1.GetAtomWithIdx(atom_idx).GetProp('AtomName') for atom_idx in match_mol1 ]
        mol2_atoms_name = [ mol_chem_2.GetAtomWithIdx(atom_idx).GetProp('AtomName') for atom_idx in match_polymer ]
        dict_matches['Names'] = dict(zip(mol1_atoms_name, mol2_atoms_name))
    # else:
    #     raise ValueError("Неверное значение type_match_dict. Допустимые значения: 'Indexes' и 'Names'.")
        

    # Копирование зарядов из исходной молекулы
    for idx in match_polymer:
        atom = mol_chem_2.GetAtomWithIdx(idx)
        if atom.GetFormalCharge() != 0:
            substructure.GetAtomWithIdx(match_polymer.index(idx)).SetFormalCharge(atom.GetFormalCharge())

    return substructure, dict_matches

def remove_hydrogens_preserve_indices(mol):
    """
    Убирает все протоны из молекулы, сохраняя индексы тяжёлых атомов.
    
    Параметры:
        mol (Chem.Mol): исходная молекула с H.
    
    Возвращает:
        mol_noH (Chem.Mol): молекула без H.
        heavy_idx_map (dict): словарь {новый индекс тяжелого атома: исходный индекс в mol}
    """
    mol_rw = Chem.RWMol(mol)  # Создаём RWMol для редактирования

    # Запоминаем исходный индекс каждого атома до удаления H
    for atom in mol_rw.GetAtoms():
        atom.SetIntProp('_orig_idx', atom.GetIdx())

    # Проходим по атомам и собираем индексы протонов
    H_indices = [atom.GetIdx() for atom in mol_rw.GetAtoms() if atom.GetAtomicNum() == 1]

    # Переносим удаляемые H в счётчик явных H соседа, чтобы сохранить валентность
    # (иначе NH2 мономера и NH полимера получат разную валентность при MatchValences)
    for idx in H_indices:
        for neighbor in mol_rw.GetAtomWithIdx(idx).GetNeighbors():
            if neighbor.GetAtomicNum() != 1:
                neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)

    # Удаляем протонные атомы (любые индексы)
    for idx in sorted(H_indices, reverse=True):
        mol_rw.RemoveAtom(idx)

    mol_noH = mol_rw.GetMol()
    mol_noH.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol_noH)  # RemoveAtom сбрасывает информацию о кольцах

    # Словарь соответствия новых индексов → оригинальные
    heavy_idx_map = {}
    for atom in mol_noH.GetAtoms():
        heavy_idx_map[atom.GetIdx()] = atom.GetIntProp('_orig_idx')
        atom.ClearProp('_orig_idx')

    # Возвращаем молекулу и mapping
    return mol_noH, heavy_idx_map


def match_chem_v3(mol_chem_1, mol_chem_2,
                  compare_any_bond=False,
                  match_residue_number=None,
                  match_type='heavy',  # 'heavy' или 'all'
                  timeout=3):
    """
    Находит максимальную общую подструктуру между мономером и полимером.
    
    Args:
        mol_chem_1 (Chem.Mol) - мономер
        mol_chem_2 (Chem.Mol) - полимер
        compare_any_bond (bool) - сравнивать любые связи
        match_residue_number (int) - номер остатка для фильтрации
        match_type (str) - 'heavy' (только тяжелые атомы) или 'all' (все атомы)
        timeout (int) - таймаут поиска MCS
    
    Returns:
        substructure (Chem.Mol) - подструктура с протонами
        dict_matches (dict) - соответствие индексов и имен атомов
    """
    # --- 1. Предупреждение для match_type='all' ---
    if match_type == 'all':
        print("ВНИМАНИЕ: Сопоставление по всем атомам (включая водороды)")
        print("Это может занять очень много времени для больших молекул!")
    
    # --- 2. Фильтрация по номеру остатка ---
    def filter_atoms_by_residue_number(mol, residue_number):
        """Возвращает новую молекулу, содержащую только атомы с заданным номером остатка"""
        indices_to_keep = []
        for atom in mol.GetAtoms():
            monomer_info = atom.GetMonomerInfo()
            if monomer_info and monomer_info.GetResidueNumber() == residue_number:
                indices_to_keep.append(atom.GetIdx())
        
        if not indices_to_keep:
            print(f"Предупреждение: Остаток с номером {residue_number} не найден")
            return mol
        
        print(f"Фильтрация по остатку {residue_number}: оставлено {len(indices_to_keep)} атомов")
        return Chem.PathToSubmol(mol, indices_to_keep)
    
    if match_residue_number is not None:
        print(f'Поиск MCS только для остатка с номером {match_residue_number}')
        filtered_mol_chem_2 = filter_atoms_by_residue_number(mol_chem_2, match_residue_number)
    else:
        filtered_mol_chem_2 = mol_chem_2
    
    # --- 3. Подготовка молекул в зависимости от match_type ---
    if match_type == 'heavy':
        print("Сопоставление по тяжелым атомам (водороды исключены)")
        mol_1_processed = Chem.RemoveHs(mol_chem_1)
        mol_2_processed = Chem.RemoveHs(filtered_mol_chem_2)
    else:  # 'all'
        print("Сопоставление по всем атомам (включая водороды)")
        mol_1_processed = mol_chem_1
        mol_2_processed = filtered_mol_chem_2
    
    # --- 4. Настройка параметров MCS ---
    mcs_params = rdFMCS.MCSParameters()
    mcs_params.Timeout = timeout
    mcs_params.Maximize = True
    mcs_params.CompareAnyBond = compare_any_bond
    
    if match_type == 'heavy':
        mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElementsAndCharge
        mcs_params.MatchValences = False
    else:
        mcs_params.AtomCompare = rdFMCS.AtomCompare.CompareElements
        mcs_params.MatchValences = True
    
    mcs_params.BondCompare = rdFMCS.BondCompare.CompareOrder
    mcs_params.RingMatchesRingOnly = False
    mcs_params.CompleteRingsOnly = False
    
    # --- 5. Поиск MCS ---
    print("Поиск максимальной общей подструктуры...")
    mcs_result = rdFMCS.FindMCS([mol_1_processed, mol_2_processed], mcs_params)
    
    if not mcs_result or mcs_result.numAtoms == 0:
        raise ValueError("Общая подструктура не найдена!")
    
    print(f"Найдена общая подструктура из {mcs_result.numAtoms} атомов")
    
    # --- 6. Получение сопоставлений ---
    query = Chem.MolFromSmarts(mcs_result.smartsString)
    matches_1 = mol_1_processed.GetSubstructMatches(query, uniquify=False)
    matches_2 = mol_2_processed.GetSubstructMatches(query, uniquify=False)
    
    # --- 7. Выбор лучшего сопоставления (максимум атомов) ---
    # best_match_1 = None
    # best_match_2 = None
    max_size = 0
    
    for match_1 in matches_1:
        for match_2 in matches_2:
            size = len(match_1)  # количество сопоставленных атомов
            if size > max_size:
                max_size = size
                best_match_1 = match_1
                best_match_2 = match_2
    
    if best_match_1 is None:
        best_match_1 = matches_1[0]
        best_match_2 = matches_2[0]
    
    print(f"Выбрано сопоставление с {max_size} атомами")
    
    # --- 8. Создание словаря соответствий ---
    dict_matches = {}
    
    # Сохраняем индексы (без сортировки!)
    dict_matches['Indexes'] = dict(zip(best_match_1, best_match_2))
    
    # Добавляем имена атомов, если они есть
    if mol_chem_1.HasProp('AtomNames') and mol_chem_2.HasProp('AtomNames'):
        try:
            mol1_atoms_name = [mol_chem_1.GetAtomWithIdx(idx).GetProp('AtomName') 
                               for idx in best_match_1]
            mol2_atoms_name = [mol_chem_2.GetAtomWithIdx(idx).GetProp('AtomName') 
                               for idx in best_match_2]
            dict_matches['Names'] = dict(zip(mol1_atoms_name, mol2_atoms_name))
        except Exception as e:
            print(f"Предупреждение: Не удалось получить имена атомов - {e}")
    
    # --- 9. Построение подструктуры с протонами ---
    # Берем атомы из исходной молекулы-полимера (с водородами)
    # Используем best_match_2 как есть, без сортировки
    
    # Добавляем водороды, связанные с сопоставленными атомами
    all_indices = list(best_match_2)  # копируем список
    
    for idx in best_match_2:
        atom = mol_chem_2.GetAtomWithIdx(idx)
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() == 1:  # Водород
                if neighbor.GetIdx() not in all_indices:
                    all_indices.append(neighbor.GetIdx())
    
    # Не сортируем индексы, сохраняем порядок
    substructure = Chem.PathToSubmol(mol_chem_2, all_indices)
    
    # Заряды копируются автоматически через PathToSubmol
    # Дополнительно проверяем и восстанавливаем если нужно
    for i, old_idx in enumerate(all_indices):
        old_atom = mol_chem_2.GetAtomWithIdx(old_idx)
        new_atom = substructure.GetAtomWithIdx(i)
        if old_atom.GetFormalCharge() != 0:
            new_atom.SetFormalCharge(old_atom.GetFormalCharge())
    
    return substructure, dict_matches

@logged
def match_chem_v6(
    mol_chem_1, mol_chem_2,
    only_heavy_mapping=True,
    compare_any_bond=False,
    match_residue_number=None,
    timeout=30
):
    """
    Сопоставление двух молекул с использованием RDKit MCS.
    Возвращает подструктуры с H и без H, а также словарь индексов и имен.

    Args:
        mol_chem_1 (Chem.Mol) - мономер (из его атомов строится подструктура)
        mol_chem_2 (Chem.Mol) - полимер
        only_heavy_mapping (bool) - MCS только по тяжелым атомам, H сопоставляются потом
        compare_any_bond (bool) - игнорировать порядок связей
        match_residue_number (int) - искать только среди атомов этого остатка полимера
        timeout (int) - таймаут поиска MCS, с

    Returns:
        substructure_dict (dict) - {"H": подструктура с H, "noH": подструктура без H}
        dict_matches (dict) - {"Indexes": {индекс в mol_chem_1: индекс в mol_chem_2},
                               "Names": {имя в mol_chem_1: имя в mol_chem_2}} (если есть имена)
        Атом i подструктуры "H" соответствует i-й паре в "Indexes".
        Если сопоставление не найдено, возвращает (None, None).
    """
    import threading
    import warnings
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    # === 1. Фильтрация по остаткам ===
    residue_index_map = None  # индекс в подмолекуле остатка -> индекс в mol_chem_2
    mol2_for_mcs = mol_chem_2

    if match_residue_number is not None:
        residue_atom_indices = [
            atom.GetIdx()
            for atom in mol_chem_2.GetAtoms()
            if atom.GetMonomerInfo() is not None
            and atom.GetMonomerInfo().GetResidueNumber() == match_residue_number
        ]

        if not residue_atom_indices:
            warnings.warn("⚠ Указанный residue не найден в молекуле.")
            return None, None

        bond_indices = []
        for bond in mol_chem_2.GetBonds():
            if bond.GetBeginAtomIdx() in residue_atom_indices and bond.GetEndAtomIdx() in residue_atom_indices:
                bond_indices.append(bond.GetIdx())

        atom_map = {}  # заполняется RDKit: индекс в mol_chem_2 -> индекс в подмолекуле
        residue_submol = Chem.PathToSubmol(mol_chem_2, bond_indices, useQuery=False, atomMap=atom_map)
        residue_submol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(residue_submol)

        # mapping submol_idx -> polymer_idx
        residue_index_map = {new_idx: old_idx for old_idx, new_idx in atom_map.items()}
        mol2_for_mcs = residue_submol

    # === 2. Heavy mapping: удаляем H для поиска MCS ===
    if only_heavy_mapping:
        mol1_work, mol1_map = remove_hydrogens_preserve_indices(mol_chem_1)
        mol2_work, mol2_map = remove_hydrogens_preserve_indices(mol2_for_mcs)
    else:
        warnings.warn("⚠ MCS с явными протонами может быть очень медленным.")
        mol1_work = mol_chem_1
        mol2_work = mol2_for_mcs
        mol1_map = {atom.GetIdx(): atom.GetIdx() for atom in mol1_work.GetAtoms()}
        mol2_map = {atom.GetIdx(): atom.GetIdx() for atom in mol2_work.GetAtoms()}

    # индекс в mol2_work -> индекс в исходном mol_chem_2 (с учетом фильтрации по остатку)
    if residue_index_map is not None:
        mol2_map = {i: residue_index_map[j] for i, j in mol2_map.items()}

    # === 3. Настройки MCS ===
    params = rdFMCS.MCSParameters()
    params.Timeout = timeout
    params.AtomTyper = rdFMCS.AtomCompare.CompareElements
    params.AtomCompareParameters.MatchValences = True
    params.AtomCompareParameters.MatchFormalCharge = True
    params.AtomCompareParameters.RingMatchesRingOnly = True
    params.BondTyper = rdFMCS.BondCompare.CompareAny if compare_any_bond else rdFMCS.BondCompare.CompareOrder
    params.BondCompareParameters.RingMatchesRingOnly = True
    params.BondCompareParameters.CompleteRingsOnly = False

    # === 4. Анимация поиска MCS ===
    done = False
    t = threading.Thread(target=animate, args=(lambda: done, "Поиск подструктуры между молекулами"))
    t.start()
    try:
        res = rdFMCS.FindMCS([mol1_work, mol2_work], params)
    finally:
        done = True
        t.join()

    if res.canceled:
        print_red("⚠ MCS остановлен по timeout, подструктура может быть неполной")

    if res.numAtoms == 0:
        warnings.warn("⚠ Общая подструктура не найдена.")
        return None, None

    queryMol = res.queryMol

    # === 5. Получаем сопоставления ===
    # queryMol не хранит валентность и заряд, поэтому проверяем их сами
    def atoms_compatible(a1, a2):
        return (a1.GetTotalValence() == a2.GetTotalValence()
                and a1.GetFormalCharge() == a2.GetFormalCharge())

    matches1 = mol1_work.GetSubstructMatches(queryMol, uniquify=False, maxMatches=1000)
    matches2 = mol2_work.GetSubstructMatches(queryMol, uniquify=False, maxMatches=1000)

    if not matches1 or not matches2:
        warnings.warn("⚠ Сопоставление подструктур не найдено.")
        return None, None

    # Для V6 убираем скоринг, берем первую пару, совместимую по валентности и заряду
    match1 = matches1[0]
    match2 = next(
        (m2 for m2 in matches2
         if all(atoms_compatible(mol1_work.GetAtomWithIdx(i1), mol2_work.GetAtomWithIdx(i2))
                for i1, i2 in zip(match1, m2))),
        None
    )
    if match2 is None:
        print_red(f"⚠ Ни одно из {len(matches2)} сопоставлений не совпадает по валентности "
                  "и заряду атомов, взято первое. Проверьте имена атомов в результате.")
        match2 = matches2[0]

    # индексы переводим в исходные mol_chem_1 / mol_chem_2
    mapping = {mol1_map[i1]: mol2_map[i2] for i1, i2 in zip(match1, match2)}

    # === 6. Восстанавливаем H ===
    if only_heavy_mapping:
        full_mapping = mapping.copy()
        for idx1, idx2 in mapping.items():
            atom1 = mol_chem_1.GetAtomWithIdx(idx1)
            atom2 = mol_chem_2.GetAtomWithIdx(idx2)

            H1 = sorted([n.GetIdx() for n in atom1.GetNeighbors() if n.GetAtomicNum() == 1])
            H2 = sorted([n.GetIdx() for n in atom2.GetNeighbors() if n.GetAtomicNum() == 1])

            for h1, h2 in zip(H1, H2):
                full_mapping[h1] = h2
    else:
        full_mapping = mapping

    # Порядок: по возрастанию индекса в mol_chem_1 — такой же, как у атомов подструктуры
    full_mapping = dict(sorted(full_mapping.items()))

    # === 7. Строим подструктуры ===
    # Удаляем из копии мономера несопоставленные атомы: оставшиеся атомы
    # сохраняют исходный порядок, свойства (AtomName, PDB info), заряды и координаты
    keep = set(full_mapping.keys())
    substructure_rw = Chem.RWMol(mol_chem_1)
    for idx in sorted((a.GetIdx() for a in mol_chem_1.GetAtoms() if a.GetIdx() not in keep), reverse=True):
        substructure_rw.RemoveAtom(idx)
    substructure_H = substructure_rw.GetMol()
    substructure_H.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(substructure_H)

    substructure_no_H, _ = remove_hydrogens_preserve_indices(substructure_H)

    substructure_dict = {"H": substructure_H, "noH": substructure_no_H}

    # === 8. Словарь соответствий ===
    dict_matches = {"Indexes": full_mapping}

    if mol_chem_1.HasProp('AtomNames') and mol_chem_2.HasProp('AtomNames'):
        name_mapping = {}
        for i1, i2 in full_mapping.items():
            name1 = mol_chem_1.GetAtomWithIdx(i1).GetProp('AtomName')
            name2 = mol_chem_2.GetAtomWithIdx(i2).GetProp('AtomName')
            name_mapping[name1] = name2
        dict_matches['Names'] = name_mapping

    return substructure_dict, dict_matches
    
class _MatchData(dict):
    """
    Словарь результатов match_mon_to_pol. Старые имена ключей из ранних ноутбуков
    ('substructure_noH') перенаправляются на новые, не дублируясь в словаре.
    """
    _OLD_KEYS = {'substructure_noH': 'substructure_no_H'}

    def __missing__(self, key):
        if key in self._OLD_KEYS:
            return self[self._OLD_KEYS[key]]
        raise KeyError(key)


@logged
def match_mon_to_pol(monomer_dict, polymer_dict, compare_any_bond = False,
                     match_residue_number = None, monomer_key = True, timeout = 3,
                     only_heavy_mapping = True):
    """
    Аргументы:
        monomer_dict - словарь мономера (состоит из одной пары имя: rdkit.Chem)
        polymer_dict - словарь полимера (состоит из одной пар имя: rdkit.Chem)
        only_heavy_mapping - MCS только по тяжелым атомам (H сопоставляются после)
    Возвращает:
        Словарь match_data_dict со следующими подсловарями:
        - substructure - сожердит общие подструктуры мономера и полимера (с H), для каждого из полимеров
        - substructure_no_H - те же подструктуры без H
        - mon_pol_matches - содержит словари соответствия номеров атомов в общей подструктуре
        с номерами атомов в полимере
        Пары, для которых сопоставление не найдено, пропускаются.
    """
    def add_match(key, mon_val, pol_val):
        sub, dict_mon_pol_matches = match_chem_v6(mon_val, pol_val,
                                                  only_heavy_mapping = only_heavy_mapping,
                                                  compare_any_bond = compare_any_bond,
                                                  match_residue_number = match_residue_number,
                                                  timeout=timeout
                                                  )
        if sub is None:
            print_red(f'Сопоставление для {key} не найдено, пропускаем.')
            return

        match_data_dict['substructure'][key] = sub['H']
        match_data_dict['substructure_no_H'][key] = sub['noH']
        match_data_dict['N_match_atoms'][key] = sub['H'].GetNumAtoms()
        match_data_dict['mon_pol_matches'][key] = dict_mon_pol_matches['Indexes']
        match_data_dict['mon_pol_atom_names'][key] = dict_mon_pol_matches.get('Names')

    try:
        match_data_dict = _MatchData({'substructure': {},
                           'substructure_no_H': {},
                           'mon_pol_matches': {},
                           'N_match_atoms': {},
                           'mon_pol_atom_names':{}})

        if (len(monomer_dict) >= 1 and len(polymer_dict) == 1) and monomer_key is True:

            pol_key, pol_val = next(iter(polymer_dict.items()))

            for mon_key, mon_val in monomer_dict.items():
                add_match(mon_key, mon_val, pol_val)

        elif len(monomer_dict) == 1 and len(polymer_dict) > 1 or monomer_key is False:
            mon_key, mon_val = next(iter(monomer_dict.items()))

            for pol_key, pol_val in polymer_dict.items():
                add_match(pol_key, mon_val, pol_val)

        else:
            print('The len of monomer_dict or polymer_dict must be 1.')
            
        return match_data_dict
    except Exception as e:
        raise e
        # print_red(f'Что-то пошло не так!\n{e}')
        
# Наборы референсных остатков для поиска канонических атомов.
# Шаблон - PDB-файл, где референсный остаток стоит в позиции residue_number
# (для белков - тримеры GXG_H.pdb), порядок его атомов в файле считается каноническим.
# Для нуклеиновых кислот достаточно добавить сюда запись со своей папкой шаблонов.
REF_TEMPLATES = {
    'protein': {
        'dir': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'molecules', 'aminoacids_template'),
        'pattern': '*_H.pdb',
        'residue_number': 2,
        # атомы остова: если они не сопоставились, референс выбран неверно
        'backbone': ['N', 'CA', 'C', 'O'],
    },
}

# Канонические аминокислоты: однобуквенный код -> (трехбуквенный код, название, name)
AMINO_ACIDS = {
    'A': ('Ala', 'аланин', 'alanine'),
    'C': ('Cys', 'цистеин', 'cysteine'),
    'D': ('Asp', 'аспарагиновая кислота', 'aspartate'),
    'E': ('Glu', 'глутаминовая кислота', 'glutamate'),
    'F': ('Phe', 'фенилаланин', 'phenylalanine'),
    'G': ('Gly', 'глицин', 'glycine'),
    'H': ('His', 'гистидин', 'histidine'),
    'I': ('Ile', 'изолейцин', 'isoleucine'),
    'K': ('Lys', 'лизин', 'lysine'),
    'L': ('Leu', 'лейцин', 'leucine'),
    'M': ('Met', 'метионин', 'methionine'),
    'N': ('Asn', 'аспарагин', 'asparagine'),
    'P': ('Pro', 'пролин', 'proline'),
    'Q': ('Gln', 'глутамин', 'glutamine'),
    'R': ('Arg', 'аргинин', 'arginine'),
    'S': ('Ser', 'серин', 'serine'),
    'T': ('Thr', 'треонин', 'threonine'),
    'V': ('Val', 'валин', 'valine'),
    'W': ('Trp', 'триптофан', 'tryptophan'),
    'Y': ('Tyr', 'тирозин', 'tyrosine'),
}
aa_full_names = {one: names[2] for one, names in AMINO_ACIDS.items()}


def aa_label(letter):
    """'C' -> 'цистеин (Cys, C)'. Для неизвестного кода возвращает сам код."""
    if letter not in AMINO_ACIDS:
        return letter
    three, name_ru, _ = AMINO_ACIDS[letter]
    return f'{name_ru} ({three}, {letter})'


def template_aa_letter(template_name):
    """Однобуквенный код остатка шаблона: 'GCG_H' -> 'C', 'AGA_H' -> 'G'."""
    core = template_name.split('_')[0]
    return core[1] if len(core) == 3 and core[1] in AMINO_ACIDS else None


def template_label(template_name):
    """'GCG_H' -> 'цистеин (Cys, C), шаблон GCG_H'."""
    letter = template_aa_letter(template_name)
    return f'{aa_label(letter)}, шаблон {template_name}' if letter else template_name


def resolve_ref_template(ref_base_name, template_paths, residue_type='protein'):
    """
    Находит файл шаблона по имени референсного остатка.

    Аргументы:
        ref_base_name (str) - имя остатка: однобуквенное ('C'), трехбуквенное ('Cys'),
            полное ('cysteine') или имя файла шаблона ('GCG_H')
        template_paths (list) - пути к файлам шаблонов
        residue_type (str) - тип остатка из REF_TEMPLATES
    Возвращает:
        str - путь к файлу шаблона
    """
    by_name = {os.path.basename(path).split('.')[0].upper(): path for path in template_paths}
    key = ref_base_name.strip().upper()
    if key in by_name:
        return by_name[key]

    if residue_type == 'protein':
        letter = None
        if len(key) == 1 and key in AMINO_ACIDS:
            letter = key
        else:
            for one, names in AMINO_ACIDS.items():
                if key in (name.upper() for name in names):
                    letter = one
                    break
        if letter is not None and f'G{letter}G_H' in by_name:
            return by_name[f'G{letter}G_H']
    else:
        # Для остальных типов ищем шаблон, в имени которого есть ref_base_name
        found = [path for name, path in by_name.items() if key in name]
        if len(found) == 1:
            return found[0]

    raise ValueError(f'Шаблон для остатка "{ref_base_name}" не найден среди: {sorted(by_name)}')


@logged
def check_parent_residue(ref_mol, mapping, residue_number, backbone, template_name, strict=True):
    """
    Проверяет, что модификация действительно построена на заявленном родительском остатке.

    Остов (backbone) и CB должны быть сопоставлены полностью, иначе ValueError
    (при strict=False - предупреждение).
    Несопоставленные атомы боковой цепи печатаются: обычно это место модификации
    (NZ ацетиллизина, NH2 цитруллина - один атом). Если не сопоставлена вся боковая цепь
    или не меньше 2 атомов и половины цепи, родитель скорее всего указан неверно:
    при strict=True это ValueError, при strict=False - предупреждение
    (для модификаций, сильно перестраивающих боковую цепь, например кинуренина из Trp).

    Ограничение: Ala и Gly содержатся почти в любом остатке, поэтому ошибочно
    заявленный Ala/Gly проверкой не ловится.

    Аргументы:
        ref_mol (Chem.Mol) - шаблон
        mapping (dict) - {индекс в модификации: индекс в шаблоне}
        residue_number (int) - номер родительского остатка в шаблоне
        backbone (list) - имена атомов остова
        template_name (str) - имя шаблона для сообщений
    """
    residue_atoms = {atom.GetIdx(): atom.GetProp('AtomName') for atom in ref_mol.GetAtoms()
                     if atom.GetAtomicNum() > 1
                     and atom.GetPDBResidueInfo().GetResidueNumber() == residue_number}
    matched = set(mapping.values())
    required = list(backbone) + (['CB'] if 'CB' in residue_atoms.values() else [])

    missing_required = [name for idx, name in residue_atoms.items()
                        if name in required and idx not in matched]
    if missing_required:
        message = (f'Модификация не содержит остов остатка {template_label(template_name)}: '
                   f'не сопоставлены {missing_required}. Проверьте ref_base_name.')
        if strict:
            raise ValueError(message)
        print_red(f'⚠ {message}')

    side_chain = [name for name in residue_atoms.values() if name not in required]
    missing_side = [name for idx, name in residue_atoms.items()
                    if name in side_chain and idx not in matched]
    if not missing_side:
        return
    report = f'атомы боковой цепи {missing_side} ({len(missing_side)} из {len(side_chain)})'
    wrong_parent = (len(missing_side) == len(side_chain)
                    or (len(missing_side) >= 2 and 2 * len(missing_side) >= len(side_chain)))
    if wrong_parent and strict:
        raise ValueError(
            f'Не сопоставлены {report} остатка {template_label(template_name)}: '
            'похоже, родительский остаток указан неверно. Проверьте ref_base_name; '
            'если модификация сильно перестраивает боковую цепь, вызовите с strict_parent=False.')
    if wrong_parent:
        print_red(f'⚠ Не сопоставлены {report}: возможно, родительский остаток указан неверно.')
    else:
        print_red(f'⚠ Не сопоставлены {report}: вероятно, это место модификации.')


@logged
def find_ref_residue(mod_mol_dict, path_to_ref_mol=None, ref_base_name=None,
                     residue_type='protein', match_residue_number=None,
                     only_heavy_mapping=True, main_match_data=True, timeout=3,
                     strict_parent=True, type_match_dict=None):
    """
    Сопоставляет модифицированный остаток с референсными остатками и выбирает референс.

    Аргументы:
        mod_mol_dict (dict) - словарь {имя: Chem.Mol} модифицированного остатка (одна пара)
        path_to_ref_mol (str) - папка с шаблонами или путь к одному шаблону;
            по умолчанию папка из REF_TEMPLATES[residue_type]
        ref_base_name (str) - родительский остаток, из которого собрана модификация:
            'C', 'Cys', 'cysteine', 'цистеин' или имя шаблона 'GCG_H' (см. resolve_ref_template).
            Нужно указывать всегда: по числу совпавших атомов родителя не определить
            (у меток цепи и кольца совпадают с Lys/Trp больше, чем с Cys).
            Старые ноутбуки вызывают функцию без него: тогда шаблон выбирается автоматически,
            как раньше, а все проверки родителя выдают предупреждения вместо ошибок.
        residue_type (str) - тип остатка из REF_TEMPLATES ('protein', ...)
        match_residue_number (int) - номер референсного остатка в шаблоне;
            по умолчанию REF_TEMPLATES[residue_type]['residue_number']
        only_heavy_mapping (bool) - MCS только по тяжелым атомам (быстрее)
        main_match_data (bool) - вернуть данные только для выбранного шаблона
        timeout (int) - таймаут поиска MCS для каждого шаблона, с
        strict_parent (bool) - ошибка, если боковая цепь родителя в основном не сопоставлена
            (см. check_parent_residue); False - только предупреждение
        type_match_dict - устаревший параметр старых ноутбуков, ни на что не влияет
    Возвращает:
        ref_chem_dict (dict) - {имя шаблона: Chem.Mol}
        match_data_dict (dict) - словарь match_mon_to_pol, ключи - имена шаблонов
        ref_name (str) - имя выбранного шаблона-триплета, например 'GKG_H'
    """
    template = REF_TEMPLATES[residue_type]
    if match_residue_number is None:
        match_residue_number = template['residue_number']

    path_to_ref_mol = path_to_ref_mol or template['dir']
    if os.path.isdir(path_to_ref_mol):
        template_paths = sorted(glob.glob(os.path.join(path_to_ref_mol, template['pattern'])))
    else:
        template_paths = [path_to_ref_mol]
    if not template_paths:
        raise FileNotFoundError(f'Шаблоны не найдены в {path_to_ref_mol}')

    if type_match_dict is not None:
        print_red('⚠ Параметр type_match_dict устарел и ни на что не влияет: '
                  'имена атомов всегда возвращаются в match_data_dict["mon_pol_atom_names"].')

    if ref_base_name:
        template_paths = [resolve_ref_template(ref_base_name, template_paths, residue_type)]
    elif len(template_paths) > 1:
        # Старый вызов без родителя: работаем как раньше, но проверки только предупреждают
        strict_parent = False
        print_red('⚠ Родительский остаток не указан (ref_base_name): шаблон будет выбран '
                  'по числу совпавших атомов, как в старых версиях. Результат ненадёжен: '
                  'для меток на цистеине так выбирается Lys или Trp. '
                  "Укажите родителя, например ref_base_name='C' или 'K'.")

    ref_chem_dict = pdb_to_chem(template_paths)
    match_data_dict = match_mon_to_pol(mod_mol_dict, ref_chem_dict,
                                       match_residue_number=match_residue_number,
                                       monomer_key=False, timeout=timeout,
                                       only_heavy_mapping=only_heavy_mapping)
    if not match_data_dict['N_match_atoms']:
        raise ValueError('Не найдено сопоставление ни с одним из шаблонов.')

    max_name = max(match_data_dict['N_match_atoms'], key=match_data_dict['N_match_atoms'].get)
    max_val = match_data_dict['N_match_atoms'][max_name]
    print(f'Родительский остаток: {template_label(max_name)}; сопоставлено атомов: {max_val}.')

    check_parent_residue(ref_chem_dict[max_name], match_data_dict['mon_pol_matches'][max_name],
                         match_residue_number, template.get('backbone', []), max_name,
                         strict=strict_parent)

    if main_match_data:
        match_data_dict = {key: {max_name: val[max_name]} for key, val in match_data_dict.items()}
    return {max_name: ref_chem_dict[max_name]}, match_data_dict, max_name


# Старое имя, используется в ноутбуках
find_ref_aa = find_ref_residue


@logged
def renumber_residue_atoms(mol, ref_base_name=None, residue_type='protein',
                           path_to_ref_mol=None, only_heavy_mapping=True, timeout=3,
                           strict_parent=True):
    """
    Перенумеровывает атомы остатка: основание (атомы родительского остатка) - в порядке
    шаблона, остальные атомы - обходом молекулярного графа от основания.

    1. В остатке ищется родительский остаток (find_ref_residue); его атомы получают номера
       в том порядке, в котором стоят в шаблоне (amber: N H CA HA CB HB1 HB2 ... C O).
    2. Новые атомы вставляются сразу за каноническим тяжёлым атомом, к которому пришиты
       (и за его водородами): для метки на Cys - N H CA HA CB HB1 HB2 SG [метка] C O,
       для N-метилирования - N H [метил] CA ...
    3. Внутри вставки - обход в ширину, каждый водород сразу за своим тяжёлым атомом.
       Порядок ветвей задаётся каноническим рангом RDKit без учёта стереохимии, поэтому
       результат не зависит от исходного порядка атомов (например, от того, как был записан
       SMILES) и от R/S стереоцентров.
    Если индексы уже в таком порядке, возвращается копия молекулы без изменений.

    Аргументы:
        mol (Chem.Mol) - остаток (с H или без H)
        ref_base_name (str) - родительский остаток (см. find_ref_residue), обязателен
        residue_type (str) - тип остатка из REF_TEMPLATES ('protein', ...)
        path_to_ref_mol (str) - папка с шаблонами или путь к шаблону
        only_heavy_mapping (bool) - MCS только по тяжелым атомам
        timeout (int) - таймаут поиска MCS, с
        strict_parent (bool) - см. find_ref_residue
    Возвращает:
        Chem.Mol - перенумерованная молекула. Свойства: 'RefResidue' - имя шаблона,
        'OldIndices' - JSON-список: на позиции нового индекса стоит старый индекс атома,
        'ParentAtoms' - JSON-список новых индексов атомов, сопоставленных с шаблоном
        родительского остатка (для подсветки: draw_molecule(highlight_atoms=...)).
    """
    _, match_data, _ = find_ref_residue({'residue': mol}, path_to_ref_mol=path_to_ref_mol,
                                     ref_base_name=ref_base_name, residue_type=residue_type,
                                     only_heavy_mapping=only_heavy_mapping, timeout=timeout,
                                     strict_parent=strict_parent)
    ref_name, mapping = next(iter(match_data['mon_pol_matches'].items()))

    # канонический ранг атомов: не зависит от исходной нумерации и от стереохимии
    # (стереохимия, прочитанная из случайной 3D-структуры, меняла порядок при каждом запуске)
    rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True, includeChirality=False))
    placed = set(mapping)

    def by_rank(atoms):
        return sorted(atoms, key=lambda atom: rank[atom.GetIdx()])

    def new_block(anchor_idx):
        """Новые атомы, пришитые к каноническому атому anchor_idx: обход в ширину."""
        block = []
        anchor = mol.GetAtomWithIdx(anchor_idx)
        # лишние водороды самого канонического атома (например, у изменённой NZ)
        for h in by_rank(anchor.GetNeighbors()):
            if h.GetAtomicNum() == 1 and h.GetIdx() not in placed:
                placed.add(h.GetIdx())
                block.append(h.GetIdx())
        queue = [a.GetIdx() for a in by_rank(anchor.GetNeighbors())
                 if a.GetAtomicNum() > 1 and a.GetIdx() not in placed]
        placed.update(queue)
        while queue:
            idx = queue.pop(0)
            block.append(idx)
            atom = mol.GetAtomWithIdx(idx)
            for nbr in by_rank(atom.GetNeighbors()):
                if nbr.GetIdx() in placed:
                    continue
                placed.add(nbr.GetIdx())
                if nbr.GetAtomicNum() == 1:
                    block.append(nbr.GetIdx())
                else:
                    queue.append(nbr.GetIdx())
        return block

    # канонические атомы в порядке шаблона; вставка новых атомов - перед следующим
    # каноническим тяжёлым атомом, т.е. после водородов своего якоря
    canonical = sorted(mapping, key=mapping.get)
    new_order, anchor = [], None
    for idx in canonical:
        if mol.GetAtomWithIdx(idx).GetAtomicNum() > 1:
            if anchor is not None:
                new_order += new_block(anchor)
            anchor = idx
        new_order.append(idx)
    if anchor is not None:
        new_order += new_block(anchor)

    leftover = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIdx() not in placed]
    if leftover:
        print_red(f'⚠ Атомы {leftover} не связаны с основанием остатка и поставлены в конец.')
        new_order += leftover

    if new_order == list(range(mol.GetNumAtoms())):
        print_green('Индексы атомов уже канонические, перенумерация не требуется.')
        new_mol = Chem.Mol(mol)
    else:
        moved = [(old, new) for new, old in enumerate(new_order) if old != new]
        print(f'Перенумеровано атомов: {len(moved)} из {mol.GetNumAtoms()} '
              f'(канонических: {len(canonical)})')
        new_mol = Chem.RenumberAtoms(mol, new_order)

    new_mol.SetProp('RefResidue', ref_name)
    new_mol.SetProp('OldIndices', json.dumps(new_order))
    new_index = {old: new for new, old in enumerate(new_order)}
    new_mol.SetProp('ParentAtoms', json.dumps(sorted(new_index[idx] for idx in canonical)))
    return new_mol


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

def path_parser(path: str, file_types: list):
    """
    Парсит путь и возвращает путь к директории и имя файла без расширений из списка file_types.
    """
    # Форматирование расширений, добавление '.' если отсутствует
    formated_file_types = ['.' + t if not t.startswith('.') else t for t in file_types]
    
    # Разделение пути и имени файла
    path_to_file, file_name = os.path.split(path)
    if path_to_file:
        os.makedirs(path_to_file, exist_ok=True)
    # else:
    #     path_to_file = 'current directory'

    # Удаление расширения из списка в конце имени файла. Раньше расширения вырезались
    # replace'ом в любом месте имени: 'X_rn_H.smiles' с ['smi', 'smiles'] давало 'X_rn_Hles'
    for type_name in sorted(formated_file_types, key=len, reverse=True):
        if file_name.endswith(type_name):
            file_name = file_name[:-len(type_name)]
            break

    return path_to_file, file_name
        
def find_amino_nitrogen(mol):
    """
    Находит индекс атома азота в составе аминокислоты.
    Предполагается, что любая а.к. имеет вид основу [H][N]CC=O.
    """
    aa_base = '[H][N]CC=O'
    sub_base = mol.GetSubstructMatch(Chem.MolFromSmarts(aa_base))
    if sub_base:
        print(f"atom number: {sub_base[1]} became the root atom: 0")
        return sub_base[1]
    return -1

def _with_atom_map(mol):
    """Копия молекулы, в которой номер атома в SMILES = индекс + 1 (0 в SMILES - «без номера»)."""
    mapped = Chem.Mol(mol)
    for atom in mapped.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    return mapped


def _reroot_via_smiles(rdkit_mol, rootedAtAtom, func_name, sanitize):
    """
    Старое поведение rootedAtAtom в save_chem_to_pdb / save_chem_to_mol: молекула
    переписывается через SMILES с корнем в атоме rootedAtAtom. Индексы атомов меняются,
    имена атомов, параметры остатка, свойства молекулы и координаты теряются.
    """
    print_red(f'⚠ {func_name}(rootedAtAtom={rootedAtAtom}) устарел: атомы переставляются через '
              'SMILES, индексы меняются, имена атомов и параметры остатка теряются. '
              'Для порядка атомов остатка используйте renumber_residue_atoms.')
    smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True,
                                     allHsExplicit=True, rootedAtAtom=rootedAtAtom)
    return Chem.MolFromSmiles(smiles_string, sanitize=sanitize)


@logged
def save_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, rootedAtAtom=-1,
                        atom_map=False):
    """
    Сохраняет молекулу в SMILES (<path>.smiles). Молекула не меняется.

    Аргументы:
        rdkit_mol (Chem.Mol) - молекула
        path (str) - путь к файлу; расширение .smi/.smiles можно не писать
        canonical (bool) - канонический порядок обхода графа (True) или порядок индексов (False)
        allHsExplicit (bool) - писать водороды в квадратных скобках у каждого атома
        rootedAtAtom (int) - атом, с которого начинается запись строки (-1 - по умолчанию).
            Меняет только текст строки, не молекулу
        atom_map (bool) - записать у каждого атома его номер (индекс + 1, т.к. 0 в SMILES -
            «без номера»); smi_to_chem по этим номерам восстановит исходные индексы
    """
    path_to_file, file_name = path_parser(path, ['smi', 'smiles'])
    smiles_string = Chem.MolToSmiles(_with_atom_map(rdkit_mol) if atom_map else rdkit_mol,
                                     canonical=canonical,
                                     allHsExplicit=allHsExplicit,
                                     rootedAtAtom=rootedAtAtom)
    output_path = os.path.join(path_to_file, f'{file_name}.smiles')
    with open(output_path, "w") as file:
        file.write(smiles_string)
    print(f'Сохранено: {output_path}')


@logged
def save_aa_chem_to_smiles(rdkit_mol, path, canonical=True, allHsExplicit=True, make_N_root=False,
                           atom_map=False):
    """
    Старое имя save_chem_to_smiles (параметры те же, порядок позиционных аргументов прежний).
    make_N_root=True - строка начинается с азота аминогруппы (find_amino_nitrogen);
    меняется только текст строки, не молекула.
    """
    rootedAtAtom = find_amino_nitrogen(rdkit_mol) if make_N_root else -1
    save_chem_to_smiles(rdkit_mol, path, canonical=canonical, allHsExplicit=allHsExplicit,
                        rootedAtAtom=rootedAtAtom, atom_map=atom_map)


@logged
def save_chem_to_mol(rdkit_mol, path, rootedAtAtom=-1):
    """
    Сохраняет молекулу в формате MOL (<path>.mol).
    rootedAtAtom - устаревший параметр (см. _reroot_via_smiles): функция предупреждает
    и переставляет атомы, как раньше.
    """
    if rootedAtAtom != -1:
        rdkit_mol = _reroot_via_smiles(rdkit_mol, rootedAtAtom, 'save_chem_to_mol', sanitize=True)

    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True)
    with open(f"{path}.mol", "w") as file:
        file.write(Chem.MolToMolBlock(rdkit_mol))


@logged
def save_chem_to_pdb(rdkit_mol, path, rootedAtAtom=-1, coords='2D', stereo=None, ca_config='L',
                     stereo_default='R', random_seed=42):
    """
    Сохраняет молекулу в PDB (<path>.pdb) с её индексацией атомов. Если у атомов есть
    PDB-информация (имена атомов, остаток - после rdkit_pdb_modification или из PDB-файла),
    она записывается; иначе RDKit пишет свои имена (элемент + номер), остаток UNL, HETATM.
    Переданная молекула не меняется (координаты задаются на копии).

    Аргументы:
        rdkit_mol (Chem.Mol) - молекула
        path (str) - путь без расширения
        rootedAtAtom (int) - устаревший параметр (см. _reroot_via_smiles): функция
            предупреждает и переставляет атомы, как раньше
        coords (str) - '2D' (по умолчанию) или '3D'; stereo, ca_config, stereo_default,
            random_seed - параметры 3D-структуры и стереохимии, см. set_mol_coords.
            До 2026-09 функция всегда строила 3D-структуру со случайной стереохимией.
    """
    rdkit_mol = Chem.Mol(rdkit_mol)
    if rootedAtAtom != -1:
        rdkit_mol = _reroot_via_smiles(rdkit_mol, rootedAtAtom, 'save_chem_to_pdb', sanitize=False)
    failed = Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL, catchErrors=True)
    if failed != Chem.SanitizeFlags.SANITIZE_NONE:
        print_red(f'⚠ Санитизация не прошла ({failed}): молекула сохраняется как есть.')
    path_dir = path.rsplit('/', 1)
    if len(path_dir) > 1:
        os.makedirs(path_dir[0], exist_ok=True)
    set_mol_coords(rdkit_mol, coords=coords, stereo=stereo, ca_config=ca_config,
                   stereo_default=stereo_default, random_seed=random_seed)
    pdb_block = Chem.MolToPDBBlock(rdkit_mol)
    with open(f"{path}.pdb", "w") as file:
        file.write(pdb_block)

@logged
def generate_atom_names_by_ref_aa(mod_aa_mol, ref_aa_mol, dict_match, output_path):
    """
    Переименовывает атомы в модифицированной аминокислоте на основе референсной.
    
    Аргументы:
        mod_aa_mol: RDKit объект Chem.Mol для модифицированной молекулы.
        ref_aa_mol: RDKit объект Chem.Mol для референсной молекулы.
        dict_match: Словарь сопоставления индексов атомов (модифицированные -> референсные).
        output_path: Путь для сохранения модифицированного PDB файла.
    """
    # Получаем текущие имена атомов модифицированной молекулы
    all_mod_atom_names = [str(atom.GetProp("AtomName")) if atom.HasProp("AtomName") else "" for atom in mod_aa_mol.GetAtoms()]
    all_ref_atom_names = [str(atom.GetProp("AtomName")) if atom.HasProp("AtomName") else "" for atom in ref_aa_mol.GetAtoms()]
    print(all_mod_atom_names)

    for mod_indx, ref_indx in dict_match.items():
        mod_atom_name = all_mod_atom_names[mod_indx]
        ref_atom_name = all_ref_atom_names[ref_indx]

        # Если имя из референсной молекулы уже есть в модифицированной
        if ref_atom_name in all_mod_atom_names:
            new_name = ref_atom_name
            counter = 1
            while new_name in all_mod_atom_names:
                # Изменяем только числовую часть имени
                match = re.match(r"(\D+)(\d*)", ref_atom_name)
                if match:
                    prefix = match.group(1)  # Буквенная часть
                    suffix = match.group(2)  # Числовая часть (может быть пустой)
                    new_name = f"{prefix}{int(suffix) + counter if suffix else counter}"
                else:
                    # Если имя не содержит числовой части, добавляем ее
                    new_name = f"{ref_atom_name}{counter}"
                counter += 1

            print(f'Переименовываем {mod_indx}:{ref_atom_name} -> {mod_indx}:{new_name}')
            index_rename_mod_atom = all_mod_atom_names.index(ref_atom_name)
            all_mod_atom_names[index_rename_mod_atom] = new_name
            all_mod_atom_names[mod_indx] = ref_atom_name
        else:
            # Если имени нет в модифицированной молекуле, обновляем напрямую
            all_mod_atom_names[mod_indx] = ref_atom_name


        print(f'({mod_indx}:{mod_atom_name}) -> ({ref_indx}:{ref_atom_name})')

    # Проверяем, что все имена уникальны
    name_counts = Counter(all_mod_atom_names)
    duplicates = [name for name, count in name_counts.items() if count > 1]

    if duplicates:
        print("Обнаружены повторяющиеся имена атомов в модифицированной молекуле:")
        for name in duplicates:
            print(f"Имя: {name}, количество повторений: {name_counts[name]}")
        raise ValueError("Есть повторяющиеся имена атомов. Проверьте логи.")
    else:
        print("Все имена атомов уникальны.")

    return all_mod_atom_names        


def rdkit_pdb_modification_old(rdkit_mol, resname='MOD', resid=1, segid='A'):
    """
    Модифицирует имена атомов в молекуле по правилам аминокислот:
    - N, H, CA, C, O имеют стандартные имена
    - Боковые атомы получают буквенные метки по греческому алфавиту
    - Протоны наследуют имя родительского атома и получают числовой суффикс
    """
    greek_alphabet = {0: 'B', 1: 'G', 2: 'D', 3: 'E', 4: 'Z', 5: 'H', 
                      6: 'T', 7: 'I', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 
                      12: 'X', 13: 'O', 14: 'P', 15: 'R', 16: 'S'}

    # Найдём индекс CA (альфа-углерода)
    submatch = rdkit_mol.GetSubstructMatch(Chem.MolFromSmarts('[H][N]C([H])C=O'))
    if not submatch:
        raise ValueError("Не удалось найти структуру аминокислоты [H][N]CC=O")
    idx_H, idx_N, idx_CA, idx_HA ,idx_C, idx_O = submatch[0], submatch[1], submatch[2], submatch[3], submatch[4], submatch[5] 

    # Префиксы по индексам атомов
    atom_names = {idx_H: {'atom_symbol': 'H','atom_letter':''}, idx_N: {'atom_symbol': 'N','atom_letter':''}, 
                  idx_CA: {'atom_symbol': 'C','atom_letter':'A'}, idx_HA: {'atom_symbol': 'H','atom_letter':'A'},
                  idx_C: {'atom_symbol': 'C','atom_letter':''}, idx_O: {'atom_symbol': 'O','atom_letter':''}}
    # Старт нумерации с атомов, следующих за CA
    # visited = set(atom_names.keys())
    # atom_names = {}
    greek_counter = 0
    hydrogen_counts = {}

    queue = [idx_CA]
    while queue:
        current_idx = queue.pop(0)
        current_atom = rdkit_mol.GetAtomWithIdx(current_idx)
        neighbors = defaultdict(int)

        for atom in current_atom.GetNeighbors():
            if atom.GetIdx() not in atom_names:
                neighbors[atom.GetAtomicNum()] += 1

        
        count_hidrogen = neighbors[1] 
        count_heavy = sum(neighbors.values()) - neighbors.get(1, 0) 

        hidrogen_index = 1 if count_hidrogen > 1 else None
        heavy_index = 1 if count_heavy > 1 else None
            
        for neighbor in current_atom.GetNeighbors():
            nbr_idx = neighbor.GetIdx()
            if nbr_idx in atom_names.keys():
                continue
                
            if neighbor.GetAtomicNum() != 1:
                # Тяжёлый атом
                symbol = neighbor.GetSymbol()
                if heavy_index and count_heavy > 1:
                    greek_index = greek_alphabet.get(greek_counter, f"Y{greek_counter}") + str(count_heavy)
                    atom_names[nbr_idx] = {'atom_symbol': symbol,'atom_letter': greek_index}
                    count_heavy -= 1
                    queue.append(nbr_idx)
                elif heavy_index and count_heavy == 1:
                    greek_index = greek_alphabet.get(greek_counter, f"Y{greek_counter}") + str(count_heavy)
                    atom_names[nbr_idx] = {'atom_symbol': symbol,'atom_letter': greek_index}
                    greek_counter += 1
                    queue.append(nbr_idx)
                else:
                    greek_index = greek_alphabet.get(greek_counter, f"Y{greek_counter}")
                    atom_names[nbr_idx] = {'atom_symbol': symbol,'atom_letter': greek_index}
                    greek_counter += 1
                    queue.append(nbr_idx)
            else :
                # Это водород — имя зависит от родителя
                parent_letter = atom_names[current_idx]['atom_letter']
                if hidrogen_index and count_hidrogen > 1:
                    greek_index = parent_letter + str(count_hidrogen)
                    atom_names[nbr_idx] = {'atom_symbol': 'H','atom_letter': greek_index}
                    count_hidrogen -= 1
                elif hidrogen_index and count_hidrogen == 1:
                    greek_index = parent_letter + str(count_hidrogen)
                    atom_names[nbr_idx] = {'atom_symbol': 'H','atom_letter': greek_index}
                    count_hidrogen -= 1
                else:
                    atom_names[nbr_idx] = {'atom_symbol': 'H','atom_letter': parent_letter} #f"H{parent_name}{hydrogen_counts[parent_name]}"

    # Объединяем всё
    for atom in rdkit_mol.GetAtoms():
        idx = atom.GetIdx()
        name = atom_names[idx]['atom_symbol']+atom_names[idx]['atom_letter']
        name = name[:4].ljust(4)  # PDB формат требует длину 4 символа

        info = Chem.AtomPDBResidueInfo()
        info.SetName(name)
        info.SetResidueName(resname)
        info.SetResidueNumber(resid)
        info.SetChainId(segid)
        atom.SetProp("AtomName", name.strip())
        atom.SetMonomerInfo(info)

    return rdkit_mol  

def find_backbone_match(rdkit_mol, C_terminal=False):
    """
    Находит атомы остова аминокислотного остатка по шаблону H-N-CA(H)-C(=O)
    (тот же, что в rdkit_pdb_modification_old). Водороды должны быть явными атомами.

    Аргументы:
        rdkit_mol (Chem.Mol) - остаток
        C_terminal (bool) - C-концевой остаток: у C два кислорода (OC1 - двойная связь, OC2 - одинарная)
    Возвращает:
        (CA_idx, backbone) - backbone = [N, CA, C, O] или [N, CA, C, OC1, OC2] при C_terminal
    Если подходящих остовов нет или их больше одного, вызывает ValueError: выбор был бы угадыванием.
    """
    smarts = '[H][N][C]([H])C(=O)[O]' if C_terminal else '[H][N][C]([H])C=O'
    matches = rdkit_mol.GetSubstructMatches(Chem.MolFromSmarts(smarts), uniquify=False)
    # совпадения, отличающиеся только выбором H при N и CA, - это один и тот же остов
    backbones = sorted({(m[1], m[2], m[4], m[5], m[6]) if C_terminal else (m[1], m[2], m[4], m[5])
                        for m in matches})
    if not backbones:
        raise ValueError(f"Остов аминокислоты ({smarts}) не найден: проверьте, что водороды явные "
                         f"и что C_terminal={C_terminal} соответствует остатку.")
    if len(backbones) > 1:
        raise ValueError(f"Найдено {len(backbones)} возможных остовов (индексы N, CA, C, O...): "
                         f"{backbones}. Остов неоднозначен.")
    backbone = list(backbones[0])
    return backbone[1], backbone


GREEK_LEVELS = 'BGDEZHTIKLMN'  # буква уровня боковой цепи: 1 связь от CA - B, 2 - G, 3 - D, ...
NAMING_METHODS = ('greek', 'index')


class _NamingError(ValueError):
    """Буквенный способ неприменим к остатку (слишком много уровней, длинные имена)."""


def _greek_heavy_names(mol, CA_idx, heavy_atoms):
    """
    Буквенные имена тяжёлых атомов боковой цепи: элемент + буква уровня (число связей от CA)
    + номер, если на уровне несколько атомов (1..n в порядке индексов).
    heavy_atoms - атомы, которые нужно назвать; путь от CA идёт только через них.
    Возвращает ({индекс: имя}, [индексы, не связанные с CA через heavy_atoms]).
    """
    depth = {CA_idx: 0}
    queue = deque([CA_idx])
    while queue:
        idx = queue.popleft()
        for nbr in sorted(a.GetIdx() for a in mol.GetAtomWithIdx(idx).GetNeighbors()):
            if nbr in heavy_atoms and nbr not in depth:
                depth[nbr] = depth[idx] + 1
                queue.append(nbr)
    levels = defaultdict(list)
    for idx, d in depth.items():
        if idx != CA_idx:
            levels[d].append(idx)
    if levels and max(levels) > len(GREEK_LEVELS):
        raise _NamingError(f'боковая цепь длиннее {len(GREEK_LEVELS)} уровней '
                           f'(самый дальний атом в {max(levels)} связях от CA)')
    names = {}
    for d in sorted(levels):
        atoms = sorted(levels[d])
        for k, idx in enumerate(atoms, 1):
            element = mol.GetAtomWithIdx(idx).GetSymbol().upper()
            names[idx] = f'{element}{GREEK_LEVELS[d - 1]}' + (str(k) if len(atoms) > 1 else '')
    unreached = sorted(set(heavy_atoms) - set(depth))
    return names, unreached


def _hydrogen_names(mol, heavy_names, hydrogens):
    """
    Имена водородов при названных тяжёлых атомах: H + имя атома без элемента, при нескольких
    водородах - ещё номер: N -> H (H1 H2 H3), CA -> HA, CB -> HB1 HB2, SG -> HG, CD1 -> HD11 HD12.
    hydrogens - водороды, которые нужно назвать.
    """
    names = {}
    for idx, name in heavy_names.items():
        atom = mol.GetAtomWithIdx(idx)
        suffix = name[len(atom.GetSymbol()):]
        hs = sorted(h.GetIdx() for h in atom.GetNeighbors()
                    if h.GetAtomicNum() == 1 and h.GetIdx() in hydrogens)
        for k, h_idx in enumerate(hs, 1):
            names[h_idx] = f'H{suffix}' + (str(k) if len(hs) > 1 else '')
    return names


def _pdb_atom_name_field(name, element):
    """Поле имени атома PDB (колонки 13-16): однобуквенный элемент начинается с 14-й колонки."""
    if len(name) < 4 and len(element) == 1:
        return f' {name:<3}'
    return f'{name:<4}'


@logged
def rdkit_pdb_modification(rdkit_mol, resname="MOD", resid=1, segid="A",
                           force_numeric=False, C_terminal=False, naming='greek', parent_atoms=None):
    """
    Назначает атомам остатка PDB-имена и параметры остатка (имя, номер, цепь).
    Меняет и возвращает переданную молекулу (save_aa_chem_to_pdb передаёт сюда копию).
    Имена строятся только по графу молекулы и индексам атомов; старые имена не используются,
    поэтому повторный вызов на той же молекуле даёт те же имена.

    Остов (find_backbone_match): N, CA, C, O (C-концевой остаток: OC1, OC2).

    naming='greek' - буквенный способ (по умолчанию), как в amber14sb.
        Тяжёлые атомы боковой цепи делятся на уровни по числу связей от CA:
        1 - B, 2 - G, 3 - D, 4 - E, 5 - Z, 6 - H, 7 - T, 8 - I, 9 - K, 10 - L, 11 - M, 12 - N.
        Имя = элемент + буква уровня; номер добавляется, только если на уровне несколько
        тяжёлых атомов (1..n в порядке индексов): CB, SG; CD1, CD2.
        Водороды: H + имя атома без элемента, при нескольких водородах - ещё номер:
        при N - H (H1 H2 H3), при CA - HA, HB1 HB2, HG, HD11 HD12.
        Если уровней больше 12 или имя длиннее 4 символов, буквенный способ неприменим:
        функция сообщает об этом и называет атомы способом 'index'.
    naming='index' - атомы родительского остатка (parent_atoms) называются буквенным способом,
        остальные - элемент + индекс атома: C8, N15, H122 (индекс из шага 3.1 виден в имени).
        Без parent_atoms родительскими считаются только атомы остова и их водороды.

    Аргументы:
        rdkit_mol (Chem.Mol) - остаток с явными водородами
        resname, resid, segid - имя (до 3 символов), номер и цепь остатка
        force_numeric (bool) - устаревший параметр, равносилен naming='index'
        C_terminal (bool) - C-концевой остаток (OC1, OC2)
        naming (str) - 'greek' или 'index'
        parent_atoms (list) - индексы атомов родительского остатка (свойство 'ParentAtoms'
            после renumber_residue_atoms); используется при naming='index'
    Возвращает:
        Chem.Mol - та же молекула; имена в свойстве атомов 'AtomName' и в PDB residue info.
    """
    if force_numeric:
        print_red("⚠ force_numeric устарел: используется naming='index'.")
        naming = 'index'
    if naming not in NAMING_METHODS:
        raise ValueError(f"naming='{naming}': допустимые способы {NAMING_METHODS}")

    old_resnames = {atom.GetPDBResidueInfo().GetResidueName().strip()
                    for atom in rdkit_mol.GetAtoms() if atom.GetPDBResidueInfo() is not None}
    old_resnames -= {'', 'UNL'}
    if old_resnames:
        print(f"Атомы уже названы (остаток {', '.join(sorted(old_resnames))}): "
              "имена назначаются заново, старые имена не используются.")

    CA_idx, backbone = find_backbone_match(rdkit_mol, C_terminal)
    backbone_names = ["N", "CA", "C", "OC1", "OC2"] if C_terminal else ["N", "CA", "C", "O"]
    backbone_heavy = dict(zip(backbone, backbone_names))

    heavy_all = {a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetAtomicNum() > 1}
    hydrogens_all = {a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetAtomicNum() == 1}
    side_all = heavy_all - set(backbone)

    def greek_names(heavy_side, hydrogens):
        heavy_names, unreached = _greek_heavy_names(rdkit_mol, CA_idx, heavy_side)
        heavy_names.update(backbone_heavy)
        names = dict(heavy_names)
        names.update(_hydrogen_names(rdkit_mol, heavy_names, hydrogens))
        long_names = sorted(name for name in names.values() if len(name) > 4)
        if long_names:
            raise _NamingError(f'имена длиннее 4 символов: {long_names}')
        return names, unreached

    method = naming
    if naming == 'greek':
        try:
            names, unreached = greek_names(side_all, hydrogens_all)
        except _NamingError as e:
            print_red(f"⚠ Буквенный способ неприменим: {e}. Атомы названы способом 'index'"
                      + ("." if parent_atoms is not None else
                         " (parent_atoms не переданы: буквенные имена только у остова)."))
            method = 'index'
        else:
            if unreached:
                raise ValueError(f'Атомы {unreached} не связаны с CA через боковую цепь: '
                                 'это не один остаток.')

    if method == 'index':
        if parent_atoms is None:
            parent = set(backbone) | {h.GetIdx() for idx in backbone
                                      for h in rdkit_mol.GetAtomWithIdx(idx).GetNeighbors()
                                      if h.GetAtomicNum() == 1}
        else:
            parent = {int(i) for i in parent_atoms}
            bad = sorted(i for i in parent if not 0 <= i < rdkit_mol.GetNumAtoms())
            if bad:
                raise ValueError(f'parent_atoms: индексы вне молекулы ({rdkit_mol.GetNumAtoms()} атомов): {bad}')
            missing = sorted(set(backbone) - parent)
            if missing:
                raise ValueError(f'parent_atoms не содержат атомы остова {missing}: '
                                 'список не от этой молекулы или от другой нумерации.')
        try:
            names, unreached = greek_names(side_all & parent, hydrogens_all & parent)
        except _NamingError as e:
            raise ValueError(f'Родительский остаток нельзя назвать буквенным способом: {e}')
        if unreached:
            raise ValueError(f'Атомы родительского остатка {unreached} не связаны с CA '
                             'через другие атомы родительского остатка.')
        for atom in rdkit_mol.GetAtoms():
            if atom.GetIdx() not in names:
                names[atom.GetIdx()] = f'{atom.GetSymbol().upper()}{atom.GetIdx()}'
        long_names = sorted(name for name in names.values() if len(name) > 4)
        if long_names:
            raise ValueError(f'Имена длиннее 4 символов (поле PDB): {long_names}')

    counts = Counter(names.values())
    duplicates = {name: sorted(i for i, n in names.items() if n == name)
                  for name, c in counts.items() if c > 1}
    if duplicates:
        raise ValueError(f'Повторяющиеся имена атомов (имя: индексы): {duplicates}')

    for atom in rdkit_mol.GetAtoms():
        name = names[atom.GetIdx()]
        info = Chem.AtomPDBResidueInfo()
        info.SetName(_pdb_atom_name_field(name, atom.GetSymbol()))
        info.SetResidueName(resname)
        info.SetResidueNumber(resid)
        info.SetChainId(segid)
        info.SetIsHeteroAtom(False)
        atom.SetProp("AtomName", name)
        atom.SetMonomerInfo(info)

    if method == 'greek':
        print(f"Имена атомов (буквенный способ): {' '.join(names[i] for i in sorted(names))}")
    else:
        rule_named = [i for i in sorted(names) if names[i] != f'{rdkit_mol.GetAtomWithIdx(i).GetSymbol().upper()}{i}']
        print(f"Имена атомов (способ 'index'): родительский остаток - "
              f"{' '.join(names[i] for i in rule_named)}; "
              f"остальные {rdkit_mol.GetNumAtoms() - len(rule_named)} атомов - элемент + индекс.")
    log_note('имена атомов', method=method, names={i: names[i] for i in sorted(names)})
    return rdkit_mol


COORD_TYPES = ('2D', '3D')


def _ca_volume(conf, N_idx, CA_idx, C_idx, CB_idx):
    """
    Ориентированный объём у CA: (N-CA)·((C-CA)×(CB-CA)). У L-аминокислот он положительный
    (проверено по всем шаблонам molecules/aminoacids_template/*_H.pdb), у D - отрицательный.
    В отличие от R/S, знак не зависит от приоритетов CIP (L-цистеин - это R).
    """
    p = [np.array(conf.GetAtomPosition(i)) for i in (N_idx, CA_idx, C_idx, CB_idx)]
    return float(np.dot(p[0] - p[1], np.cross(p[2] - p[1], p[3] - p[1])))


def _embed_3d(mol, random_seed):
    if AllChem.EmbedMolecule(mol, randomSeed=random_seed, enforceChirality=True) != 0:
        raise ValueError('Не удалось построить 3D-структуру с заданной стереохимией '
                         '(EmbedMolecule). Проверьте параметр stereo.')


def set_mol_coords(rdkit_mol, coords='2D', stereo=None, ca_config='L', stereo_default='R',
                   random_seed=42):
    """
    Задаёт координаты молекулы: 2D-раскладку или 3D-структуру с заданной стереохимией.
    Меняет переданную молекулу (функции сохранения передают сюда копию).

    coords='2D' (по умолчанию) - плоская раскладка RDKit. Она однозначна, не зависит от
        случайных чисел, и по ней RDKit при чтении PDB не придумывает стереохимию.
        Стереохимия в 2D PDB не сохраняется, параметры stereo и ca_config не используются.
    coords='3D' - 3D-структура (EmbedMolecule + оптимизация UFF) с заданной стереохимией:
        1. stereo={индекс: 'R' или 'S'} - явно заданные стереоцентры (по правилам CIP);
        2. CA аминокислоты (если в молекуле найден остов N-CA-C=O с явными H) -
           ca_config: 'L' (по умолчанию, как в белках), 'D' или None (как остальные центры).
           L/D задаётся по геометрии, а не через R/S: L-цистеин по CIP - R, остальные L - S;
        3. остальные стереоцентры: если конфигурация уже задана в молекуле (например, @ в
           SMILES), она сохраняется; иначе ставится stereo_default ('R' или 'S') и функция
           сообщает, какие центры так заданы;
        4. random_seed фиксирует построение: одна и та же молекула даёт одни и те же координаты.
        После оптимизации конфигурации проверяются по 3D-координатам.

    Аргументы:
        rdkit_mol (Chem.Mol) - молекула (для 3D - с явными водородами)
        coords (str) - '2D' или '3D'
        stereo (dict) - {индекс атома: 'R'/'S'}
        ca_config (str) - 'L', 'D' или None
        stereo_default (str) - 'R' или 'S' для незаданных стереоцентров
        random_seed (int) - затравка случайных чисел для 3D
    Возвращает:
        Chem.Mol - та же молекула с одним конформером.
    """
    if coords not in COORD_TYPES:
        raise ValueError(f"coords='{coords}': допустимые значения {COORD_TYPES}")
    if coords == '2D':
        if stereo:
            print_red("⚠ stereo задаётся только при coords='3D': в 2D стереохимия не сохраняется.")
        rdkit_mol.RemoveAllConformers()
        rdDepictor.Compute2DCoords(rdkit_mol)
        return rdkit_mol

    from rdkit.Chem import rdCIPLabeler
    stereo = {int(k): str(v).upper() for k, v in (stereo or {}).items()}
    if stereo_default not in ('R', 'S'):
        raise ValueError(f"stereo_default='{stereo_default}': допустимо 'R' или 'S'")
    if ca_config not in ('L', 'D', None):
        raise ValueError(f"ca_config='{ca_config}': допустимо 'L', 'D' или None")
    if any(v not in ('R', 'S') for v in stereo.values()):
        raise ValueError(f"stereo: конфигурации задаются как 'R' или 'S': {stereo}")

    centers = [idx for idx, _ in Chem.FindMolChiralCenters(rdkit_mol, includeUnassigned=True,
                                                           useLegacyImplementation=False)]
    bad = sorted(set(stereo) - set(centers))
    if bad:
        raise ValueError(f'stereo: атомы {bad} не стереоцентры (стереоцентры: {centers})')

    # CA и его соседи по остову
    ca = None
    if ca_config is not None:
        try:
            CA_idx, backbone = find_backbone_match(rdkit_mol)
        except ValueError as e:
            print(f'Остов аминокислоты не найден ({e}); CA задаётся как остальные стереоцентры.')
        else:
            N_idx, C_idx = backbone[0], backbone[2]
            cb = [a.GetIdx() for a in rdkit_mol.GetAtomWithIdx(CA_idx).GetNeighbors()
                  if a.GetAtomicNum() > 1 and a.GetIdx() not in (N_idx, C_idx)]
            if CA_idx in stereo:
                print(f'CA ({CA_idx}) задан в stereo: {stereo[CA_idx]}, ca_config не используется.')
            elif len(cb) == 1 and CA_idx in centers:
                ca = (N_idx, CA_idx, C_idx, cb[0])
    ca_idx = ca[1] if ca else None

    # R/S: явно заданные, уже заданные в молекуле, по умолчанию
    target = dict(stereo)
    by_default = []
    for idx in centers:
        if idx in target or idx == ca_idx:
            continue
        tag = rdkit_mol.GetAtomWithIdx(idx).GetChiralTag()
        if tag in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
            continue
        target[idx] = stereo_default
        by_default.append(idx)
    for idx in target:
        rdkit_mol.GetAtomWithIdx(idx).SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    # CA: тег подбирается по ориентации в 3D
    if ca is not None:
        ca_atom = rdkit_mol.GetAtomWithIdx(ca_idx)
        was_set = ca_atom.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW,
                                             Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
        if not was_set:
            ca_atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
        probe = Chem.Mol(rdkit_mol)
        _embed_3d(probe, random_seed)
        if (_ca_volume(probe.GetConformer(), *ca) > 0) != (ca_config == 'L'):
            ca_atom.InvertChirality()
            if was_set:
                print_red(f'⚠ CA ({ca_idx}) в исходной молекуле не {ca_config}; '
                          f'задаётся {ca_config} (ca_config).')

    # подгонка R/S: метка CIP центра может зависеть от соседних центров, поэтому несколько проходов
    for _ in range(3):
        rdCIPLabeler.AssignCIPLabels(rdkit_mol)
        wrong = [idx for idx, cip in target.items()
                 if rdkit_mol.GetAtomWithIdx(idx).HasProp('_CIPCode')
                 and rdkit_mol.GetAtomWithIdx(idx).GetProp('_CIPCode') != cip]
        if not wrong:
            break
        for idx in wrong:
            rdkit_mol.GetAtomWithIdx(idx).InvertChirality()
    no_label = [idx for idx in target if not rdkit_mol.GetAtomWithIdx(idx).HasProp('_CIPCode')]
    if no_label:
        print_red(f'⚠ Для центров {no_label} метка R/S не определяется (CIP): '
                  'их конфигурация выбрана произвольно, но воспроизводимо.')

    rdkit_mol.RemoveAllConformers()
    _embed_3d(rdkit_mol, random_seed)
    AllChem.UFFOptimizeMolecule(rdkit_mol)

    # проверка по 3D-координатам
    check = Chem.Mol(rdkit_mol)
    Chem.AssignStereochemistryFrom3D(check)
    rdCIPLabeler.AssignCIPLabels(check)

    def cip(idx):
        atom = check.GetAtomWithIdx(idx)
        return atom.GetProp('_CIPCode') if atom.HasProp('_CIPCode') else '?'

    got = {idx: cip(idx) for idx in target}
    mismatch = {idx: (target[idx], got[idx]) for idx in target
                if idx not in no_label and got[idx] != target[idx]}
    if ca is not None and (_ca_volume(rdkit_mol.GetConformer(), *ca) > 0) != (ca_config == 'L'):
        mismatch[ca_idx] = (ca_config, 'L' if ca_config == 'D' else 'D')
    if mismatch:
        raise ValueError(f'3D-структура не совпала с заданной стереохимией (атом: (задано, получено)): {mismatch}')

    parts = []
    if ca is not None:
        parts.append(f'CA ({ca_idx}) - {ca_config} ({cip(ca_idx)} по CIP)')
    if stereo:
        parts.append('заданы: ' + ', '.join(f'{i} - {c}' for i, c in sorted(stereo.items())))
    kept = sorted(set(centers) - set(target) - {ca_idx})
    if kept:
        parts.append('из молекулы: ' + ', '.join(f'{i} - {cip(i)}' for i in kept))
    if by_default:
        parts.append(f'по умолчанию {stereo_default}: {by_default}')
    print('3D-структура, стереоцентры: ' + ('; '.join(parts) if parts else 'нет'))
    return rdkit_mol


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
                print(f"Не учтенный вариант {atom.GetProp('AtomName')}:",
                     f"{name}, {hyb}, {n_hydrogens}, {n_heavy_neighbors}", sep='\n')
                geom_n = '-'
        elif hyb == Chem.HybridizationType.SP2:
            if n_hydrogens == 1 and n_heavy_neighbors == 2:
                geom_n = 1
            elif n_hydrogens == 2 and n_heavy_neighbors == 1:
                geom_n = 3
            elif n_hydrogens == 1 and n_heavy_neighbors == 1:
                geom_n = 2
            else:
                print(f"Не учтенный вариант {atom.GetProp('AtomName')}:",
                     f"{name}, {hyb}, {n_hydrogens}, {n_heavy_neighbors}", sep='\n')
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
        
@logged
def save_aa_chem_to_pdb(rdkit_mol, path, resname = 'MOD', resid = 1, segid = 'A', make_N_root=False,
                        atom_names_list=None, naming='greek', parent_atoms=None, coords='2D',
                        stereo=None, ca_config='L', stereo_default='R', random_seed=42):
    """
    Сохраняет остаток в PDB: имена атомов (rdkit_pdb_modification), имя, номер и цепь остатка.
    Пишутся два файла: <path>.pdb и <path>_no_H.pdb. Переданная молекула не меняется
    (всё делается на копии), поэтому повторный запуск ячейки даёт тот же результат.

    Аргументы:
        rdkit_mol - RDKit.Chem молекула с явными водородами
        path - строка с путем к файлу и его именем для сохранения
        resname, resid, segid - имя остатка (до 3 символов), номер, цепь
        make_N_root - устаревший параметр: переставить атомы через SMILES с корнем в аминогруппе
        atom_names_list - не используется, оставлен для совместимости вызовов
        naming - способ именования атомов: 'greek' (по умолчанию) или 'index'
        parent_atoms - индексы атомов родительского остатка для naming='index'
        coords - '2D' (по умолчанию) или '3D'; stereo, ca_config, stereo_default,
            random_seed - параметры 3D-структуры и стереохимии, см. set_mol_coords.
            До 2026-09 функция всегда строила 3D-структуру со случайной стереохимией.
    """
    rdkit_mol = Chem.Mol(rdkit_mol)
    if make_N_root and parent_atoms is not None:
        raise ValueError('make_N_root переставляет атомы, и индексы parent_atoms перестают им '
                         'соответствовать: используйте что-то одно.')
    if make_N_root:
        root_idx = find_amino_nitrogen(rdkit_mol)
        if root_idx != -1:
            smiles_string = Chem.MolToSmiles(rdkit_mol, canonical=True, allHsExplicit=True, rootedAtAtom=root_idx)

            rdkit_mol = Chem.MolFromSmiles(smiles_string, sanitize=False)
            if rdkit_mol is None:
                raise ValueError("Ошибка при создании молекулы после перенумерации атомов")
    try:
        # Chem.SanitizeMol(rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE, catchErrors=True)
        Chem.SanitizeMol(rdkit_mol)
    except ValueError as e:
        print("Ошибка санации:", e)
        
    # Получение пути и имени файла
    path_to_file, file_name = path_parser(path, ['pdb'])
    
    set_mol_coords(rdkit_mol, coords=coords, stereo=stereo, ca_config=ca_config,
                   stereo_default=stereo_default, random_seed=random_seed)

    if len(resname) > 3:
        print_red(f"⚠ Имя остатка '{resname}' длиннее 3 символов: RDKit запишет в PDB только "
                  f"'{resname[:3]}'.")
    rdkit_mol = rdkit_pdb_modification(rdkit_mol, resname=resname, resid=resid, segid=segid,
                                       naming=naming, parent_atoms=parent_atoms)
    # RemoveHs с пересчётом валентностей: иначе у ароматических атомов, потерявших явные H,
    # не определено число водородов, кольцо не кекулизуется и PDB не записывается
    # (так было при удалении H через RemoveAtom и при RemoveHs(sanitize=False))
    rdkit_mol_no_H = Chem.RemoveHs(rdkit_mol)

    # Запись в файл: текст PDB формируется до открытия файла, чтобы при ошибке
    # не оставить на месте старого файла пустой

    output_path = f"{path_to_file}/{file_name}.pdb" if path_to_file else f"{file_name}.pdb"
    pdb_block = Chem.MolToPDBBlock(rdkit_mol)
    with open(output_path, "w") as file:
        file.write(pdb_block)
    print(f'{file_name}.pdb saved to {path_to_file or "current directory"}')

    if '_H'in file_name:
        file_name = file_name.replace('_H', '_no_H')
    else:
        file_name = file_name + '_no_H'
    output_path = f"{path_to_file}/{file_name}.pdb" if path_to_file else f"{file_name}.pdb"
    pdb_block = Chem.MolToPDBBlock(rdkit_mol_no_H)
    with open(output_path, "w") as file:
        file.write(pdb_block)
    print(f'{file_name}.pdb saved to {path_to_file or "current directory"}')

def smi_to_mol2(smi_input, output_format, file_name=None, addh=True):
    """
    Converts an SMI string or file to a molecule file format (mol2, mol, mdl).

    Args:
      smi_input: The SMI string or path to an SMI file.
      output_format: The desired output format (mol2, mol, mdl).
      file_name (optional): The desired output file name (excluding extension).

    Returns:
      None. Raises an error for invalid input or format.
    """

    valid_formats = ['mol2', 'mol', 'mdl']
    if output_format not in valid_formats:
        raise ValueError(f"Invalid output format: {output_format}")
        
    # Check if SMI is a string
    if not isinstance(smi_input, str):
        raise TypeError("Input must be a string")

    # Read SMI string from file or argument
    if smi_input.lower().endswith('.smi') or smi_input.lower().endswith('.smiles') :
        smi_str = open_smi(smi_input) 
    else:
        smi_str = smi_input.strip()
        if file_name is None:
            file_name = 'convert_mol' 

    # Print current molecule
    print(f"Current molecule: {smi_str}")

    # Create molecule, add hydrogens, and generate 3D coordinates
    mol = pybel.readstring("smi", smi_str)
    if  addh:
        mol.addh()
    mol.make3D()

    # Construct output path based on input and arguments
    output_path = file_name + '.' + output_format if file_name else smi_input.split('.')[0] + '.'+output_format

    # Write molecule to file
    mol.write(output_format, output_path, overwrite=True)
    print('output path:',output_path)
    return mol



def rename_folder(folder_name):
    """
    Генерирует новое имя папки, добавляя или увеличивая числовой суффикс.
    """
    if '/' in folder_name:
        path, folder_name = folder_name.rsplit('/',1)
        path += '/'
    else: 
        path = ''
    
    if '_' in folder_name:
        head, tail = folder_name.rsplit('_',1)
        if tail.isdigit():
            new_tail = str(int(tail)+1).zfill(2)
            new_name = f'{head}_{new_tail}'
        else: new_name = f"{head}_{tail}_01"
    else:
        head = ''
        new_name = folder_name + '_01'
    return f"{path}{new_name}"

def get_unique_folder_name(folder_name: dict or str):
    """
    Возвращает уникальное имя папки, проверяя наличие папки и изменяя имя при необходимости.
    """
    try:
        if isinstance(folder_name, dict):
            name_of_dir = str(*folder_name.keys())
        if isinstance(folder_name, str):
            name_of_dir = folder_name

        while os.path.isdir(name_of_dir):
            print(f"Директория {name_of_dir} уже существует.")
            name_of_dir = rename_folder(name_of_dir)
        print(f"Новое имя директории: {name_of_dir}")
        return name_of_dir
    except TypeError as e:
        print_red(f'Словарь должен состоять из одного элемента: \nTypeError: {e}')
    except: 
        print_red('Что-то не так')

def show_list_of_conf(directory, show_full = False):
    """
    Показывает список файлов-конформеров в директории.
    """
    files = os.listdir(directory)
    msgpack_files = [file for file in files if file.endswith(".msgpack")]
    n_conf = len(msgpack_files)
    print(f"Колличество конформеров: {n_conf}")
    if show_full:
        conf_text = '\n'.join(msgpack_files)
        print(conf_text)

        
def check_PDB_residue_info(atom):
    """
    Проверяет, что параметры остатка были правильно установлены для атома.
    """
    monomer_info = atom.GetMonomerInfo()
    # assert monomer_info.IsValid(), "Информация об остатке не установлена!"
    assert monomer_info.GetName().strip(), "Имя атома не установлено!"
    assert monomer_info.GetResidueName().strip(), "Имя остатка не установлено!"
    assert monomer_info.GetResidueNumber() >= 1, "Номер остатка не установлен!"
    assert monomer_info.GetChainId().strip(), "ID сегмента не установлен!"
    print(f"Параметры остатка для атома {monomer_info.GetName()} установлены корректно.", 
          monomer_info.GetName(), 
          monomer_info.GetResidueName(),
        monomer_info.GetResidueNumber(),
        monomer_info.GetChainId(),sep='\n')

def check_duplicate_atom_names(mol):
    """
    Проверяет наличие атомов с одинаковыми именами в молекуле и возвращает словарь.
    """
    # Получаем список всех имен атомов
    atom_names = [atom.GetProp('AtomName') for atom in mol.GetAtoms()]

    # Подсчитываем частоту появления каждого имени
    # Формируем словарь с именами атомов и их индексами в молекуле
    atom_indices = {}
    for i, atom in enumerate(mol.GetAtoms()):
        name = atom.GetProp('AtomName')
        if name in atom_indices:
            atom_indices[name].append(i)
        else:
            atom_indices[name] = [i]

    # Отбираем только те атомы, которые имеют одинаковые имена
    duplicates = {name: indices for name, indices in atom_indices.items() if len(indices) > 1}

    if duplicates:
        print("Обнаружены атомы с одинаковыми именами:")
        for name, indices in duplicates.items():
            print(f"Имя: '{name}', Индексы атомов: {indices}")
    else:
        print("Дубликаты имен атомов не найдены.")

    return duplicates

def set_PDB_residue_info(atom, atom_name, resname='MOD', resid=1, segid='A'):
    
    info = Chem.AtomPDBResidueInfo()
    info.SetName(_pdb_atom_name_field(atom_name, atom.GetSymbol()))  # колонки 13-16 PDB
    info.SetResidueName(resname)            # Устанавливаем имя остатка
    info.SetResidueNumber(resid)            # Устанавливаем номер остатка
    info.SetChainId(segid)                  # Устанавливаем ID сегмента
    atom.SetMonomerInfo(info)
    # return info
    

@logged
def modifie_residue_info(modified_mol,  index_map, resname='MOD', resid=1, segid='A'):
    """
    Переименовывает атомы по словарю {старое имя: новое имя} и задаёт всем атомам параметры
    остатка (имя, номер, цепь). Если имя атома, не упомянутого в словаре, совпадает с одним
    из новых имён, атом получает следующее свободное имя (increment_name).
    Меняет и возвращает переданную молекулу.
    """
    # ref_mol_atom_dict = {atom.GetIdx(): atom.GetProp('AtomName') for atom in reference_mol.GetAtoms() if atom.GetPDBResidueInfo().GetResidueNumber()==2 }
    
    for atom in modified_mol.GetAtoms():
        atom_name = atom.GetProp('AtomName') 
        
        if atom_name in index_map.keys():
            atom_name = index_map[atom_name]
            atom.SetProp("AtomName", atom_name)
            print(f'Имя атома {atom.GetSymbol()} с индексом {atom.GetIdx()} заменено на имя {atom_name} ')
            set_PDB_residue_info(atom, atom_name, resname, resid, segid) 
        else:
            ref_names = set(index_map.values()) 
            new_name = atom_name
            if atom_name in ref_names:
                while new_name in ref_names:
                    new_name = increment_name(new_name)
                print(f'Имя атома {atom_name} с индексом {atom.GetIdx()} заменено на имя {new_name} ')
                atom_name = new_name
                atom.SetProp("AtomName", atom_name)
            set_PDB_residue_info(atom, atom_name, resname, resid, segid)
    return modified_mol

def increment_name(name):
    match = re.match(r'(\D+)(?:(\d*))$', name)
    if match:
        base, num_str = match.groups()
        num = int(num_str or '0')
        return f'{base}{num + 1}'
    else:
        return f'{name}1' # это странно, типа для случая CH1A 

def write_modified_molecule(mol, output_path):
    """
    Записывает измененную молекулу в PDB-файл.
    """
    writer = Chem.PDBWriter(output_path)
    for atom in mol.GetAtoms():
        atom.SetMonomerType(atom.GetProp('AtomName'))
    writer.write(mol)
    writer.close()
    
@logged
def save_molecule_as_pdb(molecule, filename, format_coord='3D'):
    """
    Сохраняет молекулу в формате PDB.

    :param molecule: Молекула в формате RDKit Mol.
    :param filename: Имя файла для сохранения.
    :param format: Формат представления ('2D' или '3D'). По умолчанию '3D'.
    """
    if format_coord == '2D':
        # Рассчитываем 2D координаты
        AllChem.Compute2DCoords(molecule)
    elif format_coord == '3D':
        # Рассчитываем 3D координаты (если еще не рассчитаны)
        Chem.SanitizeMol(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
    else:
        raise ValueError("Недопустимый формат. Выберите '2D' или '3D'.")
    
    path_to_file, file_name = path_parser(filename, ['pdb'])
    save_path = f'{path_to_file}/{file_name}_{format_coord}.pdb'
    # Сохраняем молекулу в PDB формате
    with open(save_path, 'w') as pdb_file:
        pdb_file.write(Chem.MolToPDBBlock(molecule))
    print(f"Молекула успешно сохранена в файле {save_path} в формате {format_coord}.")

@logged
def add_names_from_residue(modified_chem, index_map, resname='MOD', resid=1, segid='A'):

    
    mod_mol = list(modified_chem.values())[0] # костыль работы с словарем молекулы
    # ref_mol = list(reference_chem.values())[0] # костыль работы с словарем молекулы
   

    # старые ноутбуки передают сюда целиком словарь данных сопоставления (match_data_dict)
    if isinstance(index_map, dict) and 'mon_pol_matches' in index_map:
        key_map = next(iter(index_map['mon_pol_matches'].keys()))
        index_map = index_map['mon_pol_matches'][key_map]
    # Присваиваем новые имена атомам в модифицированной молекуле
    # Перезадаем параметры модифицированного остатка  
    mod_residue = modifie_residue_info(mod_mol, index_map, resname, resid, segid)
    
    # Проверка наличия дублирующих имен атомов
    has_duplicates = check_duplicate_atom_names(mod_residue)

    return mod_residue
        
@logged
def save_charges_json(name: str, charge_list: list, path: str = ".") -> None:
    """
    Сохраняет список зарядов в JSON-файл с указанием абсолютного пути
    
    Параметры:
    name (str): Название файла (без расширения)
    charge_list (list): Список зарядов для сохранения
    path (str): Путь для сохранения (по умолчанию текущая директория)
    """
    # Создаем директорию, если она не существует
    os.makedirs(path, exist_ok=True)
    
    # Формируем полный путь к файлу
    file_path = os.path.join(path, f"{name}.json")
    absolute_path = os.path.abspath(file_path)
    
    # Сохраняем данные
    with open(file_path, 'w') as convert_file:
        json.dump(charge_list, convert_file, indent=4)
    
    # Выводим информативное сообщение
    print(f"Файл успешно сохранен: \n{absolute_path}")



## Работа с листом зарядов

@logged
def make_substructure_charge_list(pol_name, charge_array, match_dict, ):
    monomer_charge = [0] * (len(match_dict['substructure'][pol_name].GetAtoms()))
    # charge_array = charge_array.round(5)
    for monomer_index, polymer_index in enumerate(match_dict['mon_pol_matches'][pol_name].values()):
        monomer_charge[monomer_index] = charge_array[polymer_index]
    monomer_charge = np.array(monomer_charge).round(4)
    return monomer_charge


@logged
def calculate_true_sum(array):
    '''
    checking the actual sum of an array
    '''
    return sum([D(f'{val}') for val in array])

@logged
def save_json(name: str, list_data: list, path: str = ".") -> None:
    """
    Сохраняет список зарядов в JSON-файл с указанием абсолютного пути
    
    Параметры:
    name (str): Название файла (без расширения)
    list_data (list): Список зарядов для сохранения
    path (str): Путь для сохранения (по умолчанию текущая директория)
    """
    # Создаем директорию, если она не существует
    os.makedirs(path, exist_ok=True)
    
    # Формируем полный путь к файлу
    file_path = os.path.join(path, f"{name}.json")
    absolute_path = os.path.abspath(file_path)
    
    # Сохраняем данные
    with open(file_path, 'w') as convert_file:
        json.dump(list_data, convert_file, indent=4)
    
    # Выводим информативное сообщение
    print(f"Файл успешно сохранен: \n{absolute_path}")
        
# Финтифлюшки 

def print_green(text):
    """
    Выводит текст зеленым цветом в терминале.
    """
    print("\033[38;5;28m" + text + "\033[0m")        

def print_red(text):
    """
    Выводит текст зеленым цветом в терминале.
    """
    print("\33[31m" + text + "\33[0m") 
    
def animate(done_flag, sh_file_name):
    """
    Отображает анимацию выполнения процесса в терминале с именем sh файла.
    """
    for c in itertools.cycle(['.  ', '.. ', '...']):
        if done_flag():
            break
        sys.stdout.write(f'\rВыполнение {sh_file_name}{c}')
        sys.stdout.flush()
        time.sleep(0.5)
    # sys.stdout.write(f'\r{sh_file_name} выполнен успешно!  ')
    print_green(f'\n{sh_file_name} выполнен успешно!')

def format_time(seconds):
    """
    Форматирует время в секундах в строку формата HH:MM:SS.
    """
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02}:{mins:02}:{secs:02}"

# ---------------------------------------------

aa_dict = {one: names[0] for one, names in AMINO_ACIDS.items()}


# =============================================================================
# ИМПОРТИРОВАННЫЕ ФУНКЦИИ
# Перенесены из ячеек ноутбуков <PTM>/1_charge_calculation*.ipynb (Шаг 4.1),
# где они были определены локально. Код перенесён без изменений.
# =============================================================================

# Источник: Lysine_3M/1_charge_calculation.ipynb (идентичная копия в AF_546_*, Lysine_*;
# в Lysine_Cro вариант с префиксом имени водорода 'HW' вместо 'HW1')
@logged
def add_protons_and_renumber_H(modifie_residue, resname='MOD', resid=1, segid='A', h_name='HW1'):
    """
    Шаг 4.1: достраивает водороды у атомов с неполной валентностью. acpype не строит
    топологию для таких атомов, поэтому у вырезанного из тримера остатка появляются протоны,
    которых нет в белке: на N (вместо связи с предыдущим остатком) и на C (вместо связи
    со следующим). В 3_edd_topology они удаляются из топологии.

    1. Атомы остатка сохраняют свои индексы, имена и параметры остатка: новые водороды
       добавляются в конец (индексы и заряды шага 5 совпадают с шагом 4).
    2. Новые водороды получают свободные имена h_name, h_name+1, ... (HW1, HW2, ...),
       параметры остатка resname, resid, segid и свойство атома 'AddedH' = True.
    3. Список индексов новых водородов - в свойстве молекулы 'AddedH' (JSON).
    Атомы с формальным зарядом (например, O- сульфогруппы) не протонируются: RDKit
    считает их валентность полной.

    Аргументы:
        modifie_residue (Chem.Mol) - остаток после шага 4 (у всех атомов есть 'AtomName')
        resname, resid, segid - параметры остатка для новых водородов
        h_name (str) - имя первого нового водорода (в Lysine_Cro использовалось 'HW')
    Возвращает:
        Chem.Mol - новая молекула с 2D-координатами (исходная не меняется).
    """
    n_old = modifie_residue.GetNumAtoms()
    unnamed = [a.GetIdx() for a in modifie_residue.GetAtoms() if not a.HasProp('AtomName')]
    if unnamed:
        raise ValueError(f'У атомов {unnamed} нет имён (AtomName): на вход нужен остаток после шага 4.')
    names = [a.GetProp('AtomName') for a in modifie_residue.GetAtoms()]

    H_modifie_residue = Chem.AddHs(modifie_residue, addCoords=True)
    added = list(range(n_old, H_modifie_residue.GetNumAtoms()))

    report = []
    for idx in added:
        atom = H_modifie_residue.GetAtomWithIdx(idx)
        new_name = h_name
        while new_name in names:
            new_name = increment_name(new_name)
        names.append(new_name)
        set_PDB_residue_info(atom, new_name, resname, resid, segid)
        atom.SetProp('AtomName', new_name)
        atom.SetBoolProp('AddedH', True)
        heavy = atom.GetNeighbors()[0]
        report.append(f"{new_name} - {heavy.GetProp('AtomName')} ({heavy.GetIdx()})")
    H_modifie_residue.SetProp('AddedH', json.dumps(added))
    AllChem.Compute2DCoords(H_modifie_residue)

    if added:
        print(f"Добавлены водороды (имя - атом): {', '.join(report)}")
    else:
        print_green('Все атомы со стандартной валентностью, водороды не добавлены.')
    log_note('добавленные водороды', added={i: r for i, r in zip(added, report)})
    return H_modifie_residue
