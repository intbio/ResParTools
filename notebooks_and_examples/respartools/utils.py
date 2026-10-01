"""Мелкие вспомогательные функции: имена ключей, декоратор data_to_dict, пути, вывод в консоль."""
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

from .log import _LOG, logged

# notebooks_and_examples/: папка, где лежат ResParTools.py, пакет и данные (шаблоны, силовые поля)
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


## Вспомогательные мини функции
def get_name(path: str, index=0):
    """
    Аргумент: 
        path - строка с путем к файлу или строка SMILES
    Возвращает:
        Имя для ключа в словаре: для SMILES - index, для пути - имя файла без расширения
    """
    from rdkit import rdBase
    # проверка «это SMILES?» не должна печатать ошибки разбора RDKit для путей к файлам
    # (RDKit пишет их в C++ stderr, перенаправление sys.stderr их не скрывает)
    blocker = rdBase.BlockLogs()
    try:
        mol = Chem.MolFromSmiles(path)
    finally:
        del blocker
    if mol:
        return index
    return os.path.basename(path).split('.')[0]


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

def increment_name(name):
    match = re.match(r'(\D+)(?:(\d*))$', name)
    if match:
        base, num_str = match.groups()
        num = int(num_str or '0')
        return f'{base}{num + 1}'
    else:
        return f'{name}1' # это странно, типа для случая CH1A 


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
