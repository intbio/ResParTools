"""
ResParTools - точка входа для ноутбуков: import ResParTools as pt.

Код находится в пакете respartools/ рядом с этим файлом (модули log, utils, fileio, draw,
matching, residue, forcefield, charges, legacy). Здесь собираются все их имена, включая
служебные (с подчёркиванием), чтобы старые и новые ноутбуки работали как с единым модулем.
"""
from respartools import *  # noqa: F401,F403
import respartools as _package

for _module in _package.MODULES:
    globals().update({_k: _v for _k, _v in vars(_module).items() if not _k.startswith('__')})
del _module
