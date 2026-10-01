"""
ResParTools: подготовка модифицированных остатков (аминокислоты, нуклеиновые кислоты)
для силового поля GROMACS amber14sb.

Модули:
    log         - лог для отладки (start_log, stop_log, log_note, @logged)
    utils       - вспомогательные функции
    fileio      - чтение и запись молекул (SMILES, PDB, mol2, MOL, JSON)
    draw        - рисование молекул
    matching    - сопоставление подструктур, шаблоны родительских остатков
    residue     - перенумерация, имена атомов, координаты, параметры остатка, протонирование
    forcefield  - файлы силового поля (hdb, atp, r2b)
    charges     - заряды: ограничения из rtp, списки зарядов
    legacy      - совместимость со старыми ноутбуками

В ноутбуках пакет подключается как раньше: import ResParTools as pt
(ResParTools.py рядом с пакетом отдаёт все его имена).
"""
from .log import *  # noqa: F401,F403
from .utils import *  # noqa: F401,F403
from .fileio import *  # noqa: F401,F403
from .draw import *  # noqa: F401,F403
from .matching import *  # noqa: F401,F403
from .residue import *  # noqa: F401,F403
from .forcefield import *  # noqa: F401,F403
from .charges import *  # noqa: F401,F403
from .legacy import *  # noqa: F401,F403
from . import log, utils, fileio, draw, matching, residue, forcefield, charges, legacy

MODULES = (log, utils, fileio, draw, matching, residue, forcefield, charges, legacy)
