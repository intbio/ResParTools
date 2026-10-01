"""
Общие настройки тестов ResParTools.

Тесты запускаются из корня репозитория в окружении darwin_resp (там есть pytest, RDKit и psiresp):
    conda activate darwin_resp
    python -m pytest tests -q                 # все тесты (~5-10 минут, AF546 - самая долгая часть)
    python -m pytest tests -q -m "not slow"   # без флуоресцентной метки AF546
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks_and_examples')
sys.path.insert(0, ROOT)


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: долгие тесты (флуоресцентная метка AF546, ~130 атомов)')
