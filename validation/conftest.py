"""
Проверка корректности ResParTools на референсных остатках: результат библиотеки сравнивается с
независимым источником (файлы базового поля amber14sb, ручные эталоны), а не с файлами, которые
сделал тот же код. Описание и список проверок - validation/README.md.

    conda activate darwin_resp
    python -m pytest validation -q
"""
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks_and_examples')
sys.path.insert(0, ROOT)
