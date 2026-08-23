# -*- coding: utf-8 -*-
import py_compile
import sys

sys.stdout.reconfigure(encoding='utf-8')

try:
    py_compile.compile('train_model.py', doraise=True)
    print('✓ train_model.py: компиляция успешна')
except py_compile.PyCompileError as e:
    print(f'✗ train_model.py: ОШИБКА - {e}')
    sys.exit(1)

try:
    py_compile.compile('real_calculator.py', doraise=True)
    print('✓ real_calculator.py: компиляция успешна')
except py_compile.PyCompileError as e:
    print(f'✗ real_calculator.py: ОШИБКА - {e}')
    sys.exit(1)

print('\n✓ Все файлы проверены: синтаксис корректен')
