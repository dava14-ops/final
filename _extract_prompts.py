# -*- coding: utf-8 -*-
import re
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)

# train_model.py prompts
with open('train_model.py', encoding='utf-8') as f:
    lines = f.readlines()

print("=" * 70)
print("=== train_model.py ===")
print("=" * 70)
for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if any(fn in stripped for fn in ['ask_float(', 'ask_int(', 'ask_yesno(', 'ask_choice(']):
        m = re.search(r'ask_\w+\(["\']([^"\']+)["\']', stripped)
        if m:
            print(f"  L{i:5d}: {m.group(1)}")
    elif re.search(r'ask\(["\']', stripped) and 'def ask' not in stripped and 'ask_' not in stripped:
        m = re.search(r'ask\(["\']([^"\']+)["\']', stripped)
        if m:
            print(f"  L{i:5d}: {m.group(1)}")

print()
print("=" * 70)
print("=== real_calculator.py ===")
print("=" * 70)

with open('real_calculator.py', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines, 1):
    stripped = line.strip()
    if any(fn in stripped for fn in ['ask_float(', 'ask_int(', 'ask_str(', 'ask_choice(']):
        m = re.search(r'ask_\w+\(["\']([^"\']+)["\']', stripped)
        if m:
            print(f"  L{i:5d}: {m.group(1)}")
    elif re.search(r'ask\(["\']', stripped) and 'def ask' not in stripped and 'ask_' not in stripped:
        m = re.search(r'ask\(["\']([^"\']+)["\']', stripped)
        if m:
            print(f"  L{i:5d}: {m.group(1)}")
