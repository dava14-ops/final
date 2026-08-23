with open(r'C:/workspace/Итог.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    if 'generate_base_instrument' in line:
        print(f'Line {i}: {line.rstrip()}')
