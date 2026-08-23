with open(r'C:/workspace/train_model.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i, line in enumerate(lines, 1):
    if 'make_rng' in line or 'make_rng' in line.lower():
        print(f'Line {i}: {line.rstrip()}')
