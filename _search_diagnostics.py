import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find run_post_training_diagnostics
start = None
for i, line in enumerate(lines, 1):
    if 'def run_post_training_diagnostics' in line:
        start = i
        break

if start is None:
    print("Function not found")
else:
    print(f"Function starts at line {start}")
    # Read from start to find the next function or end
    for i in range(start-1, min(start+200, len(lines))):
        print(f'{i+1}: {lines[i].rstrip()}')
