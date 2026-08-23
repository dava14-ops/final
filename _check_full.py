import ast
import sys

with open(r'C:/workspace/mc_recovery_test.py', 'r', encoding='utf-8') as f:
    source = f.read()

try:
    ast.parse(source)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax Error: {e}")
    sys.exit(1)

# Check imports
import importlib.util
modules_to_check = [
    ('Итог', r'C:/workspace/Итог.py'),
    ('train_model', r'C:/workspace/train_model.py'),
    ('mc_recovery_stats', r'C:/workspace/mc_recovery_stats.py'),
]

for name, path in modules_to_check:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            print(f"Module '{name}' loaded successfully")
        except Exception as e:
            print(f"Module '{name}' load FAILED: {e}")
    else:
        print(f"Module '{name}' NOT FOUND")

# Check if mc_recovery_test can be imported
print("\n--- Checking mc_recovery_test imports ---")
spec = importlib.util.spec_from_file_location("mc_recovery_test", r'C:/workspace/mc_recovery_test.py')
if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        print("mc_recovery_test loaded successfully")
        
        # Check key symbols
        for name in ['run_mc_recovery', '_run_single_replication', '_cluster_bootstrap_gamma', '_make_fit_options']:
            if hasattr(module, name):
                print(f"  {name}: EXISTS")
            else:
                print(f"  {name}: MISSING")
        
        # Check argparse constants
        for name in ['DEFAULT_N_SIMS', 'DEFAULT_N_TRACTORS', 'DEFAULT_GAMMA_TRUE', 'DEFAULT_SEED']:
            if hasattr(module, name):
                print(f"  {name}: {getattr(module, name)}")
    except Exception as e:
        print(f"mc_recovery_test load FAILED: {e}")
else:
    print("mc_recovery_test NOT FOUND")
