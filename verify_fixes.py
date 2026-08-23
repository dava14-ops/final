# -*- coding: utf-8 -*-
"""Portable compile-time verification for the core runtime modules."""
from __future__ import annotations

import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = [
    ROOT / "predictor.py",
    ROOT / "prediction_engine.py",
    ROOT / "cli.py",
    ROOT / "service.py",
    ROOT / "severity_model.py",
    ROOT / "premium_engine.py",
    ROOT / "train_model.py",
]


def main() -> int:
    all_ok = True
    for path in FILES:
        try:
            py_compile.compile(str(path), doraise=True)
            print(f"[OK] {path.relative_to(ROOT)}")
        except (py_compile.PyCompileError, FileNotFoundError) as exc:
            print(f"[FAIL] {path.relative_to(ROOT)}: {exc}")
            all_ok = False

    if all_ok:
        print("\n[SUCCESS] All core files compile successfully!")
        return 0

    print("\n[FAILURE] Some files have errors!")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())