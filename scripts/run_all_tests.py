"""Standalone test runner for PRISM test suite."""
from __future__ import annotations

import importlib
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

test_files = [
    "test_sector_pipeline",
    "test_resolve_ledger",
    "test_dm_test",
    "test_scanner",
    "test_fase7_coverage",
    "test_horizon_ui",
    "test_model_real",
    "test_pipeline_smoke",
]


def run_all():
    passed = 0
    failed = 0
    print("=" * 60)
    print("RUNNING PRISM COMPLETE TEST SUITE")
    print("=" * 60)

    for mod_name in test_files:
        print(f"\n[TEST FILE] {mod_name}.py:")
        try:
            mod = importlib.import_module(mod_name)
            test_funcs = [fn for fn in dir(mod) if fn.startswith("test_") and callable(getattr(mod, fn))]
            for fn in test_funcs:
                try:
                    func = getattr(mod, fn)
                    # Check if function requires arguments like tmp_path
                    import inspect

                    sig = inspect.signature(func)
                    if "tmp_path" in sig.parameters:
                        import tempfile

                        with tempfile.TemporaryDirectory() as tmpdir:
                            func(Path(tmpdir))
                    else:
                        func()
                    print(f"  ✓ {fn} PASSED")
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {fn} FAILED: {e}")
                    traceback.print_exc()
                    failed += 1
        except Exception as e:
            print(f"  ✗ Import failed for {mod_name}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"SUMMARY: {passed} PASSED, {failed} FAILED across {len(test_files)} test files.")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    run_all()
