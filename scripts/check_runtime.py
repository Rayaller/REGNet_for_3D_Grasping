import importlib
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


REQUIRED_MODULES = [
    "numpy",
    "open3d",
    "tensorboardX",
    "tqdm",
    "transforms3d",
]

EXTENSION_MODULES = [
    "pn2_ext",
    "dgcnn_ext",
]

EXTENSION_FALLBACKS = {
    "pn2_ext": "multi_model.utils.pn2_utils.pn2_ext",
    "dgcnn_ext": "multi_model.utils.pn2_utils.functions.dgcnn_ext",
}


def import_or_fail(module_name):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        print(f"[check_runtime] failed to import {module_name}: {exc}", file=sys.stderr)
        raise
    version = getattr(module, "__version__", "unknown")
    print(f"[check_runtime] {module_name}: {version}")
    return module


def import_extension_or_fail(module_name):
    fallback_name = EXTENSION_FALLBACKS.get(module_name)
    try:
        return import_or_fail(module_name)
    except Exception as top_level_exc:
        if fallback_name is None:
            raise
        print(
            f"[check_runtime] top-level import failed for {module_name}, "
            f"trying package fallback {fallback_name}"
        )
        try:
            module = import_or_fail(fallback_name)
        except Exception:
            raise top_level_exc
        print(f"[check_runtime] {module_name}: loaded via {fallback_name}")
        return module


def main():
    print(f"[check_runtime] python: {sys.version.split()[0]}")
    print(f"[check_runtime] executable: {sys.executable}")
    print(f"[check_runtime] cwd: {os.getcwd()}")

    torch = import_or_fail("torch")
    print(f"[check_runtime] torch cuda build: {torch.version.cuda}")
    print(f"[check_runtime] torch cuda available: {torch.cuda.is_available()}")

    for name in REQUIRED_MODULES:
        import_or_fail(name)

    for name in EXTENSION_MODULES:
        import_extension_or_fail(name)

    print("[check_runtime] runtime check passed")


if __name__ == "__main__":
    main()
