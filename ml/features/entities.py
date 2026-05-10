"""Entity re-exports — thin wrapper over definitions.py."""

try:
    from definitions import driver  # Feast context (ml/features/ on sys.path)
except ImportError:
    from ml.features.definitions import driver  # normal package import

__all__ = ["driver"]
