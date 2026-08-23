"""
DEPRECATED: Compatibility wrapper.

Все новые импорты должны использовать:
    from benchmarks.canonical_schema import ...

Этот файл сохранён для обратной совместимости со старыми скриптами.
Будет удалён в v0.3.0.
"""
import warnings
warnings.warn(
    "canonical_schema.py в корне проекта устарел. "
    "Используйте: from benchmarks.canonical_schema import ...",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export из benchmarks
from benchmarks.canonical_schema import (  # noqa: F401
    TractorMeta,
    ObservationWindow,
    FailureEvent,
    CanonicalFleetDataset,
    DataProvenance,
)
