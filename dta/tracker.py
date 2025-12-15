"""
Runtime taint tracking for Python.

Provides decorators and context managers to mark sources and sinks,
and tracks taint flow during execution.
"""
import inspect
import sys
from typing import Any, Callable, Dict, List, Optional, Set
from collections import defaultdict
import traceback


class TaintTracker:
    """
    Global taint tracker that records source->sink flows during execution.

    Usage:
        tracker = TaintTracker()
        tracker.enable()

        @tracker.mark_source
        def get_user_input():
            return input()

        @tracker.mark_sink("sql_exec")
        def execute_sql(query):
            ...

        # After execution, get records:
        records = tracker.get_records()
    """

    def __init__(self):
        self.enabled = False
        self._tainted_objects: Dict[int, Dict[str, Any]] = {}  # id(obj) -> taint metadata
        self._records: List[Dict[str, Any]] = []
        self._current_trace: List[Dict[str, Any]] = []
        self._source_registry: Set[Callable] = set()
        self._sink_registry: Dict[Callable, str] = {}  # func -> sink_type

    def enable(self):
        """Enable taint tracking."""
        self.enabled = True

    def disable(self):
        """Disable taint tracking."""
        self.enabled = False

    def mark_source(self, func: Callable) -> Callable:
        """
        Decorator to mark a function as a taint source.

        Returns from this function will be marked as tainted.
        """
        self._source_registry.add(func)

        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if self.enabled:
                self._taint_object(result, self._get_source_position(func))
            return result

        wrapper.__name__ = func.__name__
        wrapper.__module__ = func.__module__
        return wrapper

    def mark_sink(self, sink_type: str):
        """
        Decorator factory to mark a function as a taint sink.

        Args:
            sink_type: Type of sink (e.g., "sql_exec", "command_exec", "file_write")
        """

        def decorator(func: Callable) -> Callable:
            self._sink_registry[func] = sink_type

            def wrapper(*args, **kwargs):
                if self.enabled:
                    # Check if any argument is tainted
                    tainted_args = []
                    for arg in args:
                        if self._is_tainted(arg):
                            tainted_args.append(arg)
                    for v in kwargs.values():
                        if self._is_tainted(v):
                            tainted_args.append(v)

                    if tainted_args:
                        # Record taint flow
                        source_meta = self._get_taint_metadata(tainted_args[0])
                        sink_pos = self._get_sink_position(func)
                        if source_meta and sink_pos:
                            self._record_flow(source_meta, sink_pos, sink_type)

                return func(*args, **kwargs)

            wrapper.__name__ = func.__name__
            wrapper.__module__ = func.__module__
            return wrapper

        return decorator

    def _taint_object(self, obj: Any, source_pos: Optional[Dict[str, Any]]) -> None:
        """Mark an object as tainted with source position."""
        if source_pos:
            obj_id = id(obj)
            self._tainted_objects[obj_id] = {
                "source": source_pos,
                "tainted_at": len(self._current_trace),
            }

    def _is_tainted(self, obj: Any) -> bool:
        """Check if an object is tainted."""
        return id(obj) in self._tainted_objects

    def _get_taint_metadata(self, obj: Any) -> Optional[Dict[str, Any]]:
        """Get taint metadata for an object."""
        obj_id = id(obj)
        return self._tainted_objects.get(obj_id)

    def _get_source_position(self, func: Callable) -> Optional[Dict[str, Any]]:
        """Extract source position from function."""
        try:
            file = inspect.getfile(func)
            lines, start_line = inspect.getsourcelines(func)
            return {
                "file": file,
                "line": start_line,
                "col": 0,  # Simplified: start of function
            }
        except (OSError, TypeError):
            return None

    def _get_sink_position(self, func: Callable) -> Optional[Dict[str, Any]]:
        """Extract sink position from function."""
        try:
            file = inspect.getfile(func)
            lines, start_line = inspect.getsourcelines(func)
            return {
                "file": file,
                "line": start_line,
                "col": 0,
            }
        except (OSError, TypeError):
            return None

    def _record_flow(
        self,
        source_meta: Dict[str, Any],
        sink_pos: Dict[str, Any],
        sink_type: str,
    ) -> None:
        """Record a taint flow from source to sink."""
        record = {
            "source": source_meta.get("source"),
            "sink": sink_pos,
            "path": [],  # Simplified: no intermediate path tracking yet
            "meta": {
                "taint_kind": "user_input",  # Default
                "sink_type": sink_type,
            },
        }
        self._records.append(record)

    def get_records(self) -> List[Dict[str, Any]]:
        """Get all recorded taint flows."""
        return self._records.copy()

    def clear(self):
        """Clear all records and taint state."""
        self._records.clear()
        self._tainted_objects.clear()
        self._current_trace.clear()


# Global tracker instance
_global_tracker = TaintTracker()


def taint_source(func: Callable) -> Callable:
    """Convenience decorator to mark a function as a taint source."""
    return _global_tracker.mark_source(func)


def taint_sink(sink_type: str):
    """Convenience decorator factory to mark a function as a taint sink."""
    return _global_tracker.mark_sink(sink_type)


def get_tracker() -> TaintTracker:
    """Get the global taint tracker instance."""
    return _global_tracker

