"""
Dynamic Taint Analysis (DTA) module for NNAST.

Provides lightweight runtime taint tracking for Python code.
"""

from dta.tracker import TaintTracker, taint_source, taint_sink
from dta.instrument import instrument_file

__all__ = ["TaintTracker", "taint_source", "taint_sink", "instrument_file"]

