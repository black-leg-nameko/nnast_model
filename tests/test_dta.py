"""
Tests for the lightweight DTA (Dynamic Taint Analysis) module.
"""
import json
import tempfile
from pathlib import Path

from dta.tracker import TaintTracker, taint_source, taint_sink, get_tracker


def test_taint_source_and_sink():
    """Test basic source->sink taint flow tracking."""
    tracker = TaintTracker()
    tracker.enable()

    @tracker.mark_source
    def get_user_input():
        return "SELECT * FROM users WHERE id = 1"

    @tracker.mark_sink("sql_exec")
    def execute_sql(query):
        pass  # Simulated SQL execution

    # Execute flow
    user_data = get_user_input()
    execute_sql(user_data)

    records = tracker.get_records()
    assert len(records) == 1
    record = records[0]
    assert "source" in record
    assert "sink" in record
    assert record["meta"]["sink_type"] == "sql_exec"


def test_convenience_decorators():
    """Test convenience decorators (taint_source, taint_sink)."""
    tracker = get_tracker()
    tracker.clear()
    tracker.enable()

    @taint_source
    def read_file(path):
        return "file contents"

    @taint_sink("file_write")
    def write_file(path, content):
        pass

    content = read_file("test.txt")
    write_file("output.txt", content)

    records = tracker.get_records()
    assert len(records) >= 1


def test_multiple_flows():
    """Test tracking multiple taint flows."""
    tracker = TaintTracker()
    tracker.enable()

    @tracker.mark_source
    def get_input():
        return "user input"

    @tracker.mark_sink("sql_exec")
    def sql_exec(query):
        pass

    @tracker.mark_sink("command_exec")
    def exec_command(cmd):
        pass

    data = get_input()
    sql_exec(data)
    exec_command(data)

    records = tracker.get_records()
    assert len(records) == 2
    assert all(r["meta"]["sink_type"] in ("sql_exec", "command_exec") for r in records)


def test_taint_not_propagated():
    """Test that non-tainted data doesn't trigger sinks."""
    tracker = TaintTracker()
    tracker.enable()

    def get_safe_data():
        return "safe data"

    @tracker.mark_sink("sql_exec")
    def execute_sql(query):
        pass

    safe = get_safe_data()
    execute_sql(safe)

    records = tracker.get_records()
    assert len(records) == 0  # No taint flow recorded

