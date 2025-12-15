"""
Example script demonstrating taint tracking usage.

This script shows how to mark sources and sinks, and how taint flows
are automatically tracked during execution.
"""
from dta.tracker import taint_source, taint_sink, get_tracker


# Mark functions as taint sources
@taint_source
def get_user_input():
    """Simulate user input (e.g., from HTTP request)."""
    return "SELECT * FROM users WHERE id = 1"


@taint_source
def read_config_file(path: str):
    """Simulate reading a config file."""
    return "config_value"


# Mark functions as taint sinks
@taint_sink("sql_exec")
def execute_sql(query: str):
    """Simulate SQL execution."""
    print(f"[SQL] Executing: {query}")
    # In real code, this would execute the query


@taint_sink("command_exec")
def execute_command(cmd: str):
    """Simulate command execution."""
    print(f"[CMD] Executing: {cmd}")
    # In real code, this would execute the command


@taint_sink("file_write")
def write_file(path: str, content: str):
    """Simulate file writing."""
    print(f"[FILE] Writing to {path}: {content[:50]}...")
    # In real code, this would write to file


def main():
    # Enable taint tracking
    tracker = get_tracker()
    tracker.enable()

    # Simulate taint flows
    user_query = get_user_input()
    execute_sql(user_query)  # Taint flow: source -> sql_exec sink

    config_value = read_config_file("config.ini")
    execute_command(config_value)  # Taint flow: source -> command_exec sink

    # Non-tainted data shouldn't trigger tracking
    safe_data = "safe string"
    write_file("output.txt", safe_data)  # No taint flow recorded

    # Get and print records
    records = tracker.get_records()
    print(f"\n[INFO] Recorded {len(records)} taint flows:")
    for i, record in enumerate(records, 1):
        print(f"\nFlow {i}:")
        print(f"  Source: {record['source']}")
        print(f"  Sink: {record['sink']}")
        print(f"  Sink Type: {record['meta']['sink_type']}")


if __name__ == "__main__":
    main()

