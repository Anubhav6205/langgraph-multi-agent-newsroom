import inspect
from rich.console import Console

console = Console()


def section(title: str, color: str = "bold white"):
    """Horizontal rule header — marks entry into a function/module."""
    console.rule(f"[{color}]{title}[/]")


def log_in(label: str, value):
    """Log an incoming state/input value (cyan). Auto-reads caller line number."""
    line = inspect.currentframe().f_back.f_lineno
    console.print(f"[cyan]  L{line} | {label}:[/] {value}")


def log_out(label: str, value):
    """Log an outgoing result/decision value (green). Auto-reads caller line number."""
    line = inspect.currentframe().f_back.f_lineno
    console.print(f"[green]  L{line} | {label}:[/] [bold]{value}[/]")


def log_messages(messages: list, preview_len: int = 80):
    """Dump a full message list — type, role/name, tool_calls, content preview."""
    line = inspect.currentframe().f_back.f_lineno
    console.print(f"[cyan]  L{line} | messages count:[/] {len(messages)}")
    for i, m in enumerate(messages):
        role = getattr(m, "role", getattr(m, "type", "?"))
        name = getattr(m, "name", None)
        tool_calls = getattr(m, "tool_calls", [])
        preview = str(m.content)[:preview_len]

        parts = [f"type={type(m).__name__}", f"role={role}"]
        if name:
            parts.append(f"name={name!r}")
        if tool_calls:
            parts.append(f"tool_calls={tool_calls}")
        parts.append(f"content[:80]={preview!r}")

        console.print(f"[dim]       msg[{i}] {' | '.join(parts)}[/]")
