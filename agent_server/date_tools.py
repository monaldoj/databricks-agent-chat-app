"""Small deterministic tools that give models reliable calendar context."""

from datetime import datetime, timezone

from agents import function_tool


def format_today_utc(now: datetime | None = None) -> str:
    """Format a supplied instant, or the current instant, as a UTC date."""
    current = now or datetime.now(timezone.utc)
    return current.astimezone(timezone.utc).strftime("%A, %B %d, %Y (UTC)")


@function_tool
def get_todays_date() -> str:
    """Return today's date and year in UTC for time-sensitive web searches."""
    return format_today_utc()
