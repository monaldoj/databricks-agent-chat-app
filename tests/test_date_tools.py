from datetime import datetime, timezone

from agent_server.date_tools import format_today_utc, get_todays_date


def test_formats_today_in_utc_with_year():
    instant = datetime(2026, 9, 15, 1, 30, tzinfo=timezone.utc)
    assert format_today_utc(instant) == "Tuesday, September 15, 2026 (UTC)"


def test_date_function_is_exposed_as_an_agent_tool():
    assert get_todays_date.name == "get_todays_date"
