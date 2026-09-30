from decimal import Decimal
from io import StringIO

import pytest

from agent_server.csv_totals import monthly_totals_by_category


def test_aggregates_valid_rows_by_month_and_category():
    result = monthly_totals_by_category(
        StringIO(
            "date,category,amount\n"
            "2026-01-02,food,10.25\n"
            "2026-01-31,food,4.75\n"
            "2026-01-15,travel,20\n"
            "2026-02-01,food,6\n"
        )
    )

    assert result.totals == {
        "2026-01": {"food": Decimal("15.00"), "travel": Decimal("20")},
        "2026-02": {"food": Decimal("6")},
    }
    assert result.malformed_rows == []


def test_reports_invalid_rows_and_keeps_valid_rows():
    result = monthly_totals_by_category(
        StringIO(
            "date,category,amount\n"
            "2026-01-02,food,10\n"
            "not-a-date,food,5\n"
            "2026-01-03,,4\n"
            "2026-01-04,travel,not-a-number\n"
            "2026-01-05,travel,NaN\n"
        )
    )

    assert result.totals == {"2026-01": {"food": Decimal("10")}}
    assert [(row.row_number, row.reason) for row in result.malformed_rows] == [
        (3, "date must use YYYY-MM-DD format"),
        (4, "missing value(s): category"),
        (5, "amount must be a decimal number"),
        (6, "amount must be finite"),
    ]


def test_rejects_files_without_required_headers():
    with pytest.raises(ValueError, match="missing required column\\(s\\): amount"):
        monthly_totals_by_category(StringIO("date,category\n2026-01-02,food\n"))
