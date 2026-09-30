"""Validate transaction CSV files and aggregate amounts by month and category."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import TextIO


REQUIRED_COLUMNS = ("date", "category", "amount")


@dataclass(frozen=True)
class MalformedRow:
    """A CSV row that could not be included in the totals."""

    row_number: int
    reason: str


@dataclass(frozen=True)
class MonthlyCategoryTotals:
    """The aggregation result and validation failures for one input file."""

    totals: dict[str, dict[str, Decimal]]
    malformed_rows: list[MalformedRow]


def monthly_totals_by_category(
    source: str | Path | TextIO,
) -> MonthlyCategoryTotals:
    """Read a CSV and group valid amounts by ``YYYY-MM`` and category.

    The CSV must have ``date``, ``category``, and ``amount`` columns. Dates use
    ISO format (``YYYY-MM-DD``); amounts are finite decimal values. Invalid data
    rows are recorded with their 1-based CSV line number and processing continues.
    Missing required headers are reported as a ``ValueError`` because no rows can
    be interpreted without them.
    """
    should_close = not hasattr(source, "read")
    handle = open(source, newline="", encoding="utf-8") if should_close else source

    try:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing_columns = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"CSV is missing required column(s): {joined}")

        totals: dict[str, dict[str, Decimal]] = {}
        malformed_rows: list[MalformedRow] = []
        for row_number, row in enumerate(reader, start=2):
            error = _row_error(row)
            if error:
                malformed_rows.append(MalformedRow(row_number, error))
                continue

            parsed_date = date.fromisoformat(row["date"].strip())
            category = row["category"].strip()
            amount = Decimal(row["amount"].strip())
            month = parsed_date.strftime("%Y-%m")
            monthly_totals = totals.setdefault(month, {})
            monthly_totals[category] = monthly_totals.get(category, Decimal("0")) + amount

        return MonthlyCategoryTotals(totals, malformed_rows)
    finally:
        if should_close:
            handle.close()


def _row_error(row: dict[str, str | None]) -> str | None:
    """Return a user-facing validation error for a CSV row, if any."""
    values = {column: (row[column] or "").strip() for column in REQUIRED_COLUMNS}
    missing = [column for column, value in values.items() if not value]
    if missing:
        return f"missing value(s): {', '.join(missing)}"

    try:
        date.fromisoformat(values["date"])
    except ValueError:
        return "date must use YYYY-MM-DD format"

    try:
        amount = Decimal(values["amount"])
    except InvalidOperation:
        return "amount must be a decimal number"
    if not amount.is_finite():
        return "amount must be finite"

    return None
