"""Regenerate the binary (PDF/DOCX) files in data/sample_corpus.

Usage: python scripts/make_sample_corpus.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.docgen import write_docx, write_pdf  # noqa: E402

OUT = ROOT / "data" / "sample_corpus"

HANDBOOK_PAGES = [
    "Zephyr Robotics Employee Handbook (2025 edition)\n"
    "Section 1 - Paid time off. Full-time employees receive 27 days of paid time off (PTO) per "
    "calendar year, in addition to Estonian public holidays. A maximum of 5 unused PTO days may "
    "be carried over into the next year; any remaining days expire on 31 March.\n"
    "Section 2 - Parental leave. Employees with at least six months of service are entitled to "
    "18 weeks of fully paid parental leave.",
    "Section 3 - Remote work. Employees may work remotely up to 3 days per week with approval "
    "from their manager. Hardware lab and warehouse roles are excluded from remote work.\n"
    "Section 4 - Travel and expenses. Meal expenses on business trips are reimbursed up to $65 "
    "per day; amounts above the daily limit are not reimbursed. Receipts are required for any "
    "single expense over $25. Expense claims must be submitted in the Ledgerly app within 30 "
    "days of the expense date.",
]

SPEC_PARAGRAPHS = [
    "Zephyr X2 Product Specification",
    "The Zephyr X2 is our second-generation autonomous inspection drone, launched in March 2024. "
    "It replaces the Zephyr X1 and is designed for indoor warehouses and outdoor industrial sites.",
    "Flight time: the X2 achieves up to 47 minutes of flight on a single battery, compared with "
    "31 minutes for the Zephyr X1.",
    "Environmental limits: the X2 is rated IP54 and operates in temperatures from -10 °C to 45 °C. "
    "Operation outside this range is not supported.",
    "List price: $3,499 per unit (excluding support plans).",
]

SPEC_TABLE = [
    ["Specification", "Zephyr X2", "Zephyr X1"],
    ["Flight time", "47 min", "31 min"],
    ["Max payload", "2.3 kg", "1.2 kg"],
    ["Takeoff weight", "4.1 kg", "3.6 kg"],
    ["Ingress rating", "IP54", "IP43"],
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_pdf(OUT / "employee_handbook.pdf", HANDBOOK_PAGES)
    write_docx(OUT / "zephyr_x2_spec.docx", SPEC_PARAGRAPHS, table=SPEC_TABLE)
    print(f"Wrote PDF and DOCX files to {OUT}")


if __name__ == "__main__":
    main()
