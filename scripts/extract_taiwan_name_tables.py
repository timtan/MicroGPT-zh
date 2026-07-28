"""Extract aggregate Taiwan name tables from the MOI 112 name-statistics PDF text.

Run pdftotext with layout preservation first:

    pdftotext -layout 112namestat.pdf 112namestat.txt
    uv run python scripts/extract_taiwan_name_tables.py 112namestat.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def rows_between(lines: list[str], start: str, end: str) -> list[str]:
    # Ignore the table-of-contents entries near the start of the publication.
    start_index = next(i for i, line in enumerate(lines) if "\f" in line and start in line)
    end_index = next(
        i
        for i, line in enumerate(lines[start_index + 1 :], start_index + 1)
        if "\f" in line and end in line
    )
    return lines[start_index:end_index]


def next_nonempty(lines: list[str], index: int) -> tuple[int, str]:
    for next_index in range(index + 1, len(lines)):
        stripped = lines[next_index].strip()
        if stripped:
            return next_index, stripped
    raise ValueError("unexpected end of table")


def parse_ranked_rows(
    lines: list[str], label: str, value_label: str = "人數"
) -> list[tuple[int, str, int]]:
    rows: list[tuple[int, str, int]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("排名"):
            continue
        ranks = [int(value) for value in stripped.split()[1:] if value.isdigit()]
        label_index, names_line = next_nonempty(lines, index)
        if not names_line.startswith(label):
            continue
        _, values_line = next_nonempty(lines, label_index)
        if not values_line.startswith(value_label):
            continue
        names = names_line.split()[1:]
        values = [int(value.replace(",", "")) for value in values_line.split()[1:]]
        if not (len(ranks) == len(names) == len(values)):
            raise ValueError(f"misaligned {label} row near: {stripped}")
        rows.extend(zip(ranks, names, values))
    return rows


def write_csv(path: Path, header: tuple[str, ...], rows: list[tuple[object, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_text", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/sources"))
    args = parser.parse_args()

    # Keep form-feed page markers; str.splitlines() would discard them.
    lines = args.pdf_text.read_text(encoding="utf-8").split("\n")

    common_names = rows_between(lines, "表五十一", "表五十二")
    given_rows = [
        ("male", rank, name, population)
        for rank, name, population in parse_ranked_rows(common_names, "男性名")
    ] + [
        ("female", rank, name, population)
        for rank, name, population in parse_ranked_rows(common_names, "女性名")
    ]

    full_names = rows_between(lines, "表五十二", "表五十三")
    full_rows = [
        ("male", rank, name, population)
        for rank, name, population in parse_ranked_rows(full_names, "男性姓名")
    ] + [
        ("female", rank, name, population)
        for rank, name, population in parse_ranked_rows(full_names, "女性姓名")
    ]

    surname_lines = rows_between(lines, "表五十七", "內政部全國姓名統計分析工作人員表")
    surname_rows = parse_ranked_rows(surname_lines, "姓氏", value_label="計")

    write_csv(
        args.output_dir / "taiwan_given_names_112.csv",
        ("gender", "rank", "given_name", "population"),
        given_rows,
    )
    write_csv(
        args.output_dir / "taiwan_common_full_names_112.csv",
        ("gender", "rank", "full_name", "population"),
        full_rows,
    )
    write_csv(
        args.output_dir / "taiwan_surnames_112.csv",
        ("rank", "surname", "population"),
        surname_rows,
    )
    print(
        f"extracted {len(given_rows)} given names, {len(full_rows)} full names, "
        f"and {len(surname_rows)} surnames"
    )


if __name__ == "__main__":
    main()
