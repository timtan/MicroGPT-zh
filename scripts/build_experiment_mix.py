"""Create the reproducible 70% Jin Yong / 30% Taiwan-name fine-tuning mix."""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jinyong", type=Path, default=Path("input.txt"))
    parser.add_argument("--general", type=Path, default=Path("data/taiwan_general_names.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/experiment_c_mix.txt"))
    parser.add_argument("--size", type=int, default=10_000)
    parser.add_argument("--jinyong-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.jinyong_ratio < 1:
        parser.error("--jinyong-ratio must be between 0 and 1")

    jinyong = read_names(args.jinyong)
    general = read_names(args.general)
    rng = random.Random(args.seed)
    jinyong_count = round(args.size * args.jinyong_ratio)
    general_count = args.size - jinyong_count

    jinyong_rows = [jinyong[index % len(jinyong)] for index in range(jinyong_count)]
    general_rows = rng.sample(general, general_count)
    rows = jinyong_rows + general_rows
    rng.shuffle(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(
        f"wrote {len(rows)} rows: {jinyong_count} Jin Yong ({args.jinyong_ratio:.0%}), "
        f"{general_count} Taiwan general"
    )


if __name__ == "__main__":
    main()
