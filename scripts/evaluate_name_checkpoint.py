"""Generate a fixed sample batch and report objective name-generation metrics."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path


def read_names(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("input.txt"))
    parser.add_argument("--vocab-input", type=Path, action="append", default=[])
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--sample-seed", type=int, default=20260728)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    command = [
        sys.executable,
        "microgpt.py",
        "--inference",
        "--checkpoint",
        str(args.checkpoint),
        "--input",
        str(args.input),
        "--samples",
        str(args.samples),
        "--sample-seed",
        str(args.sample_seed),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
    ]
    for path in args.vocab_input:
        command.extend(("--vocab-input", str(path)))

    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    samples = [
        line.split(":", 1)[1].strip()
        for line in completed.stdout.splitlines()
        if line.startswith("sample ")
    ]
    if len(samples) != args.samples:
        raise ValueError(f"expected {args.samples} samples, got {len(samples)}")

    counts = Counter(samples)
    references = {str(path): read_names(path) for path in args.reference}
    result = {
        "label": args.label,
        "checkpoint": str(args.checkpoint),
        "sampling": {
            "samples": args.samples,
            "sample_seed": args.sample_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "metrics": {
            "unique": len(counts),
            "duplicate_draws": args.samples - len(counts),
            "empty": counts.get("", 0),
            "mean_length": statistics.mean(map(len, samples)),
            "lengths": dict(sorted(Counter(map(len, samples)).items())),
            "reference_hits": {
                name: sum(sample in values for sample in samples)
                for name, values in references.items()
            },
        },
        "most_common": counts.most_common(20),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
