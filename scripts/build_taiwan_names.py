"""Build a Taiwan-style name corpus from a large Chinese-name candidate pool."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from opencc import OpenCC


RECENT_GIVEN_NAMES = {
    "承恩", "宥廷", "品睿", "宸睿", "宇恩", "宇翔", "承翰", "宥辰", "柏睿", "睿恩",
    "恩碩", "子睿", "子宸", "子恩", "品妍", "子晴", "詠晴", "品妤", "禹彤", "羽彤",
    "芯語", "宥蓁", "語彤", "苡晴", "苡菲", "雨霏", "芸菲", "苡安", "玥彤", "欣妤",
    "詩涵", "思妤", "宜蓁", "佳穎", "宜庭", "怡萱", "雅筑", "郁婷", "鈺婷", "柏諺",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def is_cjk(text: str) -> bool:
    return all("\u3400" <= char <= "\u9fff" for char in text)


def deterministic_unit_interval(text: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    integer = int.from_bytes(digest[:8], "big")
    return (integer + 1) / (2**64 + 1)


def allocate(total: int, weighted_items: list[tuple[str, int]]) -> dict[str, int]:
    weight_sum = sum(weight for _, weight in weighted_items)
    exact = [(item, total * weight / weight_sum) for item, weight in weighted_items]
    counts = {item: int(value) for item, value in exact}
    remainder = total - sum(counts.values())
    for item, _ in sorted(exact, key=lambda pair: pair[1] - int(pair[1]), reverse=True)[:remainder]:
        counts[item] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="Chinese_Names_Corpus_Gender text file")
    parser.add_argument("--official-dir", type=Path, default=Path("data/sources"))
    parser.add_argument("--jinyong", type=Path, default=Path("input.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/taiwan_general_names.txt"))
    parser.add_argument("--preview", type=Path, default=Path("data/taiwan_general_names_preview.txt"))
    parser.add_argument("--metadata", type=Path, default=Path("data/taiwan_general_names_metadata.json"))
    parser.add_argument("--size", type=int, default=20_000)
    parser.add_argument("--surname-count", type=int, default=80)
    parser.add_argument("--min-char-frequency", type=int, default=100)
    parser.add_argument("--min-given-surname-coverage", type=int, default=10)
    parser.add_argument("--single-name-ratio", type=float, default=0.025)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    surname_rows = read_csv(args.official_dir / "taiwan_surnames_112.csv")[: args.surname_count]
    surnames = {row["surname"]: int(row["population"]) for row in surname_rows}
    official_given_rows = read_csv(args.official_dir / "taiwan_given_names_112.csv")
    official_given = {row["given_name"] for row in official_given_rows}
    official_given_chars = set("".join(official_given | RECENT_GIVEN_NAMES))
    held_out = {
        row["full_name"]
        for row in read_csv(args.official_dir / "taiwan_common_full_names_112.csv")
    }
    jinyong = {
        line.strip()
        for line in args.jinyong.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    converter = OpenCC("s2twp")
    candidates: dict[str, str] = {}
    with args.source.open(encoding="utf-8-sig") as file:
        for raw_line in file:
            raw_line = raw_line.strip()
            if not raw_line or "," not in raw_line:
                continue
            raw_name, raw_gender = raw_line.rsplit(",", 1)
            if raw_gender not in {"男", "女"} or len(raw_name) not in {2, 3}:
                continue
            name = unicodedata.normalize("NFC", converter.convert(raw_name))
            if len(name) not in {2, 3} or name[0] not in surnames or not is_cjk(name):
                continue
            if name in held_out or name in jinyong:
                continue
            gender = "male" if raw_gender == "男" else "female"
            candidates.setdefault(name, gender)

    char_frequency = Counter("".join(name[1:] for name in candidates))
    candidates = {
        name: gender
        for name, gender in candidates.items()
        if all(
            char_frequency[char] >= args.min_char_frequency and char in official_given_chars
            for char in name[1:]
        )
    }

    given_surnames: dict[str, set[str]] = defaultdict(set)
    for name in candidates:
        given_surnames[name[1:]].add(name[0])
    given_coverage = {given: len(values) for given, values in given_surnames.items()}
    candidates = {
        name: gender
        for name, gender in candidates.items()
        if given_coverage[name[1:]] >= args.min_given_surname_coverage
        or name[1:] in official_given
        or name[1:] in RECENT_GIVEN_NAMES
    }

    by_bucket: dict[tuple[str, str, int], list[str]] = defaultdict(list)
    for name, gender in candidates.items():
        by_bucket[(name[0], gender, len(name) - 1)].append(name)

    surname_quota = allocate(args.size, list(surnames.items()))
    selected: list[str] = []
    for surname, quota in surname_quota.items():
        single_quota = round(quota * args.single_name_ratio)
        double_quota = quota - single_quota
        for gender, gender_quota in (("male", quota // 2), ("female", quota - quota // 2)):
            gender_single = single_quota // 2 if gender == "male" else single_quota - single_quota // 2
            gender_double = gender_quota - gender_single
            for given_length, bucket_quota in ((1, gender_single), (2, gender_double)):
                bucket = by_bucket[(surname, gender, given_length)]
                ranked: list[tuple[float, str]] = []
                for name in bucket:
                    given = name[1:]
                    weight = 1.0 + math.log1p(given_coverage[given])
                    if given in official_given:
                        weight *= 2.0
                    if given in RECENT_GIVEN_NAMES:
                        weight *= 1.5
                    unit = deterministic_unit_interval(name, args.seed)
                    ranked.append((-math.log(unit) / weight, name))
                selected.extend(name for _, name in sorted(ranked)[:bucket_quota])

    selected = sorted(set(selected))
    if len(selected) < args.size * 0.98:
        raise ValueError(f"only selected {len(selected)} of requested {args.size} names")

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(selected) + "\n", encoding="utf-8")

    preview_pool = selected.copy()
    rng.shuffle(preview_pool)
    preview_names = preview_pool[:100]
    args.preview.write_text("\n".join(preview_names) + "\n", encoding="utf-8")

    selected_given = {name[1:] for name in selected}
    selected_chars = set("".join(selected))
    metadata = {
        "source": "wainshine/Chinese-Names-Corpus (Apache-2.0)",
        "source_commit": "47d4af8d816f6212787ddfc49173cac3b994b58d",
        "seed": args.seed,
        "requested_size": args.size,
        "selected_names": len(selected),
        "unique_given_names": len(selected_given),
        "vocab_chars": len(selected_chars),
        "official_given_character_whitelist": len(official_given_chars),
        "minimum_given_surname_coverage": args.min_given_surname_coverage,
        "male": sum(candidates[name] == "male" for name in selected),
        "female": sum(candidates[name] == "female" for name in selected),
        "two_character_full_names": sum(len(name) == 2 for name in selected),
        "three_character_full_names": sum(len(name) == 3 for name in selected),
        "official_full_names_held_out": len(held_out),
        "jinyong_overlap": len(set(selected) & jinyong),
        "top_surnames": Counter(name[0] for name in selected).most_common(20),
        "top_given_names": Counter(name[1:] for name in selected).most_common(20),
    }
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
