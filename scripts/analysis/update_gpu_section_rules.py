from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update GPU section rules from a candidate rule table")
    parser.add_argument("--rules", required=True, help="Path to gpu_section_rules.yaml")
    parser.add_argument("--candidates", required=True, help="Path to gpu_rule_candidates_<conference>.csv")
    parser.add_argument("--apply-action", default="promote_to_strong_keep")
    parser.add_argument("--dry-run", action="store_true", help="Print merged rules without writing them")
    parser.add_argument("--in-place", action="store_true", help="Update the rules file in place")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def _load_rules(rules_path: str | Path) -> dict:
    return yaml.safe_load(Path(rules_path).read_text(encoding="utf-8")) or {}


def _load_candidate_titles(candidates_path: str | Path, apply_action: str) -> list[str]:
    with Path(candidates_path).open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        return sorted(
            {
                (row.get("normalized_section_title") or "").strip()
                for row in rows
                if (row.get("suggested_action") or "").strip() == apply_action
                and (row.get("normalized_section_title") or "").strip()
            }
        )


def collect_updated_rules(
    rules_path: str | Path,
    candidates_path: str | Path,
    apply_action: str = "promote_to_strong_keep",
) -> dict:
    rules = _load_rules(rules_path)
    current_titles = set(rules.get("strong_keep_titles", []))
    candidate_titles = _load_candidate_titles(candidates_path, apply_action)
    merged_titles = sorted(current_titles.union(candidate_titles))
    updated = dict(rules)
    updated["strong_keep_titles"] = merged_titles
    return updated


def apply_rule_updates(
    rules_path: str | Path,
    candidates_path: str | Path,
    apply_action: str = "promote_to_strong_keep",
) -> dict:
    updated = collect_updated_rules(rules_path, candidates_path, apply_action=apply_action)
    Path(rules_path).write_text(
        yaml.safe_dump(updated, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return updated


def main(argv=None) -> None:
    args = parse_args(argv)
    updated = collect_updated_rules(args.rules, args.candidates, apply_action=args.apply_action)
    if args.in_place:
        Path(args.rules).write_text(
            yaml.safe_dump(updated, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return
    print(yaml.safe_dump(updated, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
