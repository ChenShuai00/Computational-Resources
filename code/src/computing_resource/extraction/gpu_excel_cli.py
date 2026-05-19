from __future__ import annotations

import argparse
from pathlib import Path

from computing_resource.extraction.gpu_excel_export import export_normalized_gpu_excel
from computing_resource.extraction.gpu_excel_renormalize import renormalize_gpu_excel


DEFAULT_CATALOG_PATH = Path("config/ml_hardware/ml_hardware.xlsx")


def _add_export_arguments(
    parser: argparse.ArgumentParser,
    *,
    input_required: bool,
    output_required: bool,
    catalog_required: bool,
) -> None:
    parser.add_argument("--input-dir", required=input_required, type=Path)
    parser.add_argument("--output-path", required=output_required, type=Path)
    parser.add_argument("--catalog-path", required=catalog_required, type=Path)


def _add_renormalize_arguments(
    parser: argparse.ArgumentParser,
    *,
    input_required: bool,
    catalog_required: bool,
) -> None:
    parser.add_argument("--input-path", required=input_required, type=Path)
    parser.add_argument("--catalog-path", required=catalog_required, type=Path)
    destination_group = parser.add_mutually_exclusive_group()
    destination_group.add_argument("--output-path", type=Path)
    destination_group.add_argument("--in-place", action="store_true")
    parser.add_argument("--default-variant-rules-path", type=Path)


def _add_export_and_renormalize_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--conference")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--catalog-path", type=Path)
    parser.add_argument("--export-output-path", type=Path)
    parser.add_argument("--renormalized-output-path", type=Path)
    parser.add_argument("--default-variant-rules-path", type=Path)


def build_export_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export normalized GPU extraction results to an Excel workbook."
    )
    _add_export_arguments(
        parser,
        input_required=True,
        output_required=True,
        catalog_required=True,
    )
    return parser


def build_renormalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-normalize an exported GPU Excel workbook with the latest rules and catalog."
    )
    _add_renormalize_arguments(
        parser,
        input_required=True,
        catalog_required=True,
    )
    return parser


def build_export_and_renormalize_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export normalized GPU extraction results to Excel and immediately re-normalize the workbook."
    )
    _add_export_and_renormalize_arguments(parser)
    return parser


def resolve_export_and_renormalize_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    conference = args.conference
    conference_root = Path("data/processed/gpu") / conference if conference else None

    input_dir = args.input_dir or (conference_root / "extract" if conference_root else None)
    catalog_path = args.catalog_path or DEFAULT_CATALOG_PATH
    export_output_path = args.export_output_path or (
        conference_root / f"{conference}_gpu.xlsx" if conference_root else None
    )
    renormalized_output_path = args.renormalized_output_path or (
        conference_root / f"{conference}_gpu_normalized.xlsx" if conference_root else None
    )

    missing_args = [
        name
        for name, value in (
            ("input-dir", input_dir),
            ("export-output-path", export_output_path),
            ("renormalized-output-path", renormalized_output_path),
        )
        if value is None
    ]
    if missing_args:
        raise ValueError(
            "Missing required arguments: "
            + ", ".join(f"--{name}" for name in missing_args)
            + ". Provide --conference or pass explicit paths."
        )

    return {
        "input_dir": input_dir,
        "catalog_path": catalog_path,
        "export_output_path": export_output_path,
        "renormalized_output_path": renormalized_output_path,
        "default_variant_rules_path": args.default_variant_rules_path,
    }


def export_and_renormalize_gpu_excel(
    *,
    input_dir: str | Path,
    catalog_path: str | Path,
    export_output_path: str | Path,
    renormalized_output_path: str | Path,
    default_variant_rules_path: str | Path | None = None,
) -> Path:
    exported_path = export_normalized_gpu_excel(
        input_dir=input_dir,
        output_path=export_output_path,
        catalog_path=catalog_path,
    )
    return renormalize_gpu_excel(
        input_path=exported_path,
        catalog_path=catalog_path,
        output_path=renormalized_output_path,
        default_variant_rules_path=default_variant_rules_path,
    )


def _handle_export(args: argparse.Namespace) -> Path:
    output_path = export_normalized_gpu_excel(
        input_dir=args.input_dir,
        output_path=args.output_path,
        catalog_path=args.catalog_path,
    )
    print(f"wrote {output_path}")
    return output_path


def _handle_renormalize(args: argparse.Namespace) -> Path:
    output_path = renormalize_gpu_excel(
        input_path=args.input_path,
        catalog_path=args.catalog_path,
        output_path=args.output_path,
        in_place=args.in_place,
        default_variant_rules_path=args.default_variant_rules_path,
    )
    print(f"wrote {output_path}")
    return output_path


def _handle_export_and_renormalize(args: argparse.Namespace) -> Path:
    paths = resolve_export_and_renormalize_paths(args)
    output_path = export_and_renormalize_gpu_excel(
        input_dir=paths["input_dir"],
        catalog_path=paths["catalog_path"],
        export_output_path=paths["export_output_path"],
        renormalized_output_path=paths["renormalized_output_path"],
        default_variant_rules_path=paths["default_variant_rules_path"],
    )
    print(f"wrote {paths['export_output_path']}")
    print(f"wrote {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU Excel analysis utilities for export and re-normalization workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        description="Export normalized GPU extraction results to an Excel workbook.",
        help="Export normalized GPU extraction results to Excel.",
    )
    _add_export_arguments(
        export_parser,
        input_required=True,
        output_required=True,
        catalog_required=True,
    )
    export_parser.set_defaults(handler=_handle_export)

    renormalize_parser = subparsers.add_parser(
        "renormalize",
        description="Re-normalize an exported GPU Excel workbook with the latest rules and catalog.",
        help="Re-normalize an exported GPU Excel workbook.",
    )
    _add_renormalize_arguments(
        renormalize_parser,
        input_required=True,
        catalog_required=True,
    )
    renormalize_parser.set_defaults(handler=_handle_renormalize)

    export_and_renormalize_parser = subparsers.add_parser(
        "export-and-renormalize",
        description="Export normalized GPU extraction results to Excel and immediately re-normalize the workbook.",
        help="Export GPU extraction results and immediately re-normalize the workbook.",
    )
    _add_export_and_renormalize_arguments(export_and_renormalize_parser)
    export_and_renormalize_parser.set_defaults(handler=_handle_export_and_renormalize)

    return parser


def main_export(argv: list[str] | None = None) -> Path:
    args = build_export_parser().parse_args(argv)
    return _handle_export(args)


def main_renormalize(argv: list[str] | None = None) -> Path:
    args = build_renormalize_parser().parse_args(argv)
    return _handle_renormalize(args)


def main_export_and_renormalize(argv: list[str] | None = None) -> Path:
    parser = build_export_and_renormalize_parser()
    args = parser.parse_args(argv)
    try:
        return _handle_export_and_renormalize(args)
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("unreachable")


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ValueError as exc:
        if args.command == "export-and-renormalize":
            parser.error(str(exc))
        raise
