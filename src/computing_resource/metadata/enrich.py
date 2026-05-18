import argparse

from computing_resource.metadata import merge, openalex, semantic_scholar


DEFAULT_SOURCES = ["openalex", "semantic-scholar", "merge"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OpenAlex / Semantic Scholar / merge metadata enrichment in sequence")
    parser.add_argument("--conference", required=True, help="conference/volume subdirectory, such as 2025.emnlp-main")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=DEFAULT_SOURCES,
        default=list(DEFAULT_SOURCES),
        help="Metadata sources or steps to run",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def build_source_argv(conference: str) -> list[str]:
    return ["--conference", conference]


def main(argv=None):
    args = parse_args(argv)
    source_argv = build_source_argv(args.conference)

    for source in args.sources:
        if source == "openalex":
            openalex.main(source_argv)
        elif source == "semantic-scholar":
            semantic_scholar.main(source_argv)
        elif source == "merge":
            merge.main(source_argv)


if __name__ == "__main__":
    main()
