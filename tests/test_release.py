from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VERIFY_SPEC = importlib.util.spec_from_file_location("release_verify", ROOT / "code" / "verify.py")
assert VERIFY_SPEC is not None and VERIFY_SPEC.loader is not None
VERIFY_MODULE = importlib.util.module_from_spec(VERIFY_SPEC)
VERIFY_SPEC.loader.exec_module(VERIFY_MODULE)
verify_inputs = VERIFY_MODULE.verify_inputs
verify_publication_contracts = VERIFY_MODULE.verify_publication_contracts


def test_analysis_ready_inputs_are_frozen() -> None:
    assert verify_inputs(ROOT) >= 10


def test_result_manifest_has_exactly_two_reported_only_exceptions() -> None:
    manifest = pd.read_csv(ROOT / "code" / "results_manifest.csv")
    exceptions = manifest.loc[manifest.status.eq("reported_only"), "result_id"].tolist()
    assert exceptions == ["llm-extraction", "reporting-missingness"]


def test_publication_figure_contracts() -> None:
    assert verify_publication_contracts(ROOT) == 13
