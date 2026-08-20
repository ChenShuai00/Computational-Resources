# Repository migration

The publication release was assembled in a separate staging directory, fully
executed, and verified before replacing the root layout. No legacy artifact was
deleted.

The previous root trees were moved locally as follows:

| Previous path | Local archive path | Public Git status |
|---|---|---|
| `analysis/` | `_local/legacy/analysis/` | replaced by paper-first `results/` |
| `code/` | `_local/legacy/code/` | excluded; upstream PDF/API/LLM pipeline |
| `data/` | `_local/legacy/data/` | excluded; replaced by minimized `data/analysis_ready/` |
| `docs/`, `assets/` | `_local/legacy/docs/`, `_local/legacy/assets/` | excluded unless represented in the new release |
| `Findings/`, `Pooled/`, `outputs/` | `_local/legacy/` under the same names | excluded; not final-paper result scope |
| environments, tool state, and old root files | `_local/legacy/tooling/` and `_local/legacy/root_files/` | excluded |

`_local/legacy/manifests/legacy_inventory_before_move.csv` records pre-move file
counts and bytes. `legacy_key_hashes.csv` records SHA-256 digests for old root
files and the former `analysis/data/` inputs. These manifests are local-only.

The frozen manuscript source and figures were copied to `paper/` for local
cross-checking. The directory is intentionally ignored and is not part of the
public Git release. Publication figure copies required for result review live in
the appropriate `results/**/reference/publication_figures/` directory.
