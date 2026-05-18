# computing_resource

Batch utilities for processing ACL/EMNLP paper PDFs, fetching paper metadata, parsing checklists, and extracting sections, affiliations, and compute-resource information.

## Directory Overview

- `src/computing_resource/`: main application code
- `scripts/`: thin CLI entry points
- `config/default.yaml`: default configuration
- `data/raw/`: raw PDFs
- `data/external/annotations/`: human-curated or externally supplied annotations
- `data/external/metadata/`: fetched external metadata
- `data/interim/`: intermediate parse and section outputs
- `data/processed/`: final GPU, affiliation, and merged outputs
- `artifacts/`: logs, caches, and temporary files
- `docs/project-structure.md`: project structure notes

## Dependency Installation

```powershell
pip install pyyaml requests pypdf pdfplumber openai pytest
```

To let ACL event fetching prefer the official Python library before falling back to HTML parsing, install the optional package:

```powershell
pip install acl-anthology
```

## Environment Variables

LLM API keys are no longer stored in the default configuration. The GPU extraction script reads the `llm` settings from `config/default.yaml`; the current default uses OpenRouter.

If you use the current default configuration (OpenRouter):

```powershell
$env:OPENROUTER_API_KEY="your_openrouter_key"
```

If you want to use the original DeepSeek endpoint:

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
```

For OpenAlex / Semantic Scholar:

```powershell
$env:OPENALEX_API_KEY="your_openalex_key"
$env:OPENALEX_EMAIL="you@example.com"
$env:SEMANTIC_SCHOLAR_API_KEY="your_semantic_scholar_key"
```

## Workflow Overview

- `scripts/fetch_metadata.py acl-bundle`: fetch ACL metadata; also downloads PDFs when `--download-pdf` is passed explicitly.
- `scripts/fetch_metadata.py enrich`: merge ACL, OpenAlex, and Semantic Scholar results into the unified metadata directory.
- `scripts/fetch_metadata.py references-enrich`: enrich reference metadata and write it to `data/interim/reference_metadata/`.
- `scripts/parse_pdfs.py`: batch-parse PDFs through the hosted MinerU API.
- `scripts/extract_affiliations.py gpt-md`: extract author affiliations from the front matter of MinerU `full.md` files.
- `scripts/extract_gpu.py` / `scripts/extract_resources.py gpu`: extract GPU / TPU / hardware model names and counts from papers.

## ACL Metadata and PDF Download

Use the unified `acl-bundle` entry point:

```powershell
python scripts/fetch_metadata.py acl-bundle --url https://aclanthology.org/2025.emnlp-main.1/
python scripts/fetch_metadata.py acl-bundle --url https://aclanthology.org/events/emnlp-2025/
python scripts/fetch_metadata.py acl-bundle --url https://aclanthology.org/2025.emnlp-main.1/ --download-pdf
python scripts/fetch_metadata.py acl-bundle --url https://aclanthology.org/events/emnlp-2025/ --download-pdf
```

Notes:

- By default, the command fetches only ACL metadata JSON; PDFs are downloaded only when `--download-pdf` is passed explicitly.
- With a paper-page URL, the command fetches that paper's ACL metadata JSON. With `--download-pdf`, it also downloads the PDF.
- With an event URL, the command discovers main-track paper pages and fetches each paper's JSON in batch. With `--download-pdf`, it also downloads PDFs and writes an `index.json`.
- Event discovery first tries the official `acl-anthology` Python library. If the library is not installed, fails to initialize, or cannot resolve the event, the command falls back to HTML parsing.
- Metadata is written to `data/external/metadata/acl/` by default.
- PDFs are written to `data/raw/papers/<venue><year>/` by default, for example `data/raw/papers/emnlp2024/`.
- Without `--download-pdf`, the command only checks and fills JSON metadata.
- Re-running the same command with `--download-pdf` skips complete existing `JSON + PDF` pairs. If only one artifact is missing, it fills only the missing artifact. In event mode, failures for individual papers do not stop the whole batch; re-run the command to continue unfinished items.

## OpenAlex / Semantic Scholar Enrichment

After ACL metadata has been fetched, run the unified enrichment entry point:

```powershell
python scripts/fetch_metadata.py enrich --conference 2025.emnlp-main
python scripts/fetch_metadata.py enrich --conference 2025.emnlp-main --sources openalex semantic-scholar
python scripts/fetch_metadata.py enrich --conference 2025.emnlp-main --sources merge
```

Notes:

- The default order is `openalex -> semantic-scholar -> merge`.
- `merge` writes ACL, OpenAlex, and Semantic Scholar metadata together under `data/processed/merged/`.
- Use `--sources` to rerun only a specific source or step.
- `openalex`, `semantic-scholar`, and `merge` continue on per-file failures. Each output directory receives a `failed_files.json` when failures occur.

## Reference Enrichment

After `semantic-scholar` has completed, write each paper's references as standalone JSON files:

```powershell
python scripts/fetch_metadata.py references-enrich --conference 2025.emnlp-main
python scripts/fetch_metadata.py references-enrich --conference 2025.emnlp-main --batch-size 20 --throttle-seconds 2
python scripts/fetch_metadata.py references-enrich --conference 2025.emnlp-main --enable-title-fallback --batch-size 20 --throttle-seconds 2
python scripts/fetch_metadata.py references-enrich --conference 2025.emnlp-main --overwrite --batch-size 20 --throttle-seconds 2
```

Notes:

- The default input is `data/external/metadata/semantic_scholar/<conference>/`.
- The main path collects references with `paperId` values and enriches them through Semantic Scholar `paper/batch`.
- `paper/batch` requests detailed reference fields, including `corpusId`, `externalIds`, `url`, `abstract`, `venue`, `publicationVenue`, `publicationDate`, `publicationTypes`, `citationCount`, `influentialCitationCount`, `referenceCount`, `isOpenAccess`, `openAccessPdf`, `fieldsOfStudy`, `s2FieldsOfStudy`, `journal`, `citationStyles`, `authors`, and `references`.
- References without `paperId` are not resolved by title fallback by default; they are retained in `unresolved_references.json`.
- Title fallback is enabled only with `--enable-title-fallback`. It uses strict `title + year` recovery, accepts only normalized exact title matches, performs no fuzzy matching, and does not expand into cross-source broad search.
- Each enriched reference is written to `data/interim/reference_metadata/<conference>/<paperId>.json`.
- Successful runs also write `data/interim/reference_metadata/<conference>/index.json` and `unresolved_references.json`. The index records `reference_ids_resolved_by_title`.
- Without `--overwrite`, existing `<paperId>.json` files are skipped so the command can resume from already written detail files.
- With `--enable-title-fallback`, the command also writes `data/interim/reference_metadata/<conference>/title_fallback_state.json` for resumable unresolved-reference recovery.
- Detail files are written continuously by batch. If the process is interrupted, already written `<paperId>.json` files are retained and the next run continues from the remaining items.
- `--overwrite` rewrites existing `<paperId>.json` files. This is useful after requested fields change, but disables the detail-file skip/resume behavior for already completed files.
- Use `--batch-size` and `--throttle-seconds` to control throughput and throttling. For `429` or `SSLEOFError`, try `--batch-size 20 --throttle-seconds 2`, then reduce to `--batch-size 10 --throttle-seconds 3` if needed.
- Transient errors such as `429`, `SSLError`, `ConnectionError`, and `Timeout` are retried automatically. If a run is still interrupted, re-run the command to continue.
- For batch reference enrichment, run `semantic-scholar` first, then run this command.

## PDF Parsing

### Hosted MinerU Batch API

`mineru-hosted-api` uploads PDFs, polls hosted parse results, downloads the zip file returned by `full_zip_url`, and extracts it into the same directory structure.

Full workflow:

```powershell
$env:MINERU_API_TOKEN="your_token"
python scripts/parse_pdfs.py mineru-hosted-api --input-dir data/raw/papers/emnlp2025 --conference emnlp2025
python scripts/parse_pdfs.py mineru-hosted-api --input-dir data/raw/papers/emnlp2025 --conference emnlp2025 --batch-size 100 --poll-interval 10 --download-workers 8
```

If upload and polling are already complete and you only want to download and extract from an existing `all_extract_results.json`:

```powershell
python scripts/parse_pdfs.py mineru-hosted-api --output-dir data/interim/parses/emnlp2024_mineru --download-only --download-workers 8
python scripts/parse_pdfs.py mineru-hosted-api --output-dir data/interim/parses/emnlp2024_mineru --download-only --extract-results-json data/interim/parses/emnlp2024_mineru/all_extract_results.json --download-workers 8
```

If existing results contain papers with `state=failed` and you want to rerun only those failures:

```powershell
$env:MINERU_API_TOKEN="your_token"
python scripts/parse_pdfs.py mineru-hosted-api --input-dir data/raw/papers/emnlp2024 --output-dir data/interim/parses/emnlp2024_mineru --retry-failed --download-workers 8
```

Default outputs:

- extracted results under `data/interim/parses/emnlp2025/<paper_id>/auto/`
- upload records and per-batch results under `data/interim/parses/emnlp2025/batches/`
- `data/interim/parses/emnlp2025/all_extract_results.json`
- `data/interim/parses/emnlp2025/all_extract_results_summary.csv`
- `data/interim/parses/emnlp2025/parse_summary.json`

Notes:

- `--download-workers` controls concurrent downloads.
- `--max-download-retries` controls retries after a single zip download fails.
- `--download-only` reads `<output_dir>/all_extract_results.json` and only downloads/extracts results; it does not upload PDFs again.
- `--retry-failed` reads an existing result file, resubmits only PDFs with `state=failed`, and continues downloading new successful results.
- `--extract-results-json` explicitly selects the result file path, which is useful when the file is not in the default location.
- The command prints batch and download progress continuously.

## Affiliation Workflow

Workflow assumptions:

- The input is a local MinerU result directory with one subdirectory per paper.
- Affiliation extraction supports only `gpt-md` and uses only the content before `Abstract` in each paper's `full.md`.
- The repository currently keeps only the extraction step. Institution normalization / standard-library matching is no longer built in.

Input:

Local MinerU result directory: `data/interim/parses/emnlp2024/`

Command:

```powershell
python scripts/extract_affiliations.py gpt-md --input-dir data/interim/parses/emnlp2024/ --conference emnlp2024 --output-dir data/processed/affiliations/emnlp2024
```

Main outputs:

- `data/processed/affiliations/emnlp2024/affiliations.jsonl`
- `data/processed/affiliations/emnlp2024/affiliations.csv`
- `data/processed/affiliations/emnlp2024/affiliations.xlsx`

Notes:

- Only the extraction stage is retained. If institution normalization is needed later, add it through a separate rewritten workflow.
- `affiliations.jsonl` is the primary machine-readable output. `affiliations.csv` and `affiliations.xlsx` are intended mainly for manual review.
- When `gpt-md` is re-run and `affiliations.csv` already exists in the output directory, the command resumes automatically and reruns only papers marked `failed`, `missing`, or with empty affiliation results.
- `gpt-md` prints `completed/total` progress while running.

## GPU Extraction Workflow

The GPU extraction workflow aims for high-recall extraction of `GPU/TPU/hardware model + count` from parsed paper directories:

- `data/interim/parses/<conference>/<paper_id>/auto/full.md`

### Extraction Strategy

GPU extraction uses a layered, high-recall-first strategy with post-processing denoising. It does not rely on a single full-text LLM extraction pass:

1. Candidate recall: convert parse results into a structured `section_doc`, then recall candidate windows by section title, appendix patterns, hardware keywords, count patterns, and hardware-catalog aliases. Recall rules are divided into `strong_keep`, `soft_skip`, and `hard_skip` in `config/gpu_section_rules.yaml`.
2. Local extraction: call the LLM only on candidate windows, not on the whole paper. The model extracts only `raw_hardware_name + count` from the source text. Compound mentions such as `A100/RTX-6000`, `A100 or H100`, and `4 x A100 and 2 x H100` are split first.
3. Normalization: extracted `raw_hardware_name` values are matched and normalized with `ml_hardware.xlsx`. The catalog improves recall and normalization but does not veto raw extraction results. The main path is `exact/alias/rule match`; embeddings are used only to enhance candidate recall and do not directly choose the final canonical name.
4. Audit and gap filling: windows that match hardware keywords but produce no extraction are written to `review_flags`; model-output parse failures are not silently dropped. Outputs retain `candidate_windows`, `raw_extractions`, `normalized_extractions`, and `review_flags` for traceability.
5. Rule iteration: after a full conference run, section statistics produce `gpu_rule_stats_<conference>.csv` and `gpu_rule_candidates_<conference>.csv`. Approved candidate titles can then be merged back into `config/gpu_section_rules.yaml`.

### Full-Text Fallback Strategy

GPU extraction also includes a full-text fallback layer. It fills gaps but does not replace the main path:

1. Run the section-level main path first and prefer hardware results from `candidate_windows`.
2. Trigger full-text fallback whenever the main-path result is empty. Also trigger it when the main-path result still needs manual review or has produced `review_flags`.
3. Full-text fallback does not rescan the raw `full.md` as-is. It first removes candidate-window text already consumed successfully by the main path, then keeps the remaining full text.
4. Papers with empty results run gap-filling extraction over fixed-length chunks of the remaining text. Papers with non-empty but suspicious results still use hardware signals in the remaining text to control noise.
5. Fallback results are marked with `extraction_source = fulltext_fallback`, written to `fulltext_fallback_extractions`, and merged back into top-level `normalized_extractions`.

This design covers missed sections, main-path parse failures, and scattered hardware mentions while avoiding duplicate processing of sections already extracted by the main path.

### Running GPU Extraction Directly

The direct entry point is:

```powershell
python scripts/extract_gpu.py --input-dir data/interim/parses/emnlp2024 --output-dir data/processed/gpu/emnlp2024
```

To enable paper-level concurrency, pass `--workers`:

```powershell
python scripts/extract_gpu.py --input-dir data/interim/parses/emnlp2024 --output-dir data/processed/gpu/emnlp2024 --workers 4
```

You can also use the unified resource-extraction entry point:

```powershell
python scripts/extract_resources.py gpu --input-dir data/interim/parses/emnlp2024 --output-dir data/processed/gpu/emnlp2024
```

To explicitly use DeepSeek, override only the model name:

```powershell
python scripts/extract_gpu.py --input-dir data/interim/parses/emnlp2024 --output-dir data/processed/gpu/emnlp2024 --model deepseek-chat
```

Notes:

- If `--output-dir` is omitted, output defaults to `data/processed/gpu/<input_dir_name>/`.
- Each paper in the input directory produces one `<paper_id>_gpu.json`.
- Existing output files are skipped by default, so the same directory can be re-run to continue unfinished papers.
- `--overwrite` rewrites existing results.
- `--workers` controls paper-level concurrency and defaults to `1`. With `workers > 1`, multiple papers are processed concurrently, but each single paper is still processed sequentially.
- By default, the command reads `llm.model`, `llm.api_base`, and `llm.api_key_env` from `config/default.yaml`; the current default is OpenRouter.
- `--model` is the only runtime model-override option.
- The workflow requires the `openai` Python SDK and the environment variable required by the selected model.
- When `deepseek-chat` is used explicitly, the command switches back to the original DeepSeek route: `https://api.deepseek.com` + `DEEPSEEK_API_KEY`.
- Recommended final outputs should be written to `data/processed/gpu/<conference>/`, for example `data/processed/gpu/emnlp2024`.

### Output Structure

Each output JSON mainly contains:

- `candidate_windows`: why a section / appendix window entered extraction
- `raw_extractions`: raw LLM extraction results
- `normalized_extractions`: final results after catalog normalization
- `review_flags`: items that need manual review or audit
- `pred_result`: summary field retained for compatibility with the old workflow

### GPU Excel CLI

GPU Excel analysis now uses a unified subcommand script:

```powershell
python scripts/analysis/gpu_excel.py export --input-dir data/processed/gpu/emnlp2020/extract --output-path data/processed/gpu/emnlp2020/emnlp2020_gpu.xlsx --catalog-path data/processed/gpu/ml_hardware/ml_hardware.xlsx
python scripts/analysis/gpu_excel.py renormalize --input-path data/processed/gpu/emnlp2020/emnlp2020_gpu.xlsx --catalog-path data/processed/gpu/ml_hardware/ml_hardware.xlsx --output-path data/processed/gpu/emnlp2020/emnlp2020_gpu_normalized.xlsx
python scripts/analysis/gpu_excel.py export-and-renormalize --conference emnlp2020
```

Notes:

- `export`: aggregate `*_gpu.json` files into a unified Excel template.
- `renormalize`: recompute benchmark and normalization columns in an existing Excel workbook without rerunning extraction.
- `export-and-renormalize`: export first, then immediately re-normalize with the latest catalog and rules.
- `export-and-renormalize --conference emnlp2020` reads `data/processed/gpu/emnlp2020/extract` by default.
- The default catalog is `data/processed/gpu/ml_hardware/ml_hardware.xlsx`.
- The combined command first writes `data/processed/gpu/emnlp2020/emnlp2020_gpu.xlsx`, then writes `data/processed/gpu/emnlp2020/emnlp2020_gpu_normalized.xlsx`.
- The output Excel file uses a unified template with `gpu_name`, `gpu_num`, `benchmark_gpu_name`, `gpu_vendor`, `product_category`, `benchmark_*`, `normalize_status`, and `normalize_reason`.
- Benchmark fields come from `data/processed/gpu/ml_hardware/ml_hardware.xlsx`.
- To change the input directory, catalog, or output filename, pass full paths explicitly.
- Legacy entry points `export_normalized_gpu_excel.py`, `renormalize_gpu_excel.py`, and `export_and_renormalize_gpu_excel.py` are still kept as compatibility wrappers.

### Rules and Catalog

GPU extraction uses two updateable configuration sources:

- section recall rules: `config/gpu_section_rules.yaml`
- hardware catalog: `data/processed/gpu/paper_section_gpu/ml_hardware/ml_hardware.xlsx`

Details:

- `gpu_section_rules.yaml` controls strong-keep, soft-skip, and hard-skip section titles, as well as keywords and count patterns.
- `ml_hardware.xlsx` is used for alias matching and normalization, but it does not veto raw extraction results.
- To update `strong_keep_titles`, run rule analysis first, then merge approved candidate titles with `update_gpu_section_rules.py`.

### Excel Re-Normalization

If a normalized Excel workbook already exists and you later add shared normalization rules or default variant rules, re-normalize the existing Excel file without rerunning extraction:

```powershell
python scripts/analysis/gpu_excel.py renormalize --input-path data/processed/gpu/emnlp2020/emnlp2020_gpu.xlsx --catalog-path data/processed/gpu/ml_hardware/ml_hardware.xlsx --output-path data/processed/gpu/emnlp2020/emnlp2020_gpu_normalized.xlsx
```

Notes:

- This command recomputes only the normalization columns in Excel; it does not rerun LLM extraction.
- It rewrites `benchmark_gpu_name`, `gpu_vendor`, `product_category`, all `benchmark_*` columns, `normalize_status`, and `normalize_reason`.
- Rows whose final `benchmark_gpu_name == cpu` are removed before saving.
- Re-normalization uses the shared resolver, so the main extraction chain and Excel re-normalization share the same rules.
- If `config/gpu_default_benchmark_variants.yaml` exists, the command loads it automatically without extra arguments.

To explicitly specify another default-variant rules file:

```powershell
python scripts/analysis/gpu_excel.py renormalize --input-path data/processed/gpu/emnlp2020/emnlp2020_gpu.xlsx --catalog-path data/processed/gpu/ml_hardware/ml_hardware.xlsx --output-path data/processed/gpu/emnlp2020/emnlp2020_gpu_normalized.xlsx --default-variant-rules-path config/gpu_default_benchmark_variants.yaml
```

### Default Variant Rules

For names where the family is clear but the benchmark variant is still ambiguous, such as `NVIDIA A100` or `V100`, default variant rules are maintained in:

- `config/gpu_default_benchmark_variants.yaml`

After updating this file, run `gpu_excel.py renormalize` again and benchmark variants will be backfilled automatically. No conference-specific workflow switch is required.

If you need to generate a candidate table from an existing Excel workbook or export a rules file, the repository still keeps these helper scripts:

```powershell
python scripts/analysis/export_default_variant_candidates.py --input-path data/processed/gpu/emnlp2020/emnlp2020_gpu_normalized.xlsx --catalog-path data/processed/gpu/ml_hardware/ml_hardware.xlsx --output-path data/processed/gpu/emnlp2020/emnlp2020_default_variant_candidates.xlsx
python scripts/analysis/export_default_variant_rules.py --input-path data/processed/gpu/emnlp2020/emnlp2020_default_variant_candidates.xlsx --output-path config/gpu_default_benchmark_variants.yaml
```

### Post-Run Rule Analysis

After running a full conference, export section hit statistics and candidate rule tables:

```powershell
python scripts/analysis/analyze_gpu_section_rules.py --input-dir data/processed/gpu/emnlp2024 --conference emnlp2024
```

Default outputs:

- `artifacts/analysis/gpu_rule_stats_emnlp2024.csv`
- `artifacts/analysis/gpu_rule_candidates_emnlp2024.csv`

These tables can be used to continue updating `config/gpu_section_rules.yaml`.

To merge candidate titles with `suggested_action=promote_to_strong_keep` directly into the formal rules file:

```powershell
python scripts/analysis/update_gpu_section_rules.py --rules config/gpu_section_rules.yaml --candidates artifacts/analysis/gpu_rule_candidates_emnlp2024.csv --apply-action promote_to_strong_keep --in-place
```

To preview the merged rules without writing them:

```powershell
python scripts/analysis/update_gpu_section_rules.py --rules config/gpu_section_rules.yaml --candidates artifacts/analysis/gpu_rule_candidates_emnlp2024.csv --apply-action promote_to_strong_keep --dry-run
```

Notes:

- The automatic update currently handles only `strong_keep_titles`.
- The command deduplicates and sorts entries. It does not automatically modify `soft_skip_titles` or `hard_skip_titles`.
- Review `gpu_rule_candidates_<conference>.csv` before deciding whether to use `--in-place`.
