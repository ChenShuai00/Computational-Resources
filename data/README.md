# Released Extraction Data

This directory contains released data derived from the paper corpus used in the study. It is organized into three parts:

- `gpu/`: extracted compute-resource mentions and normalized GPU/resource outputs by venue and year.
- `topics/`: classified paper-topic data and topic hierarchy files.
- `affiliations/`: extracted paper affiliation institutions by venue and year.

## Directory Layout

### `gpu/`

Compute-resource extraction outputs are grouped by conference-year folders such as `acl2024/`, `emnlp2025/`, and `naacl2025/`. These folders include:

- `*_gpu.xlsx`: extracted compute-resource records for a venue-year.
- `*_gpu_normalized.xlsx`: normalized resource records for downstream analysis.
- `extract/`: per-paper JSON extraction outputs.

The `gpu/compute_role/` folder contains files used to classify or audit the role of compute-resource mentions, including labeled workbooks, review queues, applied corrections, and snippet-level CSV exports.

### `topics/`

Topic classification data is stored in:

- `acl_arr_topics_all_acl_metadata.jsonl`: paper-level topic metadata in JSON Lines format.
- `acl_arr_topics_all_acl_metadata.xlsx`: the same topic metadata in workbook format.
- `topic_level.xlsx`: topic hierarchy and level information.

### `affiliations/`

Affiliation extraction outputs are grouped first by venue (`acl/`, `emnlp/`, `naacl/`) and then by year. Each venue-year folder contains aggregated affiliation records and per-paper JSON outputs. The top-level `all_affiliations.jsonl` file provides a combined JSON Lines export across venues and years.

## File Formats

- `.json`: per-paper extraction records.
- `.jsonl`: aggregated records, one JSON object per line.
- `.xlsx`: tabular workbooks for inspection and analysis.
- `.csv`: review, labeling, or audit tables.

## Scope

At the time of release, this directory contains approximately 28,000 files across compute-resource, topic, and affiliation extraction outputs. No file in this directory is expected to exceed GitHub's 100 MB single-file limit.
