# Analysis

This directory contains the released datasets, scripts, figures, and report notes for the paper analyses.

## Main Data

- `data/`: shared paper-level and organization-level datasets used across the manuscript analyses.

## Analysis Sections

- `4.1/`: compute-resource reporting patterns over time and across venues.
- `4.2/`: GPU count, GPU model, GPU generation, memory, and reported peak TFLOPS analyses.
- `4.3/`: country-, institution-, and NLP-topic-level compute-resource analyses.
- `4.4/`: citation modeling and amplifier interaction analyses.

Most analysis modules keep their inputs, outputs, and notes together:

- `data/`: section-specific derived tables.
- `fig/`: generated figures.
- `report/`: brief result summaries or QA notes.
- `script/`: scripts used to generate the section outputs.

Some modules omit a subfolder when it is not needed.
