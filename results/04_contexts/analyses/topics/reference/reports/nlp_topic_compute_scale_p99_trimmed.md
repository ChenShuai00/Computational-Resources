# RQ2 NLP Topic Compute Scale

## Figure contract

Core conclusion: NLP topics differ in reported maximum GPU-row compute scale, both in their central tendency and in how often their papers enter the within-year high-compute tail.

Evidence chain: panel a reports each topic's median, IQR, and P90 paper-level maximum GPU-row compute capacity; panel b reports the share of each topic's papers that exceed the same-year P90 max-row compute cutoff. Archetype: quantitative grid. Backend: Python/matplotlib only. Export contract: PNG figures with source-data tables.

## Methods

Input data: GPU-only paper-level compute and ACL topic metadata. The compute measure is `paper_max_row_compute_capability_gfimp_lb1 / 1e12`, interpreted as the paper-level maximum GPU-row normalized compute capacity in TFLOP/s, rather than total paper-level summed compute. Extreme values are removed using a global P99 cutoff: papers above 19,968 TFLOP/s are excluded (62 of 6900 papers removed). The upper-tail flag is computed within each publication year using that year's inclusive P90 cutoff across all retained valid GPU-only papers, then summarized by NLP topic. Because common GPU configurations can tie at the cutoff, the overall flagged share is 19.5% rather than exactly 10%.

Yearly P90 cutoffs: 2020: cutoff 896.0 TFLOP/s, 2021: cutoff 896.0 TFLOP/s, 2022: cutoff 1,792 TFLOP/s, 2023: cutoff 2,496 TFLOP/s, 2024: cutoff 2,496 TFLOP/s, 2025: cutoff 2,496 TFLOP/s.

Statistical checks are descriptive-supporting, not causal. Across 6,838 papers and 29 topics, a Kruskal-Wallis test on log10 compute gives H=587.9, p=4.42e-106, epsilon-squared=0.082. After centering log10 compute by year, H=340.8, p=1.79e-55, epsilon-squared=0.046. Topic and top-decile membership are associated by chi-square p=1.47e-42, Cramer's V=0.201.

## Key results

Highest median compute topics:

- LLM agents: median 1,480 TFLOP/s, IQR 377.8-2,496, P90 2,496 (n=144).
- NLP and Code Models: median 896.0 TFLOP/s, IQR 312.0-2,496, P90 4,890 (n=155).
- Human-Centered NLP and Human-AI Interaction: median 624.0 TFLOP/s, IQR 249.6-2,144, P90 2,496 (n=67).
- Information Retrieval and Text Mining: median 476.4 TFLOP/s, IQR 224.9-2,496, P90 2,496 (n=210).
- Resources and Evaluation: median 462.2 TFLOP/s, IQR 311.9-2,496, P90 2,496 (n=88).
- Language Modeling: median 455.2 TFLOP/s, IQR 293.0-2,496, P90 2,496 (n=440).

Largest upper tails by P90:

- Speech Recognition, Text-to-Speech and Spoken Language Understanding: P90 4,992 TFLOP/s, median 448.0 (n=116).
- NLP and Code Models: P90 4,890 TFLOP/s, median 896.0 (n=155).
- LLM agents: P90 2,496 TFLOP/s, median 1,480 (n=144).
- Multilingualism and Cross-Lingual NLP: P90 2,496 TFLOP/s, median 312.0 (n=315).
- Phonology, Morphology, and Word Segmentation: P90 2,496 TFLOP/s, median 145.8 (n=48).
- Dialogue and Interactive Systems: P90 2,496 TFLOP/s, median 284.0 (n=395).

Topics most enriched in yearly P90-or-above compute papers:

- LLM agents: 47.2% in yearly P90-or-above compute (68/144 papers).
- NLP and Code Models: 36.1% in yearly P90-or-above compute (56/155 papers).
- Information Retrieval and Text Mining: 29.0% in yearly P90-or-above compute (61/210 papers).
- Resources and Evaluation: 27.3% in yearly P90-or-above compute (24/88 papers).
- Language Modeling: 27.3% in yearly P90-or-above compute (120/440 papers).
- Human-Centered NLP and Human-AI Interaction: 26.9% in yearly P90-or-above compute (18/67 papers).

Topics least represented in yearly P90-or-above compute papers:

- Syntax: Tagging, Chunking and Parsing: 4.9% in yearly P90-or-above compute (3/61 papers).
- Sentiment Analysis, Stylistic Analysis, and Argument Mining: 6.2% in yearly P90-or-above compute (9/145 papers).
- Discourse and Pragmatics: 6.7% in yearly P90-or-above compute (5/75 papers).
- Linguistic Theories, Cognitive Modeling, and Psycholinguistics: 6.8% in yearly P90-or-above compute (4/59 papers).

## Interpretation

The answer to the RQ is yes: NLP topic is associated with maximum GPU-row compute scale. The difference is clearest for topics tied to large models and multimodal systems, where both median max-row compute and yearly P90-or-above representation are higher. Smaller-resource and more analysis-oriented topics usually sit lower in the compute distribution, though raw medians should be read alongside the within-year upper-tail metric because topic composition changes across 2020-2025.

## Outputs

- `4.3/NLP topic/fig/nlp_topic_compute_emnlp_p99_trimmed.png`
- `4.3/NLP topic/data/nlp_topic_compute_summary_p99_trimmed.csv`
- `4.3/NLP topic/data/nlp_topic_compute_by_year_p99_trimmed.csv`
- `4.3/NLP topic/data/yearly_top10_compute_thresholds_p99_trimmed.csv`
- `4.3/NLP topic/data/paper_topic_compute_with_top10_flag_p99_trimmed.csv`
- `source_data/nlp_topic_compute_summary_p99_trimmed.xlsx`

## Review risks

Topic labels come from the existing topic-classification table and inherit its uncertainty. The compute metric captures the maximum reported or imputed GPU-row capacity per paper, not total training FLOPs or wall-clock compute. The P90-or-above share reduces year-composition confounding but does not control for venue, paper type, or model-family covariates.
