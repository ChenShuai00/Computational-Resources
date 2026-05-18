import json
import os
import time
from pathlib import Path
from typing import Any

import requests


ARR_AREA_KEYWORDS = {
    "Clinical and Biomedical Applications": (
        "biomedical knowledge extraction, discovery, and text mining; biomedical question answering; "
        "clinical and biomedical language models; clinical dialogue systems; biomedical summarization"
    ),
    "Computational Social Science and Cultural Analytics": (
        "human behavior analysis; stance, frame, hate-speech, misinformation, emotion, and cultural bias analysis; "
        "sociolinguistics; NLP tools for social analysis"
    ),
    "Dialogue and Interactive Systems": (
        "spoken dialogue systems; task-oriented dialogue; dialogue evaluation; interactive storytelling; "
        "embodied agents; multimodal, grounded, multilingual, and low-resource dialogue"
    ),
    "Discourse and Pragmatics": (
        "anaphora, coreference, and bridging resolution; coherence; cohesion; discourse relations and parsing; "
        "pragmatic inference and reasoning"
    ),
    "Efficient Methods for NLP": (
        "quantization; pruning; distillation; parameter-efficient training; data-efficient training; "
        "data augmentation; LLM efficiency; resource-constrained NLP"
    ),
    "Ethics, Bias, and Fairness": (
        "data ethics; model alignment; bias and fairness evaluation or mitigation; transparency; "
        "policy and governance; ethical critiques"
    ),
    "Generation": (
        "text-to-text and data-to-text generation; evaluation; multilingual generation; few-shot generation; "
        "retrieval-augmented generation; interactive and collaborative generation"
    ),
    "Human-Centered NLP and Human-AI Interaction": (
        "human-AI interaction and cooperation; human-in-the-loop; human-centered evaluation; user-centered design; "
        "participatory and community-based NLP"
    ),
    "Information Extraction": (
        "named entity recognition; relation and event extraction; open information extraction; "
        "knowledge base construction; entity linking; document-level extraction"
    ),
    "Information Retrieval and Text Mining": (
        "passage retrieval; dense retrieval; document representation; hashing; re-ranking; pre-training; "
        "contrastive learning; retrieval for RAG"
    ),
    "Interpretability and Analysis of Models for NLP": (
        "adversarial attacks and robustness; calibration; counterfactual and contrastive explanations; "
        "data influence; explanation faithfulness; model editing; probing; topic modeling"
    ),
    "Language Modeling": (
        "pre-training; fine-tuning; prompting; chain-of-thought; hallucinations; safety; scaling; privacy; "
        "security; sparse models; red teaming; retrieval-augmented language models; watermarking"
    ),
    "LLM agents": (
        "tool use; function calling; multimodal agents; multi-agent systems; planning; communication; "
        "coordination; environment interaction; memory; LLM-based controllers; agent evaluation"
    ),
    "Linguistic Theories, Cognitive Modeling, and Psycholinguistics": (
        "linguistic theories; cognitive modeling; computational psycholinguistics"
    ),
    "Machine Learning for NLP": (
        "graph-based methods; knowledge-augmented methods; multi-task, self-supervised, and contrastive learning; "
        "structured prediction; representation learning; transfer; optimization; causality; active learning"
    ),
    "Machine Translation": (
        "MT evaluation; bias; domain adaptation; efficient MT training and inference; few-shot or zero-shot MT; "
        "interactive, multilingual, multimodal, online, non-autoregressive, speech, and code-switching MT"
    ),
    "Multilingualism and Cross-Lingual NLP": (
        "code-switching; multilingualism; language contact and variation; cross-lingual transfer; multilingual "
        "representations, pre-training, benchmarks, and evaluation; dialects; less-resourced languages"
    ),
    "Multimodality and Language Grounding to Vision, Robotics and Beyond": (
        "vision-language navigation; cross-modal pretraining, matching, generation, extraction, and translation; "
        "vision question answering; speech and vision; video processing; robotics grounding"
    ),
    "NLP and Code Models": (
        "language-to-code generation; code-to-language generation; natural language contributions involving code"
    ),
    "NLP and Symbolic Reasoning": (
        "mathematical, symbolic, neurosymbolic, and logical reasoning; symbolic AI; logical neural networks; "
        "differentiable inductive logic programming; theorem proving; knowledge graph embeddings"
    ),
    "NLP Applications": (
        "education; essay scoring; financial, business, legal, historical, security, privacy, and social-good NLP; "
        "grammatical error correction; knowledge graphs; fact checking; misinformation"
    ),
    "Phonology, Morphology, and Word Segmentation": (
        "morphological inflection, induction, segmentation, analysis, and subword representations; "
        "Chinese segmentation; lemmatization; phonology; grapheme-to-phoneme conversion"
    ),
    "Question Answering": (
        "commonsense QA; reading comprehension; logical, multimodal, knowledge-base, multihop, biomedical, "
        "multilingual, conversational, math, table, and open-domain QA; RAG for QA"
    ),
    "Resources and Evaluation": (
        "corpus creation; benchmarking; language resources; multilingual corpora; lexicons; datasets; "
        "evaluation methodologies and metrics; reproducibility; statistical testing"
    ),
    "Semantics: Lexical and Sentence-Level": (
        "polysemy; lexical relationships; textual entailment; compositionality; multi-word expressions; metaphor; "
        "lexical semantic change; paraphrasing; semantic textual similarity; text simplification"
    ),
    "Sentiment Analysis, Stylistic Analysis, and Argument Mining": (
        "sentiment applications; argument generation and mining; argument quality assessment; computational "
        "affective science; stance detection; style analysis and adaptation; rhetoric and framing"
    ),
    "Speech Recognition, Text-to-Speech and Spoken Language Understanding": (
        "automatic speech recognition; text-to-speech; speech technologies; spoken dialog; spoken language "
        "grounding, translation, and understanding; QA via spoken queries"
    ),
    "Summarization": (
        "extractive, abstractive, multimodal, multilingual, conversational, query-focused, multi-document, and "
        "long-form summarization; sentence compression; evaluation; factuality"
    ),
    "Syntax: Tagging, Chunking and Parsing": (
        "chunking; constituency, dependency, deep syntax, and semantic parsing; part-of-speech tagging; "
        "morpho-syntax; multilingual and low-resource parsing; syntax-to-semantic interface"
    ),
    "Special Theme Track": "conference-specific theme track described in each conference CFP",
}

EMNLP_TOPICS = tuple(ARR_AREA_KEYWORDS)

TOPIC_ALIASES = {
    "Argument Mining": "Sentiment Analysis, Stylistic Analysis, and Argument Mining",
    "Biomedical Summarization": "Clinical and Biomedical Applications",
    "Evaluation": "Resources and Evaluation",
    "Grammatical Error Correction": "NLP Applications",
    "Knowledge Graph Embeddings": "NLP and Symbolic Reasoning",
    "Knowledge Graphs": "NLP Applications",
    "Knowledge Graphs and Question Answering": "Question Answering",
    "Knowledge Graphs and Semantic Web": "NLP Applications",
    "Narrative and Storytelling": "Generation",
    "Privacy": "NLP Applications",
    "Privacy and Security in NLP": "NLP Applications",
    "Privacy, Security, and Ethics in NLP": "NLP Applications",
    "Privacy-Preserving Text Classification on BERT Embeddings with Homomorphic Encryption": "NLP Applications",
    "Retrieval-Augmented Generation": "Generation",
    "Uncertainty Estimation in Generative LLMs": "Interpretability and Analysis of Models for NLP",
}

DEFAULT_API_BASE = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 1.0

SYSTEM_PROMPT = """You are an ACL Rolling Review area classifier for NLP papers.
Given a paper title and optional abstract, select exactly one area from the current ARR area taxonomy.
Return JSON only with keys: topic, confidence, needs_review, reason.
- topic must be copied exactly from the allowed list
- confidence must be a number between 0 and 1
- needs_review must be a boolean
- reason must be a short explanation in English
- classify by the paper's core research problem and main contribution, not incidental methods or datasets
- if the abstract is empty, classify from the title alone and set needs_review to true when uncertain
"""


def build_topic_classification_prompt(title: str, abstract: str) -> str:
    allowed_topics = "\n".join(f"- {topic}: {ARR_AREA_KEYWORDS[topic]}" for topic in EMNLP_TOPICS)
    return (
        f"Title: {title.strip()}\n"
        f"Abstract: {abstract.strip()}\n\n"
        f"Allowed topics:\n{allowed_topics}\n"
    )


def build_topic_classification_request(title: str, abstract: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": build_topic_classification_prompt(title=title, abstract=abstract),
                    }
                ],
            },
        ],
        "temperature": 0.0,
    }


def _extract_message_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Missing choices in LLM response.")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("Missing message in LLM response.")
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    raise ValueError("Missing JSON content in LLM response.")


def parse_topic_classification_response(response_payload: dict[str, Any]) -> dict[str, Any]:
    content = _extract_message_content(response_payload)
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not valid JSON.") from exc

    topic = str(parsed.get("topic", "")).strip()
    topic = TOPIC_ALIASES.get(topic, topic)
    if topic not in EMNLP_TOPICS:
        raise ValueError(f"Unknown topic returned by model: {topic}")

    confidence_raw = parsed.get("confidence")
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric.") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0 and 1.")

    needs_review = parsed.get("needs_review")
    if not isinstance(needs_review, bool):
        raise ValueError("needs_review must be a boolean.")

    reason = str(parsed.get("reason", "")).strip()
    if not reason:
        raise ValueError("reason must be non-empty.")

    return {
        "topic": topic,
        "confidence": confidence,
        "needs_review": needs_review,
        "reason": reason,
    }


def reconstruct_abstract_from_inverted_index(abstract_inverted_index: dict[str, Any]) -> str:
    if not isinstance(abstract_inverted_index, dict) or not abstract_inverted_index:
        return ""

    positions: dict[int, str] = {}
    for token, indexes in abstract_inverted_index.items():
        if not isinstance(token, str) or not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int) and index >= 0:
                positions[index] = token

    if not positions:
        return ""

    return " ".join(token for _, token in sorted(positions.items()))


def _collect_text_from_items(items: Any, key: str) -> str:
    if not isinstance(items, list):
        return ""
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if content:
            parts.append(content)
    return " ".join(parts).strip()


def load_mineru_first_page_title_abstract(path: str | Path) -> tuple[str, str]:
    content_path = Path(path)
    payload = json.loads(content_path.read_text(encoding="utf-8"))
    blocks = payload[0] if isinstance(payload, list) and payload and isinstance(payload[0], list) else payload
    if not isinstance(blocks, list):
        raise ValueError(f"Unexpected MinerU content format in {content_path}")

    title = ""
    abstract_parts: list[str] = []
    in_abstract = False

    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", "")).strip().lower()
        content = block.get("content", {})
        if not isinstance(content, dict):
            content = {}

        if block_type == "title":
            title_text = _collect_text_from_items(content.get("title_content", []), key="title_content")
            normalized = title_text.strip().lower()
            if not title and normalized and normalized != "abstract":
                title = title_text.strip()
            if normalized == "abstract":
                in_abstract = True
                continue
            if in_abstract:
                break

        elif in_abstract and block_type == "paragraph":
            paragraph_text = _collect_text_from_items(content.get("paragraph_content", []), key="paragraph_content")
            if paragraph_text:
                abstract_parts.append(paragraph_text)

    abstract = " ".join(part for part in abstract_parts if part).strip()
    if not title:
        raise ValueError(f"Missing title in {content_path}")
    if not abstract:
        raise ValueError(f"Missing abstract in {content_path}")
    return title, abstract


def load_openalex_title_abstract(path: str | Path) -> tuple[str, str]:
    paper_path = Path(path)
    payload = json.loads(paper_path.read_text(encoding="utf-8"))
    openalex_raw = payload.get("openalex_raw")
    if not isinstance(openalex_raw, dict):
        raise ValueError(f"Missing openalex_raw in {paper_path}")

    title = str(openalex_raw.get("title") or openalex_raw.get("display_name") or "").strip()
    abstract = reconstruct_abstract_from_inverted_index(openalex_raw.get("abstract_inverted_index", {}))

    if not title:
        raise ValueError(f"Missing title in {paper_path}")
    if not abstract:
        raise ValueError(f"Missing abstract in {paper_path}")

    return title, abstract


def load_openalex_title(path: str | Path) -> str:
    paper_path = Path(path)
    payload = json.loads(paper_path.read_text(encoding="utf-8"))
    openalex_raw = payload.get("openalex_raw")
    if not isinstance(openalex_raw, dict):
        return ""
    return str(openalex_raw.get("title") or openalex_raw.get("display_name") or "").strip()


def load_acl_title_abstract(path: str | Path) -> tuple[str, str]:
    paper_path = Path(path)
    payload = json.loads(paper_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected ACL metadata format in {paper_path}")

    title = str(payload.get("title") or "").strip()
    abstract = str(payload.get("abstract") or "").strip()

    if not title:
        raise ValueError(f"Missing title in {paper_path}")

    return title, abstract


def load_title_abstract_with_mineru_fallback(
    openalex_path: str | Path,
    mineru_root: str | Path | None = None,
) -> tuple[str, str]:
    paper_path = Path(openalex_path)
    try:
        return load_openalex_title_abstract(paper_path)
    except ValueError as openalex_error:
        try:
            return load_acl_title_abstract(paper_path)
        except ValueError:
            pass

        if mineru_root is None:
            raise openalex_error

        mineru_path = (
            Path(mineru_root)
            / "emnlp2025_first_page"
            / "extracted"
            / f"{paper_path.stem}_first_page"
            / "content_list_v2.json"
        )
        if not mineru_path.exists():
            raise openalex_error

        mineru_title, mineru_abstract = load_mineru_first_page_title_abstract(mineru_path)
        payload = json.loads(paper_path.read_text(encoding="utf-8"))
        openalex_raw = payload.get("openalex_raw")
        if not isinstance(openalex_raw, dict):
            return mineru_title, mineru_abstract
        openalex_title = str(openalex_raw.get("title") or openalex_raw.get("display_name") or "").strip()
        return (openalex_title or mineru_title), mineru_abstract


def resolve_api_key(config: dict[str, Any], cli_api_key: str = "") -> str:
    if cli_api_key:
        return cli_api_key

    llm_cfg = config.get("llm", {}) if isinstance(config.get("llm", {}), dict) else {}
    env_name = str(llm_cfg.get("api_key_env", "")).strip()
    if env_name:
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            return env_value

    key_from_cfg = str(llm_cfg.get("deepseek_api_key", "")).strip()
    if key_from_cfg.startswith("${") and key_from_cfg.endswith("}"):
        env_value = os.getenv(key_from_cfg[2:-1].strip(), "").strip()
        if env_value:
            return env_value
    if key_from_cfg:
        return key_from_cfg

    return os.getenv("OPENAI_API_KEY", "").strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()


def classify_title_abstract(
    *,
    title: str,
    abstract: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    api_base: str = DEFAULT_API_BASE,
    timeout: float = 60.0,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay_seconds: float = DEFAULT_RETRY_DELAY_SECONDS,
) -> dict[str, Any]:
    if not title.strip():
        raise ValueError("title must be non-empty.")
    if not api_key.strip():
        raise ValueError("Missing API key.")

    payload = build_topic_classification_request(title=title, abstract=abstract, model=model)
    attempts = max(1, int(max_retries))
    last_error: requests.exceptions.RequestException | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = requests.post(
                api_base,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            return parse_topic_classification_response(response.json())
        except requests.exceptions.RequestException as exc:
            last_error = exc
            if attempt >= attempts:
                break
            if retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)

    assert last_error is not None
    raise last_error
