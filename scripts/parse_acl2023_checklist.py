import argparse
import json
import os
import re
from typing import List, Optional

from pypdf import PdfReader


CHECKBOX_YES = "\x13"
CHECKBOX_NO = "\x17"


def normalize_status(marker: Optional[str]) -> str:
    if marker == CHECKBOX_YES:
        return "✓"
    if marker == CHECKBOX_NO:
        return "✗"
    return ""


def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if line.isdigit():
        return True
    if line.startswith("The Responsible NLP Checklist used at ACL 2023"):
        return True
    if line == "assistance.":
        return True
    return False


def is_new_question_start(line: str) -> bool:
    if re.match(r"^[A-Z]\s+For every submission:\s*$", line):
        return True
    if re.match(r"^[A-Z]\s+□", line):
        return True
    if re.match(r"^□(?:[\x13\x17])?\s*[A-Z]\d+\.", line):
        return True
    return False


def parse_question_start(line: str):
    m = re.match(r"^([A-Z])\s+For every submission:\s*$", line)
    if m:
        code = m.group(1)
        return code, "For every submission.", "", True

    m = re.match(r"^([A-Z])\s+□([\x13\x17])?\s*(.+)$", line)
    if m:
        code = m.group(1)
        status = normalize_status(m.group(2))
        return code, m.group(3).strip(), status, False

    m = re.match(r"^□([\x13\x17])?\s*([A-Z]\d+)\.\s*(.+)$", line)
    if m:
        status = normalize_status(m.group(1))
        code = m.group(2)
        return code, m.group(3).strip(), status, False

    return None


def read_lines(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    lines: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not is_noise_line(line):
                lines.append(line)
    return lines


def parse_acl2023_checklist(pdf_path: str):
    lines = read_lines(pdf_path)
    result = []
    i = 0

    while i < len(lines):
        parsed = parse_question_start(lines[i])
        if not parsed:
            i += 1
            continue

        code, first_text, status, is_a_header = parsed
        inline_answer = None
        if not is_a_header and "?" in first_text:
            q_head, q_tail = first_text.split("?", 1)
            question_parts = [f"{q_head.strip()}?"]
            if q_tail.strip():
                inline_answer = q_tail.strip()
        else:
            question_parts = [first_text]
        i += 1

        if not is_a_header:
            if not first_text.endswith("?"):
                while i < len(lines) and not is_new_question_start(lines[i]):
                    if lines[i].endswith("?"):
                        question_parts.append(lines[i])
                        i += 1
                        break
                    if re.match(r"^(Left blank\.|No response\.|Yes\.|See|Sec\.|SEc\.|APP|Table )", lines[i]):
                        break
                    question_parts.append(lines[i])
                    i += 1

            answer_parts = [inline_answer] if inline_answer else []
            while i < len(lines) and not is_new_question_start(lines[i]):
                answer_parts.append(lines[i])
                i += 1
            answer = " ".join(answer_parts).strip() if answer_parts else None
        else:
            answer = None

        question = " ".join(question_parts).strip()
        question = re.sub(r"\s+", " ", question).strip()
        if answer:
            answer = re.sub(
                r"\s*The Responsible NLP Checklist used at ACL 2023.*$",
                "",
                answer,
            ).strip()
            answer = re.sub(r"\s+", " ", answer).strip()
            if not answer:
                answer = None
        result.append(
            {
                "quesiton_code": code,
                "question": question,
                "status": status,
                "answer": answer,
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("output_json")
    args = parser.parse_args()

    data = parse_acl2023_checklist(args.pdf_path)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
