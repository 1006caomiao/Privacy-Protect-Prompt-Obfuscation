#!/usr/bin/env python3
"""
Redaction formatter for JSONL files with original `text` and masked `masked_text` containing <PII> placeholders.

For each line (JSON object with keys `text` and `masked_text`):
- Compute which substrings were redacted by aligning `text` with `masked_text` (after removing '<PII>').
- Use a NER engine (spaCy by default, optional Presidio) to detect entity types in the original `text`.
- For each redacted span, if it maps to a recognized entity type and it's NOT excluded, replace it with:
    <redacted> {entity_text} ({entity_type}) </redacted>
  Otherwise, leave the original substring as-is (skip marking).

Output is written as JSONL with the same keys: `text` and `masked_text`.

Usage examples:
  python redaction_formatter.py \
      --input /path/in.jsonl \
      --output /path/out.jsonl \
      --exclude PERSON TIME

  python redaction_formatter.py \
      --input /path/in.jsonl \
      --output /path/out.jsonl \
      --engine presidio --language en

Notes:
- spaCy model must be installed, e.g.:
    python -m spacy download en_core_web_sm
    python -m spacy download zh_core_web_sm
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from dataclasses import dataclass
import re
from typing import Iterable, List, Optional, Sequence, Tuple


PII_TOKEN = "<PII>"


@dataclass
class EntitySpan:
    start: int
    end: int
    label: str


class BaseNER:
    def get_entities(self, text: str) -> List[EntitySpan]:
        raise NotImplementedError


class SpacyNER(BaseNER):
    def __init__(self, model: str) -> None:
        try:
            import spacy  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "请先安装 spaCy: pip install spacy，并下载所需模型，例如 en_core_web_sm"
            ) from exc
        try:
            self.nlp = spacy.load(model, disable=["tagger", "parser", "lemmatizer"])
        except Exception as exc:
            raise RuntimeError(
                f"加载 spaCy 模型失败: {model}。请先运行: python -m spacy download {model}"
            ) from exc

    def get_entities(self, text: str) -> List[EntitySpan]:
        doc = self.nlp(text)
        return [EntitySpan(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]


class PresidioNER(BaseNER):
    def __init__(self, language: str = "en") -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "请先安装 Presidio: pip install presidio-analyzer"
            ) from exc
        # Lazy init; keep analyzer simple
        from presidio_analyzer import AnalyzerEngine  # type: ignore
        self.analyzer = AnalyzerEngine()
        self.language = language

    def get_entities(self, text: str) -> List[EntitySpan]:
        results = self.analyzer.analyze(text=text, language=self.language)
        return [EntitySpan(r.start, r.end, r.entity_type) for r in results]


# 英文月份与星期（含常见缩写，允许缩写后带句点，如 "Mon.", "Sept."）
MONTH_WEEKDAY_RE = re.compile(
    r"(?<![A-Za-z])(?:"
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"mon(?:day)?|tue(?:s(?:day)?)?|wed(?:nesday)?|thu(?:r(?:s(?:day)?)?)?|fri(?:day)?|sat(?:urday)?|sun(?:day)?"
    r")(?:\.)?(?![A-Za-z])",
    flags=re.IGNORECASE,
)


def contains_month_or_weekday(text: str) -> bool:
    if not text:
        return False
    return MONTH_WEEKDAY_RE.search(text) is not None


def compute_deleted_spans(original: str, masked: str) -> List[Tuple[int, int]]:
    """Identify spans in `original` that were removed when producing `masked`.

    We remove PII_TOKEN from `masked` to get an anchor string, then use SequenceMatcher to
    locate deletions (or replacements) in `original` relative to the anchor.
    """
    anchor = masked.replace(PII_TOKEN, "")
    # Disable autojunk to avoid odd behavior on long strings with many repeats
    matcher = difflib.SequenceMatcher(a=original, b=anchor, autojunk=False)

    spans: List[Tuple[int, int]] = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag in ("delete", "replace"):
            if a1 > a0:
                spans.append((a0, a1))
        # inserts into anchor are irrelevant for our purpose
    return spans


def find_best_entity_label_for_span(
    span_start: int,
    span_end: int,
    entities: Sequence[EntitySpan],
    min_overlap_chars: int = 1,
) -> Optional[str]:
    """Find entity label with maximum overlap with [span_start, span_end).

    Returns None if no overlap reaches `min_overlap_chars`.
    """
    best_label: Optional[str] = None
    best_overlap = 0
    for ent in entities:
        ov_start = max(span_start, ent.start)
        ov_end = min(span_end, ent.end)
        overlap = max(0, ov_end - ov_start)
        if overlap > best_overlap and overlap >= min_overlap_chars:
            best_overlap = overlap
            best_label = ent.label
    return best_label


def build_redacted_text(
    original_text: str,
    masked_text: str,
    ner_engine: BaseNER,
    excluded_types: Iterable[str],
) -> str:
    """Construct a new masked_text by replacing only the actually masked spans.

    - If a deleted span has an identifiable entity type and not excluded -> wrap with
      "<redacted> {entity_text} ({entity_type}) </redacted>".
    - Else keep the original substring unchanged (skip marking).
    """
    deleted_spans = compute_deleted_spans(original_text, masked_text)
    if not deleted_spans:
        # Nothing was masked; return original unchanged
        return original_text

    # Precompute entities once
    entities = ner_engine.get_entities(original_text)
    excluded = {t.strip().upper() for t in excluded_types if t.strip()}

    out_parts: List[str] = []
    cursor = 0

    for start, end in deleted_spans:
        # Append non-masked chunk
        if start > cursor:
            out_parts.append(original_text[cursor:start])

        chunk = original_text[start:end]
        label = find_best_entity_label_for_span(start, end, entities)

        # 强制保留: 若文本包含英文月份或星期，则不受排除列表影响；若NER未识别，标签回退为 DATE。
        if contains_month_or_weekday(chunk):
            used_label = label if label is not None else "DATE"
            out_parts.append(f"<redacted> {chunk} ({used_label}) </redacted>")
        elif label is None or label.upper() in excluded:
            # Skip marking; put back original
            out_parts.append(chunk)
        else:
            out_parts.append(f"<redacted> {chunk} ({label}) </redacted>")

        cursor = end

    # Tail
    if cursor < len(original_text):
        out_parts.append(original_text[cursor:])

    return "".join(out_parts)


def process_jsonl(
    input_path: str,
    output_path: str,
    engine: str,
    spacy_model: str,
    language: str,
    excluded_types: List[str],
    encoding: str = "utf-8",
) -> None:
    if engine == "spacy":
        ner: BaseNER = SpacyNER(spacy_model)
    elif engine == "presidio":
        try:
            ner = PresidioNER(language=language)
        except RuntimeError as exc:
            print(f"[WARN] Presidio 初始化失败，回退到 spaCy: {exc}", file=sys.stderr)
            ner = SpacyNER(spacy_model)
    else:
        raise ValueError("engine 必须是 'spacy' 或 'presidio'")

    total = 0
    written = 0

    with open(input_path, "r", encoding=encoding) as fin, open(
        output_path, "w", encoding=encoding
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # Skip invalid JSON line
                continue

            text = obj.get("text")
            masked_text = obj.get("masked_text")
            if not isinstance(text, str) or not isinstance(masked_text, str):
                # Skip if required fields missing
                continue

            new_masked = build_redacted_text(
                original_text=text,
                masked_text=masked_text,
                ner_engine=ner,
                excluded_types=excluded_types,
            )

            out_obj = {"text": text, "masked_text": new_masked}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"处理完成：读取 {total} 行，写出 {written} 行 -> {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将含有 <PII> 掩码的 jsonl 转为带 <redacted> 标注的 masked_text"
    )
    parser.add_argument("--input", default="./processed_datasets/synthetic_dataset/synthetic_dataset_gpt-mini_0519_test.jsonl", help="输入 jsonl 路径（包含 text/masked_text）")
    parser.add_argument("--output", default="./datasets/synthetic_NER.jsonl", help="输出 jsonl 路径")
    parser.add_argument(
        "--engine",
        choices=["spacy", "presidio"],
        default="spacy",
        help="实体识别引擎，默认 spaCy",
    )
    parser.add_argument(
        "--spacy-model",
        default="en_core_web_trf",
        help="spaCy 模型名，默认 en_core_web_trf",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Presidio 语言代码（仅当 engine=presidio 时使用），默认 en",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=['DATE', 'TIME', 'GPE', 'QUANTITY', 'NORP', 'PERCENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'FAC', 'PRODUCT', 'EVENT'],
        help="排除的实体类型列表（空格分隔），如: --exclude PERSON TIME",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="文件编码（默认 utf-8）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    process_jsonl(
        input_path=args.input,
        output_path=args.output,
        engine=args.engine,
        spacy_model=args.spacy_model,
        language=args.language,
        excluded_types=args.exclude,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()

