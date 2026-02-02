#!/usr/bin/env python3
"""
Format redactions for a Hugging Face dataset which contains `user_query` (original)
and `redacted_query` (masked with [REDACTED]).

For each sample:
- Identify substrings removed in `user_query` relative to `redacted_query` after
  stripping the token [REDACTED].
- Run spaCy NER on the original text.
- For each removed span:
  - If it matches a known entity and not excluded, emit:
      <redacted> {entity_text} ({entity_type}) </redacted>
  - If it contains English month/week day names (incl. common abbreviations), force keep
    regardless of exclusions. If no entity label is detected, fallback to DATE.
  - Otherwise, keep the original substring without marking.

Outputs JSONL with keys: `text` and `masked_text`.

Usage example:
  python hf_redaction_formatter.py \
    --repo-id your_org/your_dataset \
    --split train \
    --output /path/out.jsonl \
    --spacy-model en_core_web_sm \
    --exclude PERSON TIME

Notes:
- Install deps: pip install datasets spacy
- Download a spaCy model: python -m spacy download en_core_web_sm
"""
from __future__ import annotations

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "/data1/huggingface"
import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple


PII_TOKEN = "[REDACTED]"


@dataclass
class EntitySpan:
    start: int
    end: int
    label: str


class SpacyNER:
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


# English month and weekday (incl. common abbreviations, allowing trailing dot like "Mon.")
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


# Chinese character detection (CJK Unified Ideographs)
CH_RE = re.compile(r"[\u4E00-\u9FFF]")


def chinese_ratio(text: str) -> float:
    if not text:
        return 0.0
    non_ws = [ch for ch in text if not ch.isspace()]
    if not non_ws:
        return 0.0
    cn = sum(1 for ch in non_ws if CH_RE.match(ch))
    return cn / len(non_ws)


def remove_chinese(text: str) -> str:
    if not text:
        return text
    return CH_RE.sub("", text)


def compute_deleted_spans(original: str, masked: str) -> List[Tuple[int, int]]:
    """Return spans in `original` that were removed to form `masked` (with [REDACTED] stripped)."""
    anchor = masked.replace(PII_TOKEN, "")
    matcher = difflib.SequenceMatcher(a=original, b=anchor, autojunk=False)
    spans: List[Tuple[int, int]] = []
    for tag, a0, a1, _b0, _b1 in matcher.get_opcodes():
        if tag in ("delete", "replace") and a1 > a0:
            spans.append((a0, a1))
    return spans


def find_best_entity_label_for_span(
    span_start: int,
    span_end: int,
    entities: Sequence[EntitySpan],
    min_overlap_chars: int = 1,
) -> Optional[str]:
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
    ner: SpacyNER,
    excluded_types: Iterable[str],
) -> str:
    deleted_spans = compute_deleted_spans(original_text, masked_text)
    if not deleted_spans:
        return original_text

    entities = ner.get_entities(original_text)
    excluded = {t.strip().upper() for t in excluded_types if t and t.strip()}

    out_parts: List[str] = []
    cursor = 0
    for start, end in deleted_spans:
        if start > cursor:
            out_parts.append(original_text[cursor:start])
        chunk = original_text[start:end]
        label = find_best_entity_label_for_span(start, end, entities)

        # Force keep if contains month/weekday (ignore exclusions); fallback label DATE.
        if contains_month_or_weekday(chunk):
            used_label = label if label is not None else "DATE"
            out_parts.append(f"<redacted> {chunk} ({used_label}) </redacted>")
        elif label is None or label.upper() in excluded:
            out_parts.append(chunk)
        else:
            out_parts.append(f"<redacted> {chunk} ({label}) </redacted>")
        cursor = end

    if cursor < len(original_text):
        out_parts.append(original_text[cursor:])
    return "".join(out_parts)


def iter_dataset_samples(
    repo_id: str,
    split: str,
    config_name: Optional[str],
    streaming: bool,
) -> Iterator[dict]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("请先安装 datasets: pip install datasets") from exc

    ds_kwargs = {}
    if config_name:
        ds_kwargs["name"] = config_name

    if streaming:
        dataset = load_dataset(repo_id, split=split, streaming=True, **ds_kwargs)
        for item in dataset:  # type: ignore[assignment]
            yield item
    else:
        dataset_dict = load_dataset(repo_id, **ds_kwargs)
        if split not in dataset_dict:
            raise ValueError(f"数据集不包含split: {split}. 可用: {list(dataset_dict.keys())}")
        for item in dataset_dict[split]:
            yield item


def process_hf(
    repo_id: str,
    split: str,
    output_path: str,
    spacy_model: str,
    excluded_types: List[str],
    text_field: str,
    masked_field: str,
    config_name: Optional[str],
    streaming: bool,
    limit: Optional[int],
    encoding: str,
    chinese_skip_threshold: float,
) -> None:
    ner = SpacyNER(spacy_model)
    total = 0
    written = 0

    with open(output_path, "w", encoding=encoding) as fout:
        for sample in iter_dataset_samples(repo_id, split, config_name, streaming):
            if limit is not None and total >= limit:
                break
            total += 1
            text = sample.get(text_field)
            masked = sample.get(masked_field)
            if not isinstance(text, str) or not isinstance(masked, str):
                continue

            # Chinese filtering: skip if majority Chinese; else remove Chinese chars
            ratio = chinese_ratio(text)
            if ratio >= chinese_skip_threshold:
                continue
            if ratio > 0.0:
                text = remove_chinese(text)
                masked = remove_chinese(masked)

            new_masked = build_redacted_text(
                original_text=text,
                masked_text=masked,
                ner=ner,
                excluded_types=excluded_types,
            )
            out_obj = {"text": text, "masked_text": new_masked}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"处理完成：读取 {total} 条，写出 {written} 条 -> {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 Hugging Face 数据集格式化脱敏文本为 <redacted> 标注 JSONL")
    p.add_argument("--repo-id", default="Columbia-NLP/PUPA", help="数据集仓库ID，如: username/dataset")
    p.add_argument("--split", default="train", help="数据集split，默认 train")
    p.add_argument("--output", default="./datasets/PUPA_NER.jsonl", help="输出JSONL路径")
    p.add_argument("--spacy-model", default="en_core_web_trf", help="spaCy模型名")
    p.add_argument("--exclude", nargs="*", default=['DATE', 'TIME', 'GPE', 'QUANTITY', 'NORP', 'PERCENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'FAC', 'PRODUCT', 'EVENT'], help="排除的实体类型，如 PERSON TIME")
    p.add_argument("--text-field", default="user_query", help="原文字段名，默认 user_query")
    p.add_argument(
        "--masked-field",
        default="redacted_query",
        help="掩码文本字段名，默认 redacted_query (掩码为 [REDACTED])",
    )
    p.add_argument("--config-name", default="pupa_new", help="可选的子配置名称(name/config)")
    p.add_argument("--streaming", action="store_true", help="以流式读取数据集，节省内存")
    p.add_argument("--limit", type=int, default=None, help="仅处理前N条用于测试")
    p.add_argument("--encoding", default="utf-8", help="输出文件编码，默认 utf-8")
    p.add_argument(
        "--chinese-skip-threshold",
        type=float,
        default=0.5,
        help="若原文中文字符比例≥该阈值则跳过；若0<比例<阈值则移除中文 (默认0.5)",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    process_hf(
        repo_id=args.repo_id,
        split=args.split,
        output_path=args.output,
        spacy_model=args.spacy_model,
        excluded_types=args.exclude,
        text_field=args.text_field,
        masked_field=args.masked_field,
        config_name=args.config_name,
        streaming=args.streaming,
        limit=args.limit,
        encoding=args.encoding,
        chinese_skip_threshold=args.chinese_skip_threshold,
    )


if __name__ == "__main__":
    main()


