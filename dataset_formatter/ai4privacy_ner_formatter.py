#!/usr/bin/env python3
"""
Format a Hugging Face dataset which provides pre-annotated entities under `privacy_mask`
for each `source_text`, with non-standard labels that require mapping.

Features:
- Filter by language == 'en'
- Label mapping from provided labels to normalized categories
- Month/weekday force-keep rule (ignores exclusions; fallback label DATE if missing)
- Exclusion list to bypass marking for certain mapped categories
- Output JSONL with keys: `text` and `masked_text`

Input fields (default):
- source_text: original text
- privacy_mask: list of {label, start, end, value, label_index}
- language: ISO code; only 'en' samples are processed

Usage example:
  python hf_label_mapped_formatter.py \
    --repo-id your_org/your_dataset \
    --split train \
    --output /path/out.jsonl \
    --exclude PERSON TIME
"""

from __future__ import annotations

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "/data1/huggingface"
import argparse
import json
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


@dataclass
class Entity:
    start: int
    end: int
    label: str


# Mapping from dataset-specific labels to normalized categories
LABEL_MAP: Dict[str, str] = {
    "USERNAME": "USERNAME",
    "DATEOFBIRTH": "DATE",
    "STREET": "LOCATION",
    "ZIPCODE": "ID",
    "TELEPHONENUM": "PHONE",
    "CREDITCARDNUMBER": "ID",
    "EMAIL": "EMAIL",
    "CITY": "LOCATION",
    "BUILDINGNUM": "IDENTIFIER",
    "GIVENNAME": "PERSON",
    "SURNAME": "PERSON",
    "IDCARDNUM": "ID",
    "PASSWORD": "PASSWORD",
    "DRIVERLICENSENUM": "ID",
    "SOCIALNUM": "ID",
    "ACCOUNTNUM": "ID",
    "TAXNUM": "ID",
}


# English month and weekday regex (supports common abbreviations and optional trailing dot)
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




def normalize_entities(privacy_mask: object) -> List[Entity]:
    if not isinstance(privacy_mask, list):
        return []
    ents: List[Entity] = []
    for item in privacy_mask:
        if not isinstance(item, dict):
            continue
        label_raw = str(item.get("label", "")).strip().upper()
        start = int(item.get("start", 0))
        end = int(item.get("end", 0))
        if end <= start:
            continue
        mapped = LABEL_MAP.get(label_raw, label_raw)
        ents.append(Entity(start=start, end=end, label=mapped))
    ents.sort(key=lambda e: (e.start, e.end))
    return ents


def build_masked_text_from_entities(
    text: str,
    entities: Sequence[Entity],
    excluded_types: Iterable[str],
) -> str:
    excluded = {t.strip().upper() for t in excluded_types if t and t.strip()}
    out_parts: List[str] = []
    cursor = 0

    for ent in entities:
        if ent.start >= len(text):
            continue
        s = max(cursor, max(0, ent.start))
        e = min(len(text), ent.end)
        if e <= s:
            continue
        # Append non-entity chunk
        if s > cursor:
            out_parts.append(text[cursor:s])

        chunk = text[s:e]
        # Force keep month/weekday regardless of exclusions; fallback label DATE
        if contains_month_or_weekday(chunk):
            used_label = ent.label if ent.label else "DATE"
            out_parts.append(f"<redacted> {chunk} ({used_label}) </redacted>")
        elif ent.label.upper() in excluded:
            out_parts.append(chunk)
        else:
            out_parts.append(f"<redacted> {chunk} ({ent.label}) </redacted>")

        cursor = e

    if cursor < len(text):
        out_parts.append(text[cursor:])
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
    text_field: str,
    entities_field: str,
    language_field: str,
    excluded_types: List[str],
    config_name: Optional[str],
    streaming: bool,
    limit: Optional[int],
    encoding: str,
) -> None:
    total = 0
    written = 0

    with open(output_path, "w", encoding=encoding) as fout:
        for sample in iter_dataset_samples(repo_id, split, config_name, streaming):
            if limit is not None and total >= limit:
                break
            total += 1

            lang = sample.get(language_field)
            if lang != "en":
                continue

            text = sample.get(text_field)
            privacy_mask = sample.get(entities_field)
            if not isinstance(text, str):
                continue

            ents = normalize_entities(privacy_mask)

            # Remove overlaps by skipping entities that start before current cursor
            ents.sort(key=lambda e: (e.start, e.end))

            masked_text = build_masked_text_from_entities(text, ents, excluded_types)
            out_obj = {"text": text, "masked_text": masked_text}
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    print(f"处理完成：读取 {total} 条，写出 {written} 条 -> {output_path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 Hugging Face 数据集读取 source_text 与 privacy_mask，进行标签映射并输出 <redacted> JSONL")
    p.add_argument("--repo-id", default="ai4privacy/pii-masking-400k", help="数据集仓库ID，如: username/dataset")
    p.add_argument("--split", default="validation", help="split")
    p.add_argument("--output", default="./datasets/ai4privacy_NER.jsonl", help="输出JSONL路径")
    p.add_argument("--text-field", default="source_text", help="原文字段名，默认 source_text")
    p.add_argument("--entities-field", default="privacy_mask", help="实体列表字段名，默认 privacy_mask")
    p.add_argument("--language-field", default="language", help="语言字段名，默认 language（仅处理 'en'）")
    p.add_argument("--exclude", nargs="*", default=['DATE', 'TIME', 'GPE', 'QUANTITY', 'NORP', 'PERCENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'FAC', 'PRODUCT', 'EVENT'], help="排除的实体类型（映射后名称），如 PERSON TIME")
    p.add_argument("--config-name", default=None, help="可选的子配置名称(name/config)")
    p.add_argument("--streaming", action="store_true", help="以流式读取数据集，节省内存")
    p.add_argument("--limit", type=int, default=None, help="仅处理前N条用于测试")
    p.add_argument("--encoding", default="utf-8", help="输出文件编码，默认 utf-8")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    process_hf(
        repo_id=args.repo_id,
        split=args.split,
        output_path=args.output,
        text_field=args.text_field,
        entities_field=args.entities_field,
        language_field=args.language_field,
        excluded_types=args.exclude,
        config_name=args.config_name,
        streaming=args.streaming,
        limit=args.limit,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()


