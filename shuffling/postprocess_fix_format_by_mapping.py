#!/usr/bin/env python3
"""
根据 real_to_fake_mapping 将伪词按顺序填充至 NER 模板，确保格式一致。

使用方法：
    python postprocess_fix_format_by_mapping.py \
        --input generation_results.jsonl \
        --output generation_results_fixed.jsonl

假设输入 JSONL 的每条记录包含：
    - ner_formatted_text: 带 <redacted> word (TYPE) </redacted> 的原始模板
    - obfuscated_samples: [(text, prob), ...]
    - real_to_fake_mapping: {real_word: [fake1, fake2, ...]}

生成逻辑：
    1. 解析 ner_formatted_text，依次获得 real_words 顺序列表。
    2. 按旧 obfuscated_samples 的长度 N 构造 N 条新样本：
       第 i 条样本在位置 j 使用 mapping[real_word_j][i % len(fake_list)]
    3. 新样本概率沿用旧样本的 prob。
    4. 将新样本列表写回 obfuscated_samples 字段，并写入输出文件。
"""
import argparse
import json
import logging
import os
import re
from typing import List, Tuple, Dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PLACEHOLDER_PAT = re.compile(r"<redacted>.*?</redacted>", re.DOTALL)
INSIDE_PAT = re.compile(r"<redacted>\s*([^()<>]+?)\s*\([^)]+\)\s*</redacted>", re.DOTALL)


def extract_real_words(ner_text: str) -> List[str]:
    """按出现顺序提取真实敏感词文本"""
    return [m.group(1).strip() for m in INSIDE_PAT.finditer(ner_text)]


def build_sample(
    ner_text: str,
    real_words: List[str],
    mapping: Dict[str, List[str]],
    idx: int,
    mode: str = "progressive",
) -> str:
    """用第 idx 轮的伪词序列生成新文本"""
    result = ner_text
    local_counters: Dict[str, int] = {rw: 0 for rw in real_words}

    for real_word in real_words:
        fake_list = mapping.get(real_word, [])
        if not fake_list:
            replacement = real_word
        else:
            if mode == "progressive":
                # 重复出现用同一个伪词
                replacement = fake_list[idx % len(fake_list)]
            else:  # gqs
                k = local_counters[real_word]
                replacement = fake_list[(idx + k) % len(fake_list)]

        local_counters[real_word] += 1
        # 使用 lambda 避免反斜杠等字符被 re 解释
        result, _ = PLACEHOLDER_PAT.subn(lambda _m, rep=replacement: rep, result, count=1)
    # 将替换产生的多余空格压缩为单空格，保留换行
    result = re.sub(r"[ ]{2,}", " ", result)
    return result


def process_record(rec: dict, mode: str = "progressive") -> dict:
    ner_text = rec.get("ner_formatted_text")
    samples = rec.get("obfuscated_samples", [])
    mapping = rec.get("real_to_fake_mapping", {})
    if not ner_text or not samples or not mapping:
        return rec

    real_words_seq = extract_real_words(ner_text)
    if not real_words_seq:
        return rec

    new_samples: List[Tuple[str, float]] = []
    for i, sample in enumerate(samples):
        if isinstance(sample, (list, tuple)) and len(sample) >= 2:
            prob = sample[1]
        else:
            prob = 1.0
        new_text = build_sample(ner_text, real_words_seq, mapping, i, mode=mode)
        new_samples.append((new_text, prob))

    rec["obfuscated_samples"] = new_samples
    return rec


def main():
    parser = argparse.ArgumentParser(description="根据 real_to_fake_mapping 修正伪样本格式")
    parser.add_argument("--input", default='shuffling/results/synthetic_dataset_gpt-mini_0519_test/progressive_generation_results.jsonl', help="原生成结果 JSONL")
    parser.add_argument("--output", default='shuffling/results/synthetic_dataset_gpt-mini_0519_test/progressive_generation_fixed_results.jsonl', help="输出修正后 JSONL")
    parser.add_argument("--mode", choices=["progressive", "gqs"], default="progressive", help="生成映射方式")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logging.error("输入文件不存在: %s", args.input)
        return

    total, written = 0, 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line)
                continue
            rec = process_record(rec, args.mode)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if written == 1:
                print(rec["obfuscated_samples"][0])

    logging.info("完成：读取 %d 条，写入 %d 条", total, written)


if __name__ == "__main__":
    main() 