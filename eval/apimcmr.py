#!/usr/bin/env python3
"""基于大模型的伪样本真伪判别评估脚本。

核心功能：
- 从混淆样本生成结果（JSONL）中读取原始文本与伪样本。
- 针对每条记录抽取若干伪样本，与原始文本随机组合后构造提示词。
- 通过大模型 API 进行批量判别，使用多线程并发以缩短评估时间。
- 统计模型判别正确、错误与“无法判断”的次数，并输出详细结果。

使用示例：

```bash
python eval/fake_detection_evaluator.py \
    --input_file /data1/privacy_protect/shuffling/results_1029/progressive/PUPA_NER/progressive_generation_results.jsonl \
    --output_dir /data1/privacy_protect/eval/results/PUPA_NER \
    --api_key $OPENAI_API_KEY \
    --api_base https://api.openai.com/v1 \
    --model gpt-4o-mini \
    --num_fake_per_record 3 \
    --max_workers 8
```
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import argparse
import json
import logging
import random
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
from openai import OpenAI
from tqdm import tqdm
from shuffling.utils import load_jsonl
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# ============================
# 数据结构
# ============================

@dataclass(slots=True)
class CandidateOption:
    """候选文本信息。"""
    option_id: int
    text: str
    is_real: bool
    source: str
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationTask:
    """单次评估任务（原文+多条混淆样本的组合）。"""
    record_index: int
    combination_id: int  # 组合ID
    prompt: str
    real_option_id: int  # 真实选项的ID
    record_id: Optional[str]
    candidates: List[CandidateOption]
    metadata: Dict[str, Any]


@dataclass(slots=True)
class EvaluationResult:
    """大模型判别结果。"""
    record_index: int
    combination_id: int
    record_id: Optional[str]
    real_option_id: int
    predicted_option_id: Optional[int]
    outcome: str  # 'correct' | 'incorrect' | 'undecided'
    raw_response: Optional[str]
    prompt: str
    candidates: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# ============================
# Prompt 生成与解析
# ============================
DEFAULT_SYSTEM_PROMPT = (
    "You are an authenticity checker. "
    "Your job is to identify which candidate passage is genuinely authentic (real/original). "
    "There is exactly ONE authentic passage and multiple synthetic/fake passages. "
    "Only output the 1-based index number of the authentic passage. "
    "If you cannot distinguish confidently, output -1. "
    "Do not explain or add any extra text."
)


def build_prompt(options: Sequence[CandidateOption]) -> str:
    """构造用户提示词。"""
    lines = [
        "Evaluate the following candidate passages.",
        f"There are {len(options)} candidates below, with exactly ONE authentic (real) passage.",
        "The order of candidates is randomized.",
        "",
        "Candidates:",
    ]
    for opt in options:
        lines.append(f"----- Option {opt.option_id} -----\n{opt.text.strip()}\n")
    lines.append("Answer (only the index number of the authentic passage, e.g., 1, 2, 3..., or -1 if unsure):")
    # lines.append(" /no_think")
    return "\n".join(lines)


_UNDECIDED_PATTERNS = (
    "无法判断",
    "无法区分",
    "不能判断",
    "难以判断",
    "无法确定",
    "undecided",
    "cannot decide",
    "cannot determine",
    "unable to decide",
    "unable to determine",
)


def parse_model_answer(answer: str, max_option_id: int) -> Tuple[str, Optional[int]]:
    """解析模型回复，返回outcome与预测选项。"""
    if answer is None:
        return "error", None
    cleaned = answer.strip()
    lowered = cleaned.lower()

    if any(keyword in lowered for keyword in _UNDECIDED_PATTERNS):
        return "undecided", None

    # 识别 -1 表示无法判断
    if re.search(r"(?<!\d)-\s*1(?!\d)", cleaned):
        return "undecided", None

    # 尝试解析正整数索引
    digit_match = re.search(r"\b([1-9][0-9]*)\b", cleaned)
    if digit_match:
        value = int(digit_match.group(1))
        if 1 <= value <= max_option_id:
            return "answered", value

    # 尝试解析中文数字“一二三四”
    chinese_digits = {
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    for char, value in chinese_digits.items():
        if char in cleaned and 1 <= value <= max_option_id:
            return "answered", value

    return "error", None


# ============================
# 任务准备与执行
# ============================
def prepare_combinations(
    records: List[Dict[str, Any]],
    *,
    num_fake_per_record: int,
    num_combinations_per_record: int,
    seed: int,
    min_text_length: int = 1,
    max_eval_records: int = 0,
) -> Tuple[List[EvaluationTask], int, int, int]:
    """根据输入记录生成评估组合任务列表。"""

    rng = random.Random(seed)
    tasks: List[EvaluationTask] = []
    skipped_records = 0
    valid_records = 0
    processed_records = 0

    for record_index, record in enumerate(records):
        if max_eval_records > 0 and valid_records >= max_eval_records:
            break

        processed_records += 1

        original_text = record.get("original_text") or record.get("text") or record.get("raw_text")
        fake_samples = record.get("obfuscated_samples") or []

        if not original_text or len(original_text.strip()) < min_text_length:
            skipped_records += 1
            continue

        indexed_fakes = [
            (idx, sample)
            for idx, sample in enumerate(fake_samples)
            if isinstance(sample, str) and len(sample.strip()) >= min_text_length
        ]

        if not indexed_fakes:
            skipped_records += 1
            continue

        if len(indexed_fakes) < num_fake_per_record:
            skipped_records += 1
            continue

        valid_records += 1

        record_id = (
            record.get("record_id")
            or record.get("id")
            or record.get("uuid")
            or record.get("doc_id")
        )

        # 为每条记录创建多个组合
        for combo_idx in range(num_combinations_per_record):
            # 随机选择混淆样本
            selected_fakes = rng.sample(indexed_fakes, num_fake_per_record)
            # 创建候选列表：原始样本 + 混淆样本
            candidates_data = []
            # 添加原始样本
            candidates_data.append({
                "text": original_text,
                "is_real": True,
                "source": "original"
            })
            # 添加混淆样本
            for fake_idx, fake_text in selected_fakes:
                candidates_data.append({
                    "text": fake_text,
                    "is_real": False,
                    "source": "fake",
                    "extra": {"fake_index": fake_idx}
                })
            # 随机打乱顺序
            rng.shuffle(candidates_data)
            # 构建CandidateOption列表
            candidates: List[CandidateOption] = []
            real_option_id = -1
            
            for i, candidate in enumerate(candidates_data, start=1):
                extra = candidate.get("extra", {})
                option = CandidateOption(
                    option_id=i,
                    text=candidate["text"],
                    is_real=candidate["is_real"],
                    source=candidate["source"],
                    extra=extra
                )
                candidates.append(option)
                if candidate["is_real"]:
                    real_option_id = i
            
            # 构建prompt
            prompt = build_prompt(candidates)
            # 元数据
            metadata = {
                "record_index": record_index,
                "combination_id": combo_idx,
                "fake_indices": [f[0] for f in selected_fakes],
                "num_options": len(candidates),
            }
            
            tasks.append(
                EvaluationTask(
                    record_index=record_index,
                    combination_id=combo_idx,
                    prompt=prompt,
                    real_option_id=real_option_id,
                    record_id=record_id,
                    candidates=candidates,
                    metadata=metadata,
                )
            )
    
    return tasks, skipped_records, valid_records, processed_records


def call_chat_completion(
    client: OpenAI,
    model: str,
    messages: Sequence[Dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=temperature,
            max_tokens=max_tokens,
            # extra_body={"enable_thinking": False}
        )
        return response.choices[0].message.content if response.choices else ""
    except Exception as e:
        print(f"调用模型时出错: {e}")
        return None


def run_evaluations(
    tasks: Sequence[EvaluationTask],
    client: OpenAI,
    *,
    system_prompt: Optional[str],
    max_workers: int,
    model: str,
    temperature: float,
    max_tokens: int,
) -> List[EvaluationResult]:
    """并发执行评估任务。"""
    results: List[EvaluationResult] = []
    total_tasks = len(tasks)

    if total_tasks == 0:
        return results

    def _worker(task: EvaluationTask) -> EvaluationResult:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task.prompt})

        try:
            response_text = call_chat_completion(
                client,
                model,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            status, predicted = parse_model_answer(response_text, max_option_id=len(task.candidates))
            if status == "answered":
                if predicted == task.real_option_id:
                    outcome = "correct"
                else:
                    outcome = "incorrect"
            elif status == "undecided":
                outcome = "undecided"
            else:
                outcome = "error"

        except Exception as exc:  # pylint: disable=broad-except
            logging.exception("任务执行失败（record=%s task=%s）: %s", task.record_index, task.task_index, exc)
            response_text = None
            predicted = None
            outcome = "error"

        return EvaluationResult(
            record_index=task.record_index,
            combination_id=task.combination_id,
            record_id=task.record_id,
            real_option_id=task.real_option_id,
            predicted_option_id=predicted,
            outcome=outcome,
            raw_response=response_text,
            prompt=task.prompt,
            candidates=[
                {
                    "option_id": c.option_id,
                    "is_real": c.is_real,
                    "source": c.source,
                    "text": c.text,
                    "extra": c.extra,
                }
                for c in task.candidates
            ],
            metadata=task.metadata,
        )

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(_worker, task): task for task in tasks}
        progress = tqdm(total=total_tasks, desc="Evaluating", unit="combination")
        try:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:  # pylint: disable=broad-except
                    logging.exception(
                        "任务执行过程中出现未捕获异常（record=%s combination=%s）: %s",
                        task.record_index,
                        task.combination_id,
                        exc,
                    )
                finally:
                    progress.update(1)
        finally:
            progress.close()

    return results


# ============================
# 统计与保存
# ============================
def summarize_results(
    results: Sequence[EvaluationResult],
    *,
    skipped_records: int,
    processed_records: int,
    valid_records: int,
    requested_records: int,
    loaded_records: int,
) -> Dict[str, Any]:
    """汇总评估统计。"""
    task_counts = {
        "correct": 0,
        "incorrect": 0,
        "undecided": 0,
        "error": 0,
    }

    # 统计每个组合的结果
    for result in results:
        task_counts[result.outcome] += 1
    total_combinations = len(results)
    # 计算最终指标
    incorrect_and_undecided = task_counts["incorrect"] + task_counts["undecided"]
    if total_combinations > 0:
        undecided_rate = incorrect_and_undecided / (total_combinations - task_counts["error"])
        correct_rate = task_counts["correct"] / (total_combinations - task_counts["error"])
    else:
        undecided_rate = None
        correct_rate = None
    
    # 按选项数量分组统计（可选）
    size_distribution = Counter()
    size_correct = Counter()
    for result in results:
        num_options = result.metadata.get("num_options", 0)
        size_distribution[num_options] += 1
        if result.outcome == "correct":
            size_correct[num_options] += 1
    size_accuracy = {}
    for size in size_distribution:
        if size_distribution[size] > 0:
            size_accuracy[size] = size_correct.get(size, 0) / size_distribution[size]

    summary = {
        "requested_records": requested_records,
        "loaded_records": loaded_records,
        "processed_records": processed_records,
        "valid_records": valid_records,
        "skipped_records": skipped_records,
        "evaluated_combinations": total_combinations,
        "task_counts": task_counts,
        "correct_rate": correct_rate,
        "undecided_rate": undecided_rate, 
        "incorrect_plus_undecided_rate": undecided_rate, 
        "size_distribution": dict(size_distribution),
        "size_accuracy": size_accuracy,
    }
    
    return summary


def save_results(
    results: Sequence[EvaluationResult],
    summary: Dict[str, Any],
    *,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    summary_path = os.path.join(output_dir, "summary.json")
    # 将摘要以“累计方式”保存：如果已存在且为对象，转换为数组并追加；若为数组，直接追加
    new_payload: Any = summary
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as rf:
                existing = json.load(rf)
            if isinstance(existing, list):
                new_payload = existing + [summary]
            elif isinstance(existing, dict):
                new_payload = [existing, summary]
            else:
                new_payload = [summary]
        except Exception:
            # 若旧文件损坏或非JSON，则从当前摘要重新开始
            new_payload = [summary]
    # 重写为合并后的内容（用于实现“追加”语义且保持合法JSON）
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(new_payload, f, ensure_ascii=False, indent=2)

    logging.info("统计摘要保存于: %s", summary_path)


# ============================
# CLI
# ============================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用大模型评估混淆样本质量（多选项）")
    parser.add_argument("--input_file", type=str, default="./shuffling/results/delta/1/gqs/synthetic_NER.jsonl/gqs_generation_results.jsonl", help="混淆样本生成结果 JSONL")
    parser.add_argument("--output_dir", type=str, default="./eval/delta_qwen0.6b/1/synthetic_NER", help="评估结果输出目录")

    parser.add_argument("--api_key", type=str, default="", help="大模型 API Key")
    parser.add_argument("--api_base", type=str, default="")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="模型名称")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="系统指令")
    parser.add_argument("--num_fake_per_record", type=int, default=3, help="每条记录使用的伪样本数量")
    parser.add_argument(
        "--num_combinations_per_record",
        type=int,
        default=1,
        help="每条记录生成的评估组合数"
    )
    parser.add_argument(
        "--max_eval_records",
        type=int,
        default=500,
        help="最多参与评估的有效记录数（<=0 表示使用全部，按顺序取前 N 条）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--max_workers", type=int, default=5, help="并发线程数")
    parser.add_argument("--temperature", type=float, default=0.0, help="模型采样温度")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=20,
        help="限制模型输出的最大token数（必须为正整数）",
    )
    parser.add_argument("--log_file", type=str, default=None, help="可选：日志文件路径")
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=1,
        help="过滤原文/伪样本时允许的最短长度（字符）",
    )

    return parser.parse_args()


def setup_logging(log_file: Optional[str] = None, eval_model: Optional[str] = None) -> None:
    log_format = "%(asctime)s - %(levelname)s - [model=%(eval_model)s] - %(message)s"
    if log_file:
        dir_name = os.path.dirname(log_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, mode="a", encoding="utf-8"),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

    class _ModelContextFilter(logging.Filter):
        def __init__(self, model_value: Optional[str]) -> None:
            super().__init__()
            self._model_value = model_value or "N/A"

        def filter(self, record: logging.LogRecord) -> bool:
            if not hasattr(record, "eval_model"):
                record.eval_model = self._model_value
            return True

    root_logger = logging.getLogger()
    root_logger.addFilter(_ModelContextFilter(eval_model))

    # 降低第三方库日志级别，避免输出 HTTP Request 等冗余信息
    for name in ("openai", "openai._base_client", "httpx", "httpcore"):
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        logger.propagate = False


def load_records(input_file: str) -> List[Dict[str, Any]]:
    logging.info("开始加载混淆样本文件：%s", input_file)
    records = load_jsonl(input_file)
    logging.info("共加载 %d 条记录", len(records))
    return records


def main() -> None:
    args = parse_args()
    setup_logging(args.log_file, eval_model=args.model)
    logging.info("使用的评估模型: %s", args.model)

    if not args.api_key:
        raise ValueError("未提供 API Key，请通过 --api_key 或环境变量 OPENAI_API_KEY 设置。")

    records = load_records(args.input_file)
    loaded_records = len(records)

    tasks, skipped_records, valid_records, processed_records = prepare_combinations(
        records,
        num_fake_per_record=max(1, args.num_fake_per_record),
        num_combinations_per_record=max(1, args.num_combinations_per_record),
        seed=args.seed,
        min_text_length=max(1, args.min_text_length),
        max_eval_records=args.max_eval_records,
    )

    requested_records = args.max_eval_records if args.max_eval_records > 0 else -1
    pair_count_est = len(tasks) // 2

    logging.info(
        "目标有效记录: %s，实际有效记录: %d，已处理记录: %d，跳过记录: %d，生成组合数: %d，加载记录总数: %d",
        requested_records if requested_records > 0 else "全部",
        valid_records,
        processed_records,
        skipped_records,
        len(tasks),
        loaded_records,
    )

    client = OpenAI(api_key=args.api_key, base_url=args.api_base)

    start_time = time.time()
    results = run_evaluations(
        tasks,
        client,
        system_prompt=args.system_prompt,
        max_workers=max(1, args.max_workers),
        model=args.model,
        temperature=args.temperature,
        max_tokens=max(1, args.max_tokens),
    )
    elapsed = time.time() - start_time

    logging.info("评估完成，用时 %.2f 秒", elapsed)

    summary = summarize_results(
        results,
        skipped_records=skipped_records,
        processed_records=processed_records,
        valid_records=valid_records,
        requested_records=requested_records,
        loaded_records=loaded_records,
    )

    # 在保存前将使用的评估模型写入摘要
    summary["model"] = args.model

    logging.info("=== 统计结果 ===")
    logging.info("评估组合总数: %d", len(results))
    logging.info("正确识别数: %d", summary["task_counts"]["correct"])
    logging.info("错误识别数: %d", summary["task_counts"]["incorrect"])
    logging.info("无法判断数: %d", summary["task_counts"]["undecided"])
    logging.info("错误数: %d", summary["task_counts"]["error"])
    logging.info("正确率: %.2f%%", (summary.get("correct_rate") or 0) * 100)
    logging.info("无法判断率(incorrect+undecided): %.2f%%", (summary.get("undecided_rate") or 0) * 100)

    save_results(results, summary, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
