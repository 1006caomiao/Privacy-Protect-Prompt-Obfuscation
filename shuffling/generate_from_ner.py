#!/usr/bin/env python3
"""
混淆样本生成脚本

- 输入：仅包含原文与NER后文本（含 <redacted> ... </redacted>）的JSONL文件
- 处理：可复现的随机抽样（--num_records, --seed），基于 progressive 或 gqs 进行混淆样本生成
- 输出：每条记录仅保留以下字段：
  • original_text
  • ner_formatted_text
  • obfuscated_samples  （字符串列表）
  • real_to_fake_mapping（dict[str, list[str]]）
  • generation_time_s

日志：以追加方式写入，记录本次实验参数、输出文件路径，以及“生成单条伪样本的平均时间”。
"""

import argparse
import json
import logging
import os
import random
from typing import Any, Dict, List, Tuple
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "/data/huggingface"

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import load_jsonl
from progressive_generator import SimpleEnhancedGenerator
from semantic_sampler import (
    GenerativeModelEmbeddingProvider,
    SemanticSampler,
    DedicatedEmbeddingModelProvider,
)
from gqs import GreedyQuantizedSampler, generate_samples_from_ner
from embedding_random_generator import EmbeddingRandomFillGenerator
from embedding_simple_generator import EmbeddingSimpleGenerator

import torch_npu
from torch_npu.npu import amp # 导入AMP模块
from torch_npu.contrib import transfer_to_npu    # 使能自动迁移
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


def setup_logging(log_file: str | None = None) -> None:
    """配置日志输出到文件与控制台。"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, mode='a', encoding='utf-8'),
                logging.StreamHandler(),
            ],
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)


def set_random_seed(seed: int) -> None:
    """为随机抽样与部分生成过程设置可复现的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ner_results(input_path: str) -> List[Dict[str, Any]]:
    """加载NER阶段的JSONL结果，过滤统计行。"""
    data = load_jsonl(input_path)
    results: List[Dict[str, Any]] = []
    for rec in data:
        if isinstance(rec, dict) and rec.get('type') == 'ner_statistics':
            continue
        results.append(rec)
    return results


def sample_records_deterministic(records: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    """从 records 随机抽取 k 条，保证同一 seed 下可复现。"""
    if k <= 0 or k >= len(records):
        return records
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), k)
    return [records[i] for i in indices]


def setup_obfuscation_generator(model_name: str, embedding_model_name: str | None, device: str) -> SimpleEnhancedGenerator:
    """构建 progressive 方法的生成器。"""
    logging.info(f"加载生成模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token="")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype, token="")
    model.eval()

    if embedding_model_name:
        embedding_provider = DedicatedEmbeddingModelProvider(
            model_name=embedding_model_name,
            generation_tokenizer=tokenizer,
            device='cuda' if device == 'cuda' else 'cpu',
            precompute_knn=True,
            knn_top_k=256,
            knn_batch_size=512,
        )
    else:
        embedding_provider = GenerativeModelEmbeddingProvider(
            model=model,
            tokenizer=tokenizer,
            precompute_knn=True,
            knn_top_k=256,
            knn_batch_size=512,
        )

    sampler = SemanticSampler(
        tokenizer=tokenizer,
        model=model,
        embedding_provider=embedding_provider,
        temperature=1.0,
        max_batch_size=64,
        semantic_top_k=200,
    )

    generator = SimpleEnhancedGenerator(sampler)
    return generator


def setup_gqs_generator(model_name: str, device: str) -> GreedyQuantizedSampler:
    """构建 GQS 贪婪量化采样器 (baseline)。"""
    logging.info(f"[GQS] 加载生成模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token="")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype, token="")
    model.eval()
    return GreedyQuantizedSampler(tokenizer, model, temperature=1.0, max_batch_size=8)


def setup_embedding_random_generator(model_name: str, embedding_model_name: str | None, device: str) -> EmbeddingRandomFillGenerator:
    """构建仅依赖嵌入的随机填充基线生成器。"""
    if not embedding_model_name or not str(embedding_model_name).strip():
        raise ValueError("semantic_random 方法需要提供有效的 embedding_model_name")

    logging.info(f"[SemanticRandom] 加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token="")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding_provider = DedicatedEmbeddingModelProvider(
        model_name=embedding_model_name,
        generation_tokenizer=tokenizer,
        device='cuda' if device == 'cuda' else 'cpu',
        precompute_knn=True,
        knn_top_k=256,
        knn_batch_size=512,
    )

    return EmbeddingRandomFillGenerator(
        tokenizer=tokenizer,
        embedding_provider=embedding_provider,
        neighbor_pool_size=256,
        exclude_top_ratio=0.1,
        random_pool_multiplier=2.0,
    )


def setup_embedding_simple_generator(model_name: str, embedding_model_name: str | None, device: str) -> EmbeddingSimpleGenerator:
    """构建无缓存、无规则的简易嵌入基线生成器。"""
    if not embedding_model_name or not str(embedding_model_name).strip():
        raise ValueError("semantic_simple 方法需要提供有效的 embedding_model_name")

    logging.info(f"[SemanticSimple] 加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", token="")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    embedding_provider = DedicatedEmbeddingModelProvider(
        model_name=embedding_model_name,
        generation_tokenizer=tokenizer,
        device='cuda' if device == 'cuda' else 'cpu',
        precompute_knn=True,
        knn_top_k=256,
        knn_batch_size=512,
    )

    return EmbeddingSimpleGenerator(
        tokenizer=tokenizer,
        embedding_provider=embedding_provider,
        neighbor_pool_size=128,
        exclude_top_ratio=0.0,
    )


def has_sensitive_targets(record: Dict[str, Any]) -> bool:
    """是否存在需要替换的占位片段。仅检查 NER 文本中是否含 <redacted>。"""
    ner_text = (
        record.get('ner_formatted_text')
        or record.get('masked_text')
        or record.get('ner_text')
        or ""
    )
    return '<redacted>' in ner_text


def get_original_and_ner_text(record: Dict[str, Any]) -> Tuple[str, str]:
    """从输入记录中提取原文与NER后文本，兼容常见键名。"""
    original_text = (
        record.get('original_text')
        or record.get('text')
        or record.get('raw_text')
        or ""
    )
    ner_text = (
        record.get('ner_formatted_text')
        or record.get('ner_text')
        or record.get('masked_text')
        or ""
    )
    return original_text, ner_text


def obfuscation_generation(
    ner_results: List[Dict[str, Any]],
    generator: Any,
    args: argparse.Namespace,
    method: str,
) -> List[Dict[str, Any]]:
    """执行混淆样本生成（progressive / gqs）。仅统计生成算法耗时。"""
    logging.info("=== 混淆样本生成 ===")

    generation_results: List[Dict[str, Any]] = []
    generation_time_total_s = 0.0  # 仅算法耗时（累计）
    total_generated_samples = 0    # 跨记录总样本数

    for ner_result in tqdm(ner_results, desc="生成混淆样本"):
        obf_pairs: List[Tuple[str, float]] = []
        mapping: Dict[str, List[str]] = {}

        original_text, ner_text = get_original_and_ner_text(ner_result)
        elapsed_this = 0.0

        if ner_text and has_sensitive_targets({'masked_text': ner_text}):
            t0 = time.perf_counter()
            if method == "progressive":
                obf_pairs, mapping = generator.generate_samples(
                    ner_text,
                    target_samples=args.num_obfuscated_samples,
                    num_intervals=args.num_intervals,
                    dist=args.dist,
                    redundancy_factor=args.redundancy_factor,
                    exclude_reference=True,
                )
            elif method == "gqs":
                obf_pairs, mapping = generate_samples_from_ner(
                    sampler=generator,
                    ner_text=ner_text,
                    target_samples=args.num_obfuscated_samples,
                    num_intervals=args.num_intervals,
                    dist=args.dist,
                    redundancy_factor=args.redundancy_factor,
                    exclude_reference=True,
                )
            elif method == "semantic_random":
                obf_pairs, mapping = generator.generate_samples(
                    ner_text,
                    target_samples=args.num_obfuscated_samples,
                    num_intervals=args.num_intervals,
                    dist=args.dist,
                    redundancy_factor=args.redundancy_factor,
                    exclude_reference=True,
                )
            elif method == "semantic_simple":
                obf_pairs, mapping = generator.generate_samples(
                    ner_text,
                    target_samples=args.num_obfuscated_samples,
                    num_intervals=args.num_intervals,
                    dist=args.dist,
                    redundancy_factor=args.redundancy_factor,
                    exclude_reference=True,
                )
            elapsed_this = time.perf_counter() - t0

            generation_time_total_s += elapsed_this

        # 仅保留文本列表
        obfuscated_texts: List[str] = [t for (t, _p) in (obf_pairs or [])]
        total_generated_samples += len(obfuscated_texts)

        result = {
            'original_text': original_text,
            'ner_formatted_text': ner_text,
            'obfuscated_samples': obfuscated_texts,
            'real_to_fake_mapping': mapping,
            'generation_time_s': elapsed_this,
        }

        generation_results.append(result)

    # 平均“单条伪样本”耗时（仅算法）
    avg_time_per_sample = (generation_time_total_s / total_generated_samples) if total_generated_samples > 0 else 0.0
    logging.info(f"平均单条伪样本时间（仅算法）：{avg_time_per_sample:.6f} 秒/样本，样本数 {total_generated_samples}")
    return generation_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="混淆样本生成")

    # 输入输出
    parser.add_argument(
        "--input_ner_file",
        type=str,
        default="./datasets/synthetic_NER.jsonl",
        help="NER结果JSONL路径（每行一个记录，包含 ner_formatted_text 字段）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./shuffling/results/delta/0.01/gqs/synthetic_NER.jsonl",
        help="输出目录",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="./shuffling/results/delta/0.01/gqs/log.log",
        help="可选：日志文件路径（不提供则仅输出到控制台）",
    )

    # 随机抽样与复现
    parser.add_argument(
        "--num_records",
        type=int,
        default=500,
        help="从输入中随机抽取的记录数（<=0 表示使用全部）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（影响抽样与部分生成随机性）",
    )

    # 设备
    parser.add_argument(
        "--device",
        type=int,
        default=1,
        help="CUDA设备编号；-1 表示使用 CPU",
    )

    # 生成方法与模型
    parser.add_argument(
        "--generation_method",
        type=str,
        choices=["progressive", "gqs", "semantic_random", "semantic_simple"],
        default="gqs",
        help="混淆样本生成方法",
    )
    parser.add_argument(
        "--generation_model",
        type=str,
        default="Qwen/Qwen3-0.6B", # meta-llama/Llama-3.3-70B-Instruct, ./pangu/openPangu-Embedded-7B-V1.1, deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
        help="混淆样本生成模型名称",
    )
    parser.add_argument(
        "--embedding_model_name",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="progressive 方法使用的嵌入模型（设为空则使用生成模型自身嵌入）",
    )

    # 生成参数
    parser.add_argument(
        "--num_obfuscated_samples",
        type=int,
        default=20,
        help="每条记录生成的混淆样本数量",
    )
    parser.add_argument(
        "--num_intervals",
        type=float,
        default=0.01,
        help="量化间隔",
    )
    parser.add_argument(
        "--dist",
        type=str,
        default="rel",
        help="距离类型：'rel' 或 'abs'",
    )
    parser.add_argument(
        "--redundancy_factor",
        type=float,
        default=1.2,
        help="冗余因子（>1 时先超量生成后截断）",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 设备设置
    if args.device is not None and int(args.device) >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{int(args.device)}"
    device = "cuda" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu"

    # 日志/目录
    setup_logging(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)

    # 随机种子（保证抽样可复现）
    set_random_seed(args.seed)

    # 记录实验参数（简要）
    exp_params = {
        'generation_method': args.generation_method,
        'generation_model': args.generation_model,
        'embedding_model_name': args.embedding_model_name,
        'num_obfuscated_samples': args.num_obfuscated_samples,
        'num_intervals': args.num_intervals,
        'dist': args.dist,
        'redundancy_factor': args.redundancy_factor,
        'num_records': args.num_records,
        'seed': args.seed,
        'device': device,
        'input_ner_file': args.input_ner_file,
        'output_dir': args.output_dir,
    }
    try:
        logging.info("生成实验参数: %s", json.dumps(exp_params, ensure_ascii=False))
    except Exception:
        logging.info("生成实验参数: %s", str(exp_params))

    logging.info("加载NER结果...")
    ner_results_all = load_ner_results(args.input_ner_file)
    logging.info(f"输入记录数: {len(ner_results_all)}")

    # 抽样（可复现）
    ner_results = sample_records_deterministic(ner_results_all, args.num_records, args.seed) if args.num_records and args.num_records > 0 else ner_results_all
    logging.info(f"实际用于生成的记录数: {len(ner_results)}")

    # 构建生成器
    if args.generation_method == "progressive":
        embedding_model_name = args.embedding_model_name if args.embedding_model_name and len(str(args.embedding_model_name).strip()) > 0 else None
        generator = setup_obfuscation_generator(args.generation_model, embedding_model_name, device)
    elif args.generation_method == "semantic_random":
        generator = setup_embedding_random_generator(
            model_name=args.generation_model,
            embedding_model_name=args.embedding_model_name,
            device=device,
        )
    elif args.generation_method == "semantic_simple":
        generator = setup_embedding_simple_generator(
            model_name=args.generation_model,
            embedding_model_name=args.embedding_model_name,
            device=device,
        )
    else:
        generator = setup_gqs_generator(args.generation_model, device)

    # 生成
    generation_results = obfuscation_generation(ner_results, generator, args, args.generation_method)

    # 保存
    output_file = os.path.join(args.output_dir, f"{args.generation_method}_generation_results.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for rec in generation_results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logging.info(f"生成结果已保存到: {output_file}")
    total_with_samples = sum(1 for r in generation_results if r.get('obfuscated_samples'))
    logging.info(f"成功生成混淆样本的记录: {total_with_samples}/{len(generation_results)}")

    # 追加：统计平均单条伪样本时间
    total_time = sum((r.get('generation_time_s') or 0.0) for r in generation_results)
    total_samples = sum(len(r.get('obfuscated_samples') or []) for r in generation_results)
    avg_per_sample = (total_time / total_samples) if total_samples > 0 else 0.0
    logging.info(f"平均单条伪样本时间（仅算法）：{avg_per_sample:.6f} 秒/样本，样本数 {total_samples}")


if __name__ == "__main__":
    main()

# cli
# python ./shuffling/generate_from_ner.py --input_ner_file ./datasets/ai4privacy_NER.jsonl --output_dir ./shuffling/results_1029/qwen4b/progressive/ai4privacy_NER --log_file ./shuffling/results_1029/qwen4b/progressive/log.log --generation_method progressive --device 0 --generation_model Qwen/Qwen3-4B

# python ./shuffling/generate_from_ner.py --input_ner_file ./datasets/synthetic_NER.jsonl --output_dir ./shuffling/results_1029/semantic_random/synthetic_NER --log_file ./shuffling/results_1029/semantic_random/log.log --generation_method semantic_random --device 0

# python ./shuffling/generate_from_ner.py --input_ner_file ./datasets/ai4privacy_NER.jsonl --output_dir ./shuffling/results_1029/semantic_simple/ai4privacy_NER --log_file ./shuffling/results_1029/semantic_simple/log.log --generation_method semantic_simple --device 0