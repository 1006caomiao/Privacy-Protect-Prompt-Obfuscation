#!/usr/bin/env python3
"""
隐私保护流水线 - 模块化版本
支持三个独立阶段：
1. NER识别与评估（Presidio vs 真实mask）
2. 混淆样本生成
3. 批量质量评估

每个阶段可独立控制执行
"""

import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "/data1/huggingface"

parser = argparse.ArgumentParser(description="隐私保护流水线")
    
# 输入输出参数
parser.add_argument("--input_file", type=str, default="./processed_datasets/synthetic_dataset/synthetic_dataset_gpt-mini_0519_test.jsonl",
                    help="输入数据文件路径")
parser.add_argument("--output_dir", type=str, default="./shuffling/results_1022/synthetic_dataset_gpt-mini_0519_test",
                    help="输出目录")
parser.add_argument("--log_file", type=str, default="./shuffling/results_1022/synthetic_dataset_gpt-mini_0519_test/pipeline.log",
                    help="日志文件路径")

# 处理参数
parser.add_argument("--num_records", type=int, default=1000,
                    help="处理的记录数量（-1表示全部）")
parser.add_argument("--device", type=int, default=0,
                    help="计算设备")

# 阶段控制参数
parser.add_argument("--enable_ner", action=argparse.BooleanOptionalAction, default=False,
                    help="是否执行NER识别与评估阶段")
parser.add_argument("--enable_generation", action=argparse.BooleanOptionalAction, default=True,
                    help="是否执行混淆样本生成阶段")
parser.add_argument("--enable_evaluation", action=argparse.BooleanOptionalAction, default=False,
                    help="是否执行批量质量评估阶段")

# NER引擎参数
parser.add_argument("--ner_engine", type=str, choices=["spacy", "transformers", "stanza"], default="spacy",
                    help="NER引擎类型: spacy 或 transformers")
parser.add_argument("--transformer_model", type=str, default="tner/roberta-large-ontonotes5",
                    help="Transformer NER模型名称")

# SpaCy引擎配置参数
parser.add_argument("--custom_ignore_labels", type=str, nargs="*", default=None,
                    help="自定义要忽略的实体标签列表，用空格分隔。默认忽略: FAC PRODUCT EVENT WORK_OF_ART LAW LANGUAGE")
parser.add_argument("--custom_entity_mapping", type=str, nargs="*", default=None,
                    help="自定义实体映射，格式: 'SPACY_LABEL:PRESIDIO_LABEL'，用空格分隔。")

# 置信度过滤参数
parser.add_argument("--confidence_threshold", type=float, default=0.5,
                    help="Presidio实体识别的最低置信度阈值（0.0-1.0）。低于此阈值的实体将被过滤掉。默认0.0表示不过滤")

# 模型参数
parser.add_argument("--generation_model", type=str, 
                    default="Qwen/Qwen3-8B",
                    help="混淆样本生成模型")
parser.add_argument("--embedding_model_name", type=str, 
                    default="Qwen/Qwen3-Embedding-0.6B",
                    help="用于混淆样本生成的嵌入模型名称")

# 混淆生成参数
parser.add_argument("--num_obfuscated_samples", type=int, default=20,
                    help="每条记录生成的混淆样本数量")
parser.add_argument("--num_intervals", type=float, default=0.1,
                    help="量化间隔")
parser.add_argument("--dist", type=str, default="rel",
                    help="距离类型")
parser.add_argument("--redundancy_factor", type=float, default=1.3,
                    help="冗余因子")

# 生成方法选择
parser.add_argument("--generation_method", type=str, choices=["progressive", "gqs"], default="gqs",
                    help="混淆样本生成方法: progressive 或 gqs")

# 评估参数
parser.add_argument("--eval_embedding_model_name", type=str, 
                    default="sentence-transformers/all-roberta-large-v1",
                    help="用于embedding相似度评估的SentenceTransformer模型名称")
parser.add_argument("--use_openai_embedding", action=argparse.BooleanOptionalAction, default=False,
                    help="是否使用OpenAI embedding模型")
parser.add_argument("--openai_endpoint", type=str,
                    default="https://api.ephone.chat/v1",
                    help="OpenAI API端点")
parser.add_argument("--openai_api_key", type=str,
                    default="sk-HUjGkYWvOLHJbO6LPCFRwSro45HZixBIWq7uPUIA5YwAd9dE",
                    help="OpenAI API密钥")
parser.add_argument("--openai_embedding_model", type=str,
                    default="text-embedding-3-small",
                    help="OpenAI embedding模型名称")

args = parser.parse_args()

if args.device != "-1":
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device}"

import json
import logging
import re
import time
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider, NerModelConfiguration
import string
from spacy.lang.en.stop_words import STOP_WORDS

# 导入项目现有模块
from utils import load_jsonl, save_jsonl, split_string_to_words, extract_sensitive_info
from progressive_generator import SimpleEnhancedGenerator
from semantic_sampler import GenerativeModelEmbeddingProvider, SemanticSampler, DedicatedEmbeddingModelProvider
from obfuscation_evaluation import ObfuscationEvaluator, calculate_average_evaluation_metrics
# gqs baseline 相关
from gqs import GreedyQuantizedSampler, generate_samples_from_ner
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def setup_logging(log_file: str = None):
    """设置日志配置"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)

# -----------------------------------------------------------------------------
# Presidio 分析器构建函数
# -----------------------------------------------------------------------------

# 将默认忽略标签和实体映射收敛到全局可复用变量，供两种引擎共用

DEFAULT_IGNORE_LABELS = [
    "FAC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "ORDINAL", "DATE_TIME", "ORGANIZATION", "O"
]

DEFAULT_ENTITY_MAPPING = {
    "PERSON": "PERSON",
    "PER": "PERSON",
    "LOCATION": "LOCATION",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
    "MONEY": "MONEY",
    "PERCENT": "PERCENTAGE",
    "CARDINAL": "CARDINAL",
    "ORDINAL": "ORDINAL",
    "QUANTITY": "QUANTITY",
    "NORP": "AFFILIATION",
    "AGE": "AGE",
    "ID": "ID",
    "EMAIL": "EMAIL",
    "PATIENT": "PERSON",
    "STAFF": "PERSON",
    "HOSP": "ORGANIZATION",
    "PATORG": "ORGANIZATION",
    "PHONE": "PHONE_NUMBER",
    "HCW": "PERSON",
    "HOSPITAL": "ORGANIZATION",
    "GPE": "GEO_POLITICAL_ENTITY",
    "FAC": "BUILDING",
}


def setup_presidio(
    engine_type: str = "spacy",
    transformer_model: str = "tner/roberta-large-ontonotes5",
    custom_ignore_labels: List[str] = None,
    custom_entity_mapping: List[str] = None,
    score_threshold: float = 0.0,
):
    # ------------------------------------------------------------------
    # 解析自定义 ignore/mapping 参数
    # ------------------------------------------------------------------
    labels_to_ignore = (
        custom_ignore_labels if custom_ignore_labels is not None else DEFAULT_IGNORE_LABELS
    )

    if custom_entity_mapping:
        parsed_mapping = {}
        for item in custom_entity_mapping:
            if ":" in item:
                src, tgt = item.split(":", 1)
                parsed_mapping[src.strip()] = tgt.strip()
            else:
                logging.warning(
                    f"无效的映射格式: {item}，应为 'SPACY_LABEL:PRESIDIO_LABEL'"
                )
        entity_mapping = {**DEFAULT_ENTITY_MAPPING, **parsed_mapping}
    else:
        entity_mapping = DEFAULT_ENTITY_MAPPING

    logging.info(f"忽略实体: {labels_to_ignore}")
    logging.info(f"实体映射: {entity_mapping}")

    # ------------------------------------------------------------------
    # 构建 NLP 引擎配置
    # ------------------------------------------------------------------

    if engine_type == "transformers":
        nlp_configuration = {
            "nlp_engine_name": "transformers",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": {
                        "spacy": "en_core_web_trf",
                        "transformers": transformer_model,
                    }
                }
            ],
        }
        logging.info(f"使用 Transformer 引擎: {transformer_model}")
    elif engine_type == "stanza":
        nlp_configuration = {
            "nlp_engine_name": "stanza",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": "en",  # stanza 下载 en 模型
                }
            ],
        }
        logging.info("使用 Stanza 引擎")
    else:
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": "en_core_web_trf"
                }
            ],
        }
        logging.info("使用 SpaCy 引擎")

    nlp_configuration["ner_model_configuration"] = {
        "labels_to_ignore": labels_to_ignore,
        "model_to_presidio_entity_mapping": entity_mapping,
    }

    # ------------------------------------------------------------------
    # 创建 AnalyzerEngine，并通过 default_score_threshold 控制置信度
    # ------------------------------------------------------------------

    nlp_engine_provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
    nlp_engine = nlp_engine_provider.create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine, default_score_threshold=score_threshold
    )
    
    return analyzer

def presidio_analyze_text(analyzer: AnalyzerEngine, text: str, confidence_threshold: float = 0.0) -> List[Dict]:
    """使用Presidio分析文本中的敏感实体，并进行后处理去除虚词/符号等"""
    results = analyzer.analyze(text=text, language='en')

    # 初步收集Presidio返回的实体（不再做置信度过滤，Presidio内部已处理）
    entities: List[Dict] = []
    for result in results:
        entities.append({
            'entity_type': result.entity_type,
            'start': result.start,
            'end': result.end,
            'score': result.score,
            'text': text[result.start:result.end]
        })

    # 后处理：裁剪实体前后可能包含的停用词、符号
    entities = post_process_entities(entities, text)
    return entities

def post_process_entities(entities: List[Dict], text: str) -> List[Dict]:
    """对Presidio返回的实体进行后处理：
    1. 去除实体首尾的停用词（如 the, about 等虚词、副词）
    2. 去除首尾的标点符号
    3. 若修剪后实体为空，则丢弃该实体
    4. 若修剪后与已有实体产生重复（start/end 相同），则保留第一个
    """
    if not entities:
        return entities

    punctuation_chars = (set(string.punctuation) - {'%', '$'}) | {"“", "”", "‘", "’", "–", "—", "…"}
    processed = []
    seen_spans = set()

    for ent in entities:
        start = ent['start']
        end = ent['end']

        # 去除首尾空白
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1

        # 去除首尾标点
        while start < end and text[start] in punctuation_chars:
            start += 1
        while end > start and text[end - 1] in punctuation_chars:
            end -= 1

        # 去除首尾停用词
        changed = True
        while changed and start < end:
            changed = False
            # 修剪左侧
            left_segment = text[start:end]
            left_tokens = left_segment.strip().split()
            if left_tokens:
                first_token = left_tokens[0].strip(string.punctuation).lower()
                if first_token in STOP_WORDS:
                    # 找到 first_token 在原文本中的确切结束位置
                    token_end_in_segment = left_segment.find(left_tokens[0]) + len(left_tokens[0])
                    start += token_end_in_segment
                    # 跳过后面的空白
                    while start < end and text[start].isspace():
                        start += 1
                    changed = True
            # 修剪右侧
            right_segment = text[start:end]
            right_tokens = right_segment.strip().split()
            if right_tokens:
                last_token = right_tokens[-1].strip(string.punctuation).lower()
                if last_token in STOP_WORDS:
                    # 找到 last_token 在原文本中的确切起始位置
                    token_start_in_segment = right_segment.rfind(right_tokens[-1])
                    end = start + token_start_in_segment
                    # 去掉最后可能残留的空白
                    while end > start and text[end - 1].isspace():
                        end -= 1
                    changed = True

        # 最终验证
        if end <= start:
            continue  # 空实体
        span = (start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)

        ent_text = text[start:end]
        # --- 若内部存在 'and'/'or'/'&' 等连接词，尝试拆分为多个实体 ---
        split_pattern = re.compile(r"\s+(?:and|or|&)\s+", re.IGNORECASE)
        if split_pattern.search(ent_text):
            parts = []
            cur_abs_start = start
            for m in split_pattern.finditer(ent_text):
                part_text = ent_text[cur_abs_start - start : m.start()].strip()
                if part_text:
                    abs_end = cur_abs_start + len(part_text)
                    parts.append((cur_abs_start, abs_end, part_text))
                cur_abs_start = start + m.end()
            # 处理最后一段
            final_text = ent_text[cur_abs_start - start:].strip()
            if final_text:
                parts.append((cur_abs_start, cur_abs_start + len(final_text), final_text))

            for s_span, e_span, p_text in parts:
                if e_span <= s_span or (s_span, e_span) in seen_spans:
                    continue
                seen_spans.add((s_span, e_span))
                processed.append({
                    **ent,
                    'start': s_span,
                    'end': e_span,
                    'text': p_text
                })
        else:
            processed.append({
                **ent,
                'start': start,
                'end': end,
                'text': ent_text
            })

    # 重新按位置排序
    processed.sort(key=lambda x: x['start'])
    return processed

def convert_presidio_to_ner_format(text: str, entities: List[Dict]) -> str:
    """将Presidio分析结果转换为progressive_generator.py所需的NER格式"""
    if not entities:
        return text
    
    # 按位置排序实体
    entities = sorted(entities, key=lambda x: x['start'])
    
    # 构建NER格式的文本
    ner_text = ""
    last_end = 0
    
    for entity in entities:
        # 添加实体前的文本
        ner_text += text[last_end:entity['start']]
        
        # 添加标注的实体
        entity_text = entity['text']
        entity_type = entity['entity_type']
        ner_text += f"<redacted> {entity_text} ({entity_type}) </redacted>"
        
        last_end = entity['end']
    
    # 添加最后一部分文本
    ner_text += text[last_end:]
    
    return ner_text

def evaluate_presidio_performance(original_text: str, ground_truth_masked: str, 
                                presidio_entities: List[Dict]) -> Dict[str, float]:
    """评估Presidio与真实mask的性能指标"""
    
    # 获取原始文本的所有唯一词汇
    all_words = set(word.lower() for word in split_string_to_words(original_text))
    total_words = len(all_words)
    
    # 从真实masked文本中提取敏感词
    orig_words, ground_truth_sensitive = extract_sensitive_info(original_text, ground_truth_masked)
    ground_truth_set = set(word.lower() for word in ground_truth_sensitive)
    
    # 从Presidio结果中提取敏感词
    presidio_sensitive = []
    for entity in presidio_entities:
        # 将实体文本分词
        entity_words = split_string_to_words(entity['text'])
        presidio_sensitive.extend(entity_words)
    
    presidio_set = set(word.lower() for word in presidio_sensitive)
    
    # 计算混淆矩阵的四个值
    true_positives = len(ground_truth_set & presidio_set)  # 真实敏感且被识别
    false_positives = len(presidio_set - ground_truth_set)  # 非敏感但被识别
    false_negatives = len(ground_truth_set - presidio_set)  # 真实敏感但未识别
    true_negatives = total_words - true_positives - false_positives - false_negatives  # 非敏感且未识别
    
    # 计算指标
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (true_positives + true_negatives) / total_words if total_words > 0 else 0.0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'specificity': specificity,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'total_words': total_words,
        'ground_truth_count': len(ground_truth_set),
        'presidio_count': len(presidio_set)
    }

def setup_obfuscation_generator(model_name: str, embedding_model_name: str, device: str) -> SimpleEnhancedGenerator:
    """设置混淆样本生成器 (progressive 方法)"""
    logging.info(f"加载模型: {model_name}")
    
    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device, 
        torch_dtype=torch.float16
    )
    model.eval()
    
    # 初始化嵌入提供者和采样器
    if embedding_model_name is not None:
        embedding_provider = DedicatedEmbeddingModelProvider(
            model_name=embedding_model_name,
            generation_tokenizer=tokenizer,
            device='cuda',
            precompute_knn=True,
            knn_top_k=500,
            knn_batch_size=256
        )
    else:
        embedding_provider = GenerativeModelEmbeddingProvider(
            model=model,
            tokenizer=tokenizer,
            precompute_knn=True,
            knn_top_k=200,
            knn_batch_size=256
        )
    sampler = SemanticSampler(
        tokenizer=tokenizer,
        model=model,
        embedding_provider=embedding_provider,
        temperature=1.0,
        max_batch_size=32,
        semantic_top_k=300
    )
    
    # 创建生成器
    generator = SimpleEnhancedGenerator(sampler)
    
    return generator

# -----------------------------------------------------------------------------
# GQS 基线生成器
# -----------------------------------------------------------------------------

def setup_gqs_generator(model_name: str, device: str) -> GreedyQuantizedSampler:
    """构建 GQS 贪婪量化采样器 (baseline)"""
    logging.info(f"[GQS] 加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype)
    model.eval()

    sampler = GreedyQuantizedSampler(tokenizer, model, temperature=1.0, max_batch_size=8)
    return sampler

def calculate_presidio_statistics(ner_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算Presidio标注性能统计"""
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_accuracy = 0
    total_specificity = 0
    total_records = len(ner_results)
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_words = 0
    total_ground_truth = 0
    total_presidio = 0
    
    records_with_ground_truth = 0
    records_with_presidio_entities = 0
    
    for record in ner_results:
        metrics = record['performance_metrics']
        
        total_precision += metrics['precision']
        total_recall += metrics['recall'] 
        total_f1 += metrics['f1']
        total_accuracy += metrics['accuracy']
        total_specificity += metrics['specificity']
        
        total_tp += metrics['true_positives']
        total_fp += metrics['false_positives']
        total_fn += metrics['false_negatives']
        total_tn += metrics['true_negatives']
        total_words += metrics['total_words']
        total_ground_truth += metrics['ground_truth_count']
        total_presidio += metrics['presidio_count']
        
        # 统计有真实标注的记录
        if metrics['ground_truth_count'] > 0:
            records_with_ground_truth += 1
            
        if len(record['presidio_entities']) > 0:
            records_with_presidio_entities += 1
    
    avg_precision = total_precision / total_records if total_records > 0 else 0.0
    avg_recall = total_recall / total_records if total_records > 0 else 0.0
    avg_f1 = total_f1 / total_records if total_records > 0 else 0.0
    avg_accuracy = total_accuracy / total_records if total_records > 0 else 0.0
    avg_specificity = total_specificity / total_records if total_records > 0 else 0.0
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    overall_accuracy = (total_tp + total_tn) / total_words if total_words > 0 else 0.0
    overall_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    
    return {
        'summary': {
            'total_records': total_records,
            'records_with_ground_truth': records_with_ground_truth,
            'records_with_presidio_detection': records_with_presidio_entities,
            'presidio_detection_rate': records_with_presidio_entities / total_records if total_records > 0 else 0.0,
            'ground_truth_coverage': records_with_ground_truth / total_records if total_records > 0 else 0.0,
        },
        'metrics': {
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'average_f1': avg_f1,
            'average_accuracy': avg_accuracy,
            'average_specificity': avg_specificity,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'overall_accuracy': overall_accuracy,
            'overall_specificity': overall_specificity,
        },
        'details': {
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_false_negatives': total_fn,
            'total_true_negatives': total_tn,
            'total_words_processed': total_words,
            'total_ground_truth_entities': total_ground_truth,
            'total_presidio_entities': total_presidio,
        }
    }

# === 阶段1：NER识别与评估 ===
def ner_analysis(data: List[Dict], analyzer: AnalyzerEngine, confidence_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """阶段1：NER识别与评估"""
    logging.info("=== 阶段1：NER识别与评估 ===")
    
    ner_results = []
    
    for record in tqdm(data, desc="NER识别"):
        if isinstance(record['text'], list):
            original_text = " ".join(record['text'])
        else:
            original_text = record['text']
        
        if isinstance(record.get('masked_text'), list):
            ground_truth_masked = " ".join([text for text in record['masked_text'] if text is not None])
        else:
            ground_truth_masked = record.get('masked_text', '')
        
        presidio_entities = presidio_analyze_text(analyzer, original_text, confidence_threshold)
        ner_formatted_text = convert_presidio_to_ner_format(original_text, presidio_entities)
        performance_metrics = evaluate_presidio_performance(original_text, ground_truth_masked, presidio_entities)
        
        result = {
            'original_record': record,
            'original_text': original_text,
            'ground_truth_masked': ground_truth_masked,
            'presidio_entities': presidio_entities,
            'ner_formatted_text': ner_formatted_text,
            'performance_metrics': performance_metrics,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        ner_results.append(result)
    
    return ner_results

# === 阶段2：混淆样本生成 ===
def obfuscation_generation(ner_results: List[Dict], generator, args, method: str) -> List[Dict[str, Any]]:
    """阶段2：混淆样本生成 (支持 progressive / gqs)"""
    logging.info("=== 阶段2：混淆样本生成 ===")
    
    generation_results = []
    # 仅统计伪样本生成算法本身的耗时
    generation_time_total_s = 0.0
    generation_time_calls = 0
    
    for ner_result in tqdm(ner_results, desc="生成混淆样本"):
        obfuscated_samples = []
        real_to_fake_mapping = {}
        
        if ner_result['presidio_entities']:  # 只有检测到敏感实体时才生成混淆样本
            if method == "progressive":
                _t0 = time.perf_counter()
                samples, word_mapping = generator.generate_samples(
                    ner_result['ner_formatted_text'],
                    target_samples=args.num_obfuscated_samples,
                    num_intervals=args.num_intervals,
                    dist=args.dist,
                    redundancy_factor=args.redundancy_factor,
                    exclude_reference=True
                )
                _elapsed = time.perf_counter() - _t0
                generation_time_total_s += _elapsed
                generation_time_calls += 1
                obfuscated_samples = samples
                real_to_fake_mapping = word_mapping
            elif method == "gqs":
                _t0 = time.perf_counter()
                samples, word_mapping = generate_samples_from_ner(
                    sampler=generator,
                    ner_text=ner_result['ner_formatted_text'],
                    target_samples=args.num_obfuscated_samples,
                    num_intervals=args.num_intervals,
                    dist=args.dist,
                    redundancy_factor=args.redundancy_factor,
                    exclude_reference=True
                )
                _elapsed = time.perf_counter() - _t0
                generation_time_total_s += _elapsed
                generation_time_calls += 1
                obfuscated_samples = samples
                real_to_fake_mapping = word_mapping
         
        result = {
            **ner_result,  # 包含NER阶段的所有结果
            'obfuscated_samples': obfuscated_samples,
            'real_to_fake_mapping': real_to_fake_mapping,
            'generation_info': {
                'num_obfuscated_samples': len(obfuscated_samples),
                'num_real_words': len(real_to_fake_mapping),
                'generation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                # 单条记录的伪样本生成算法耗时（秒），无生成则为0
                'generation_time_s': _elapsed if ner_result['presidio_entities'] else 0.0
            }
        }
        
        generation_results.append(result)
    # 输出平均伪样本生成算法耗时（不包含其它流程）
    avg_time = (generation_time_total_s / generation_time_calls) if generation_time_calls > 0 else 0.0
    logging.info(f"平均伪样本生成时间（仅算法）：{avg_time:.4f} 秒/记录，统计 {generation_time_calls} 条")

    return generation_results

# === 阶段3：批量质量评估 ===
def quality_evaluation(generation_results: List[Dict], evaluator: ObfuscationEvaluator) -> Tuple[List[Dict], Dict]:
    """阶段3：批量质量评估"""
    logging.info("=== 阶段3：批量质量评估 ===")
    
    # 收集所有需要评估的记录
    records_to_evaluate = [r for r in generation_results if r.get('obfuscated_samples')]
    
    if not records_to_evaluate:
        logging.warning("没有找到需要评估的混淆样本")
        return generation_results, {}
    
    logging.info(f"开始批量评估 {len(records_to_evaluate)} 条记录的混淆样本")

    # STEP 1: 一次性计算文本相似度，内部自动写回记录
    evaluator.batch_evaluate_text_similarity(records_to_evaluate)

    # STEP 2: 逐条补充 embedding 相似度（仅计算余弦相似度）
    evaluation_results = [
        evaluator.evaluate_single_embedding_similarity(rec)
        for rec in tqdm(records_to_evaluate, desc="批量余弦相似度评估")
    ]

    # 计算整体评估统计
    overall_stats = calculate_average_evaluation_metrics(evaluation_results)
    
    return evaluation_results, overall_stats

def main():
    # 设置日志和输出目录
    setup_logging(args.log_file)
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info("开始隐私保护流水线")
    logging.info(f"输入文件: {args.input_file}")
    logging.info(f"输出目录: {args.output_dir}")
    logging.info(f"启用阶段: NER={args.enable_ner}, 生成={args.enable_generation}, 评估={args.enable_evaluation}")
    if args.enable_ner:
        logging.info(f"NER引擎: {args.ner_engine}")
        if args.ner_engine == "transformers":
            logging.info(f"Transformer模型: {args.transformer_model}")
    
    # 加载数据
    logging.info("加载数据")
    data = load_jsonl(args.input_file)
    if args.num_records > 0:
        data = data[:args.num_records]
    logging.info(f"加载了 {len(data)} 条记录")
    
    start_time = time.time()
    results = None
    
    # 阶段1：NER识别与评估
    if args.enable_ner:
        analyzer = setup_presidio(
            engine_type=args.ner_engine,
            transformer_model=args.transformer_model,
            custom_ignore_labels=args.custom_ignore_labels,
            custom_entity_mapping=args.custom_entity_mapping,
            score_threshold=args.confidence_threshold,
        )
        ner_results = ner_analysis(data, analyzer, args.confidence_threshold)
        
        # 计算和保存NER统计
        ner_stats = calculate_presidio_statistics(ner_results)
        
        # 保存NER结果
        ner_output_file = os.path.join(args.output_dir, "ner_results.jsonl")
        with open(ner_output_file, 'w', encoding='utf-8') as f:
            for result in ner_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.write(json.dumps({"type": "ner_statistics", "data": ner_stats}, ensure_ascii=False) + '\n')
        
        logging.info(f"NER结果已保存到: {ner_output_file}")
        logging.info(f"NER整体F1: {ner_stats['metrics']['overall_f1']:.3f}")
        
        results = ner_results
    else:
        # 如果不执行NER，尝试加载之前的结果
        ner_output_file = os.path.join(args.output_dir, "ner_results.jsonl")
        if os.path.exists(ner_output_file):
            logging.info(f"从文件加载NER结果: {ner_output_file}")
            ner_results = []
            with open(ner_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    record = json.loads(line)
                    if record.get('type') != 'ner_statistics':
                        ner_results.append(record)
            # 若仅执行第二阶段，则也按 num_records 限制数量
            results = ner_results[:args.num_records] if args.num_records > 0 else ner_results
        else:
            logging.error("未找到NER结果文件，请先执行NER阶段")
            return
    
    # 阶段2：混淆样本生成
    if args.enable_generation and results:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.generation_method == "progressive":
            generator = setup_obfuscation_generator(args.generation_model, args.embedding_model_name, device)
        else:
            generator = setup_gqs_generator(args.generation_model, device)

        generation_results = obfuscation_generation(results, generator, args, args.generation_method)
        
        # 保存生成结果
        generation_output_file = os.path.join(args.output_dir, f"{args.generation_method}_generation_results.jsonl")
        with open(generation_output_file, 'w', encoding='utf-8') as f:
            for result in generation_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logging.info(f"生成结果已保存到: {generation_output_file}")
        
        # 统计生成情况
        total_with_samples = sum(1 for r in generation_results if r.get('obfuscated_samples'))
        logging.info(f"成功生成混淆样本的记录: {total_with_samples}/{len(generation_results)}")
        
        results = generation_results
    elif args.enable_generation:
        logging.error("没有NER结果可用于生成混淆样本")
        return
    elif not args.enable_generation and results:
        # 如果不执行生成，尝试加载之前的结果
        generation_output_file = os.path.join(args.output_dir, f"{args.generation_method}_generation_fixed_results.jsonl")
        if os.path.exists(generation_output_file):
            logging.info(f"从文件加载生成结果: {generation_output_file}")
            generation_results = []
            with open(generation_output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    generation_results.append(json.loads(line))
            results = generation_results
    
    # 阶段3：批量质量评估
    if args.enable_evaluation and results:
        if args.use_openai_embedding:
            api_key = args.openai_api_key or os.getenv('OPENAI_API_KEY')
            evaluator = ObfuscationEvaluator(
                use_openai=True,
                endpoint=args.openai_endpoint,
                openai_api_key=api_key,
                openai_model=args.openai_embedding_model
            )
        else:
            evaluator = ObfuscationEvaluator(embedding_model_name=args.eval_embedding_model_name)
        
        final_results, evaluation_stats = quality_evaluation(results, evaluator)

        # 仅保存评估统计结果
        eval_output_file = os.path.join(args.output_dir, f"{args.generation_method}_evaluation_stats.json")
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_stats, f, ensure_ascii=False, indent=2)

        logging.info(f"评估统计已保存到: {eval_output_file}")

        # 打印所有平均指标
        if evaluation_stats:
            logging.info("=== 平均文本相似度指标 ===")
            for k, v in evaluation_stats.get('avg_text_similarity', {}).items():
                logging.info(f"  {k}: {v:.4f}")

            logging.info("=== 平均Embedding相似度指标 ===")
            for k, v in evaluation_stats.get('avg_embedding_similarity', {}).items():
                logging.info(f"  {k}: {v:.4f}")

            logging.info("=== 原文与NER模板文本相似度指标 ===")
            for k, v in evaluation_stats.get('avg_ner_template_similarity', {}).items():
                logging.info(f"  {k}: {v:.4f}")

            logging.info("=== 数据概要 ===")
            for k, v in evaluation_stats.get('summary', {}).items():
                logging.info(f"  {k}: {v}")

        results = final_results
    
    total_time = time.time() - start_time
    logging.info(f"流水线完成，总耗时: {total_time:.2f}秒")

if __name__ == "__main__":
    main() 