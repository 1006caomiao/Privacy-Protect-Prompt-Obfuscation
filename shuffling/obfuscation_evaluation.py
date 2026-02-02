#!/usr/bin/env python3
"""
混淆样本评估模块
主要功能：
1. 使用metrics.py中的指标评估伪样本与真实样本的文本相似度
2. 使用embedding模型计算真实词与伪词的语义相似度
3. 从混淆样本中采样对应的伪词进行评估
"""

import logging
import re
from typing import List, Dict, Tuple, Any
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 导入项目模块
from metrics import evaluate_sim
from utils import split_string_to_words, extract_sensitive_info

# 导入embedding相关库
from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 导入OpenAI相关库

import openai
from openai import OpenAI

class ObfuscationEvaluator:
    """混淆样本评估器"""
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-roberta-large-v1", 
                 use_openai: bool = False, endpoint: str = None, openai_api_key: str = None, 
                 openai_model: str = "text-embedding-3-small"):
        """
        初始化评估器
        
        Args:
            embedding_model_name: SentenceTransformer模型名称
            use_openai: 是否使用OpenAI embedding
            endpoint: OpenAI API地址
            openai_api_key: OpenAI API密钥
            openai_model: OpenAI embedding模型名称
        """
        self.use_openai = use_openai
        
        if use_openai:
            self.openai_client = OpenAI(api_key=openai_api_key, base_url=endpoint)
            self.openai_model = openai_model
            self.embedding_model = None
            logging.info(f"成功初始化OpenAI embedding客户端，模型: {openai_model}")
        else:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.openai_client = None
            self.openai_model = None
            logging.info(f"成功加载SentenceTransformer模型: {embedding_model_name}")

        # 预加载文本相似度评估所需的指标，避免每条记录重复加载模型（尤其是BERTScore）
        import evaluate as _hf_evaluate
        self._bleu_metric = _hf_evaluate.load("bleu")
        self._rouge_metric = _hf_evaluate.load("rouge")
        self._bertscore_metric = _hf_evaluate.load("bertscore")

    # -------------------------------------------------------------
    # 文本规范化辅助：
    #   1. 仅保留字母/数字/空白和基础标点 . , ! ? : ; -
    #   2. 连续空白压缩为单空格
    # -------------------------------------------------------------
    @staticmethod
    def _normalize_text(text: str) -> str:
        import string, re
        allowed_punct = set(".,!?")
        cleaned_chars = [ch if ch.isalnum() or ch in allowed_punct or ch.isspace() else " " for ch in text]
        cleaned = "".join(cleaned_chars)
        return re.sub(r"\s+", " ", cleaned).strip()

    @classmethod
    def _normalize_list(cls, texts):
        return [cls._normalize_text(t) for t in texts]

    # ------------------------------------------------------------------
    # 批量文本相似度评估：一次性跑 BLEU / ROUGE / BERTScore，随后写回记录
    # ------------------------------------------------------------------
    def batch_evaluate_text_similarity(self, records: List[Dict[str, Any]]):
        """给定 generation 结果记录列表，一次性计算所有文本相似度并写入每条记录的
        record['text_similarity'] 字段，避免在 evaluate_single_record 中重复计算。
        该函数只负责文本级别指标，不处理 embedding 相似度。
        """

        # 收集所有预测/参考对
        all_preds, all_refs, belong_idx = [], [], []
        for idx, rec in enumerate(records):
            obf = rec.get('obfuscated_samples', [])
            if not obf:
                continue

            if isinstance(obf[0], dict):
                texts = [s['text'] for s in obf]
            elif isinstance(obf[0], (tuple, list)):
                texts = [s[0] for s in obf]
            else:
                texts = obf

            all_preds.extend(texts)
            all_refs.extend([rec.get('original_text', '')] * len(texts))
            belong_idx.extend([idx] * len(texts))

        if not all_preds:
            return  # 无内容可评估

        # ---- 统一指标计算 ----
        all_preds = self._normalize_list(all_preds)
        all_refs = self._normalize_list(all_refs)

        bleu_res = self._bleu_metric.compute(predictions=all_preds, references=all_refs)

        rouge_res = self._rouge_metric.compute(
            predictions=all_preds,
            references=all_refs,
            use_aggregator=False,
        )

        bert_res = self._bertscore_metric.compute(
            predictions=all_preds,
            references=all_refs,
            lang="en",
        )

        # ---- 按记录聚合 ----
        agg = [
            {
                'bleu': [],
                'rouge1': [],
                'rouge2': [],
                'rougeL': [],
                'rougeLsum': [],
                'bertscore_precision': [],
                'bertscore_recall': [],
                'bertscore_f1': []
            }
            for _ in records
        ]

        for i, rec_id in enumerate(belong_idx):
            agg[rec_id]['bleu'].append(bleu_res['bleu'])  # corpus bleu，相同值
            agg[rec_id]['rouge1'].append(rouge_res['rouge1'][i])
            agg[rec_id]['rouge2'].append(rouge_res['rouge2'][i])
            agg[rec_id]['rougeL'].append(rouge_res['rougeL'][i])
            agg[rec_id]['rougeLsum'].append(rouge_res['rougeLsum'][i])
            agg[rec_id]['bertscore_precision'].append(bert_res['precision'][i])
            agg[rec_id]['bertscore_recall'].append(bert_res['recall'][i])
            agg[rec_id]['bertscore_f1'].append(bert_res['f1'][i])

        for rec_id, stats in enumerate(agg):
            if not stats['bleu']:
                continue  # 该记录没有伪样本
            metrics = {k: float(np.mean(v)) for k, v in stats.items()}
            metrics['num_samples'] = len(stats['bleu'])
            records[rec_id]['text_similarity'] = metrics

        # ------------------------------------------------------------------
        # (新增) 计算原文 vs NER 模板的文本相似度 —— 批量方式
        # ------------------------------------------------------------------
        ner_preds, ner_refs, ner_idx = [], [], []
        for idx, rec in enumerate(records):
            ner_raw = rec.get('ner_formatted_text')
            if not ner_raw:
                continue
            # 将 <redacted> 实体 (TYPE) </redacted> 压缩为占位符 <redacted>
            clean_ner = re.sub(r'<redacted>\s*[^()<>]+?\s*\([^)]+\)\s*</redacted>', '<redacted>', ner_raw, flags=re.DOTALL)
            ner_preds.append(clean_ner)
            ner_refs.append(rec.get('original_text', ''))
            ner_idx.append(idx)

        if ner_preds:
            ner_preds = self._normalize_list(ner_preds)
            ner_refs = self._normalize_list(ner_refs)

            bleu_ner = self._bleu_metric.compute(predictions=ner_preds, references=ner_refs)
            rouge_ner = self._rouge_metric.compute(
                predictions=ner_preds,
                references=ner_refs,
                use_aggregator=False,
            )
            bert_ner = self._bertscore_metric.compute(
                predictions=ner_preds,
                references=ner_refs,
                lang="en",
            )

            for i, rec_id in enumerate(ner_idx):
                metrics = {
                    'bleu': bleu_ner['bleu'],
                    'rouge1': rouge_ner['rouge1'][i],
                    'rouge2': rouge_ner['rouge2'][i],
                    'rougeL': rouge_ner['rougeL'][i],
                    'rougeLsum': rouge_ner['rougeLsum'][i],
                    'bertscore_precision': bert_ner['precision'][i],
                    'bertscore_recall': bert_ner['recall'][i],
                    'bertscore_f1': bert_ner['f1'][i],
                    'num_samples': 1,
                }
                records[rec_id]['ner_template_similarity'] = metrics
        
        # 释放 GPU 显存（若使用 GPU 版 BERTScore）
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        统一的文本编码方法，支持SentenceTransformer和OpenAI
        
        Args:
            texts: 文本列表
            
        Returns:
            embedding矩阵
        """
        if self.use_openai:
            # 使用OpenAI API
            response = self.openai_client.embeddings.create(
                input=texts,
                model=self.openai_model
            )
            embeddings = [data.embedding for data in response.data]
            return np.array(embeddings)
        else:
            # 使用SentenceTransformer
            # 禁用SentenceTransformer默认的进度条显示
            return self.embedding_model.encode(texts, show_progress_bar=False)
        
    def evaluate_text_similarity(self, original_text: str, obfuscated_samples: List[str]) -> Dict[str, float]:
        """
        评估原始文本与混淆样本的文本相似度
        
        Args:
            original_text: 原始文本
            obfuscated_samples: 混淆样本列表
            
        Returns:
            包含各种相似度指标的字典
        """
        if not obfuscated_samples:
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rouge2": 0.0,
                "rougeL": 0.0,
                "rougeLsum": 0.0,
                "bertscore_precision": 0.0,
                "bertscore_recall": 0.0,
                "bertscore_f1": 0.0,
                "num_samples": 0
            }
        
        # 准备数据：每个混淆样本与原始文本比较
        pred_list = self._normalize_list(obfuscated_samples)
        gt_list = self._normalize_list([original_text] * len(obfuscated_samples))
        
        # --- 计算相似度指标（复用预加载的 evaluate.Metric 对象） ---
        # BLEU
        bleu_results = self._bleu_metric.compute(predictions=pred_list, references=gt_list)
        # ROUGE（sentence 级，使用聚合器=False 获取列表，再取平均）
        rouge_results = self._rouge_metric.compute(
            predictions=pred_list,
            references=gt_list,
            use_aggregator=False,
        )  # 返回 dict，每个 key 对应 list

        # BERTScore
        bert_results = self._bertscore_metric.compute(
            predictions=pred_list,
            references=gt_list,
            lang="en",
        )

        metrics = {
            "bleu": bleu_results["bleu"],
            "rouge1": float(np.mean(rouge_results["rouge1"])),
            "rouge2": float(np.mean(rouge_results["rouge2"])),
            "rougeL": float(np.mean(rouge_results["rougeL"])),
            "rougeLsum": float(np.mean(rouge_results["rougeLsum"])),
            "bertscore_precision": float(np.mean(bert_results["precision"])),
            "bertscore_recall": float(np.mean(bert_results["recall"])),
            "bertscore_f1": float(np.mean(bert_results["f1"])),
        }
        
        # 添加样本数量信息
        metrics["num_samples"] = len(obfuscated_samples)
        
        return metrics

    
    def evaluate_embedding_similarity(self, real_word: str, fake_words: List[str]) -> List[float]:
        """
        计算一个真词与多个伪词的相似度
        真词embedding只计算一次，然后与每个伪词比较
        
        Args:
            real_word: 真实词
            fake_words: 伪词列表
            
        Returns:
            相似度列表，与fake_words对应
        """
        if not fake_words:
            return []
        
        # 真词embedding只计算一次
        real_embedding = self._encode_texts([real_word])[0]
        
        # 批量计算所有伪词的embedding
        fake_embeddings = self._encode_texts(fake_words)
        
        # 计算相似度
        similarities = []
        for fake_embedding in fake_embeddings:
            similarity = cosine_similarity(
                real_embedding.reshape(1, -1), 
                fake_embedding.reshape(1, -1)
            )[0][0]
            similarities.append(float(similarity))
        
        return similarities
    
    def evaluate_single_embedding_similarity(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        仅评估单条记录的余弦相似度（embedding 相似度），
        文本相似度已在 batch_evaluate_text_similarity 中一次性计算并写入 record。
        
        Args:
            record: 包含原始文本、NER格式文本和混淆样本的记录
            
        Returns:
            评估结果字典，仅更新 embedding 相似度，保留已有文本相似度信息
        """
        original_text = record.get('original_text', '')
        
        # 处理混淆样本数据格式
        obfuscated_samples_data = record.get('obfuscated_samples', [])
        if obfuscated_samples_data and isinstance(obfuscated_samples_data[0], dict):
            # 如果是字典格式 [{'text': ..., 'probability': ...}, ...]
            obfuscated_samples = [sample['text'] for sample in obfuscated_samples_data]
        elif obfuscated_samples_data and isinstance(obfuscated_samples_data[0], tuple):
            # 如果是元组格式 [(text, prob), ...]
            obfuscated_samples = [sample[0] for sample in obfuscated_samples_data]
        elif obfuscated_samples_data and isinstance(obfuscated_samples_data[0], list):
            # 如果是列表格式 [[text, prob], ...]
            obfuscated_samples = [sample[0] for sample in obfuscated_samples_data]
        else:
            obfuscated_samples = []
        
        # 1. 文本相似度已由 batch_evaluate_text_similarity 预先计算，这里直接读取
        text_similarity = record.get('text_similarity', {})
        
        # 2. 直接使用生成器提供的真词到伪词映射数据
        real_to_fake_mapping = record.get('real_to_fake_mapping', {})
        all_similarities = []
        
        # 3. 使用优化的方法计算embedding相似度
        for real_word, fake_words in real_to_fake_mapping.items():
            # 一个真词对多个伪词，真词embedding只计算一次
            similarities = self.evaluate_embedding_similarity(real_word, fake_words)
            all_similarities.extend(similarities)
        
        # 计算统计指标
        if all_similarities:
            similarities_array = np.array(all_similarities)
            embedding_similarity = {
                "avg_cosine_similarity": float(np.mean(similarities_array)),
                "std_cosine_similarity": float(np.std(similarities_array)),
                "min_cosine_similarity": float(np.min(similarities_array)),
                "max_cosine_similarity": float(np.max(similarities_array)),
                "num_word_pairs": len(all_similarities)
            }
        else:
            embedding_similarity = {
                "avg_cosine_similarity": 0.0,
                "std_cosine_similarity": 0.0,
                "min_cosine_similarity": 0.0,
                "max_cosine_similarity": 0.0,
                "num_word_pairs": 0
            }
        
        return {
            'text_similarity': text_similarity,
            'ner_template_similarity': record.get('ner_template_similarity', {}),
            'embedding_similarity': embedding_similarity,
            'evaluation_info': {
                'num_obfuscated_samples': len(obfuscated_samples),
                'num_word_pairs_extracted': len(all_similarities)
            }
        }


def calculate_average_evaluation_metrics(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算所有记录的平均评估指标
    
    Args:
        evaluation_results: 所有记录的评估结果列表
        
    Returns:
        平均指标字典
    """
    if not evaluation_results:
        return {
            'avg_text_similarity': {},
            'avg_embedding_similarity': {},
            'summary': {
                'total_records': 0,
                'total_obfuscated_samples': 0,
                'total_word_pairs': 0
            }
        }
    
    # 初始化累加器
    text_metrics_sum = defaultdict(float)
    embedding_metrics_sum = defaultdict(float)
    ner_tpl_metrics_sum = defaultdict(float)

    valid_text_count = 0
    valid_embedding_count = 0
    valid_ner_tpl_count = 0
    total_samples = 0
    total_word_pairs = 0
    
    for result in evaluation_results:
        text_sim = result.get('text_similarity', {})
        emb_sim = result.get('embedding_similarity', {})
        ner_sim = result.get('ner_template_similarity', {})
        
        # 累加文本相似度指标
        if text_sim.get('num_samples', 0) > 0:
            valid_text_count += 1
            for key, value in text_sim.items():
                if key != 'num_samples' and isinstance(value, (int, float)):
                    text_metrics_sum[key] += value
        
        # 累加 NER 模板文本相似度指标
        if ner_sim.get('num_samples', 0) > 0:
            valid_ner_tpl_count += 1
            for key, value in ner_sim.items():
                if key != 'num_samples' and isinstance(value, (int, float)):
                    ner_tpl_metrics_sum[key] += value

        # 累加embedding相似度指标
        if emb_sim.get('num_word_pairs', 0) > 0:
            valid_embedding_count += 1
            for key, value in emb_sim.items():
                if key != 'num_word_pairs' and isinstance(value, (int, float)):
                    embedding_metrics_sum[key] += value
        
        # 统计总数
        total_samples += result.get('evaluation_info', {}).get('num_obfuscated_samples', 0)
        total_word_pairs += result.get('evaluation_info', {}).get('num_word_pairs_extracted', 0)
    
    # 计算平均值
    avg_text_similarity = {}
    if valid_text_count > 0:
        for key, total in text_metrics_sum.items():
            avg_text_similarity[key] = total / valid_text_count
    
    avg_embedding_similarity = {}
    if valid_embedding_count > 0:
        for key, total in embedding_metrics_sum.items():
            avg_embedding_similarity[key] = total / valid_embedding_count

    avg_ner_template_similarity = {}
    if valid_ner_tpl_count > 0:
        for key, total in ner_tpl_metrics_sum.items():
            avg_ner_template_similarity[key] = total / valid_ner_tpl_count
    
    return {
        'avg_text_similarity': avg_text_similarity,
        'avg_embedding_similarity': avg_embedding_similarity,
        'avg_ner_template_similarity': avg_ner_template_similarity,
        'summary': {
            'total_records': len(evaluation_results),
            'records_with_text_similarity': valid_text_count,
            'records_with_embedding_similarity': valid_embedding_count,
            'records_with_ner_template_similarity': valid_ner_tpl_count,
            'total_obfuscated_samples': total_samples,
            'total_word_pairs': total_word_pairs,
            'avg_samples_per_record': total_samples / len(evaluation_results) if evaluation_results else 0,
            'avg_word_pairs_per_record': total_word_pairs / len(evaluation_results) if evaluation_results else 0
        }
    }


def main():
    """测试函数"""
    print("=== 混淆样本评估模块测试 ===")
    
    # 创建评估器（默认使用SentenceTransformer）
    print("使用SentenceTransformer进行测试...")
    evaluator = ObfuscationEvaluator()
    
    # 如果要测试OpenAI，可以取消注释以下行
    # evaluator = ObfuscationEvaluator(
    #     use_openai=True,
    #     endpoint="https://api.ephone.chat/v1",
    #     openai_api_key="sk-HUjGkYWvOLHJbO6LPCFRwSro45HZixBIWq7uPUIA5YwAd9dE",
    #     openai_model="text-embedding-3-small"
    # )
    
    # 测试数据
    test_record = {
        'original_text': "Hello John, I work at Microsoft in Seattle.",
        'ner_formatted_text': "Hello <redacted> John (PERSON) </redacted>, I work at <redacted> Microsoft (ORG) </redacted> in <redacted> Seattle (LOC) </redacted>.",
        'obfuscated_samples': [
            ("Hello Alice, I work at Google in Portland.", 0.8),
            ("Hello Bob, I work at Apple in Denver.", 0.7),
            ("Hello Charlie, I work at Facebook in Austin.", 0.6)
        ],
        'real_to_fake_mapping': {
            "John": ["Alice", "Bob", "Charlie"],
            "Microsoft": ["Google", "Apple", "Facebook"], 
            "Seattle": ["Portland", "Denver", "Austin"]
        }
    }
    
    # 评估单条记录
    result = evaluator.evaluate_single_embedding_similarity(test_record)
    print(f"文本相似度: {result['text_similarity']}")
    print(f"Embedding相似度: {result['embedding_similarity']}")
    
    # 计算平均指标
    avg_metrics = calculate_average_evaluation_metrics([result])
    print(f"平均指标: {avg_metrics}")


if __name__ == "__main__":
    main() 