import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
import random
import re
import copy
from typing import List, Tuple, Dict
from tqdm import tqdm

def filter_probs_abs(probs, tgt_prob, num_intervals):
    lower = math.floor(tgt_prob * num_intervals) / num_intervals + 1e-3
    upper = lower + 1.0 / num_intervals
    
    qualifying_token_ids = ((probs >= lower) & (probs < upper)).nonzero().squeeze(-1).cpu().numpy()
    return qualifying_token_ids


def filter_probs_rel(probs, tgt_prob, magnitude: int = 1):
    rel_dist = torch.abs(probs/tgt_prob - 1)
    qualifying_token_ids = (rel_dist <= magnitude).nonzero().squeeze(-1).cpu().numpy()
    return qualifying_token_ids


class Candidate:
    """候选序列，只存储生成的部分（不包含前缀）"""
    def __init__(self):
        self.tokens: List[int] = []  # 生成的token序列
        self.prob: float = 1.0       # 累积概率
    
    def clone(self):
        new_candidate = Candidate()
        new_candidate.tokens = self.tokens.copy()
        new_candidate.prob = self.prob
        return new_candidate

# --- 嵌入服务提供者抽象 ---

class BaseEmbeddingProvider:
    """
    一个抽象基类，用于提供词向量和查找相似词元。
    """
    def __init__(self, model, tokenizer, precompute_knn: bool = False, knn_top_k: int = 200, knn_batch_size: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.device = model.device if hasattr(model, 'device') else 'cpu'
        # KNN 开关与参数（由 Provider 决定是否预计算/使用KNN）
        self.precompute_knn = precompute_knn
        self.knn_top_k = knn_top_k
        self.knn_batch_size = knn_batch_size
        # 预计算KNN缓存（两种形式：全量矩阵或按需逐token缓存）
        self.knn_neighbors_full = None   # np.ndarray[vocab_size, k] 或 None
        self.knn_k_full = 0
        self.knn_neighbors_map: Dict[int, List[int]] = {}

    def get_similar_tokens(self, token_id: int, top_k: int = 100) -> List[int]:
        """
        为一个给定的token_id找到最相似的top_k个token的ID列表。

        Args:
            token_id: 目标词元的ID。
            top_k: 返回的最相似词元的数量。

        Returns:
            相似词元ID的列表。
        """
        raise NotImplementedError

    # -- KNN 预计算/缓存 API --
    def _get_precomputed_neighbors(self, token_id: int, top_k: int) -> List[int] | None:
        if self.knn_neighbors_full is not None and self.knn_neighbors_full.shape[1] >= top_k:
            return self.knn_neighbors_full[token_id][:top_k].tolist()
        cached = self.knn_neighbors_map.get(token_id)
        if cached is not None and len(cached) >= top_k:
            return cached[:top_k]
        return None

    def _store_neighbors_map(self, token_id: int, neighbors: List[int]):
        self.knn_neighbors_map[token_id] = list(neighbors)

    def build_full_knn_cache(self, top_k: int, batch_size: int = 128):
        """为全部词元预计算 top_k 近邻。
        由子类实现（GPU/CPU 差异）。
        """
        raise NotImplementedError

class GenerativeModelEmbeddingProvider(BaseEmbeddingProvider):
    """
    使用生成模型自身的词向量来查找相似词元。
    这是推荐的方案，因为它能确保语义空间的一致性。
    """
    def __init__(self, model, tokenizer, precompute_knn: bool = False, knn_top_k: int = 200, knn_batch_size: int = 128):
        super().__init__(model, tokenizer, precompute_knn=precompute_knn, knn_top_k=knn_top_k, knn_batch_size=knn_batch_size)
        print("正在初始化生成模型的词向量...")
        # 保持词向量在GPU上以加速计算
        embedding_weight = self.model.get_input_embeddings().weight.detach()
        
        # 安全的归一化：处理零向量和NaN
        norms = torch.norm(embedding_weight, dim=1, keepdim=True)
        # 将零范数替换为1，避免除零错误
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        self.embedding_matrix_gpu = embedding_weight / norms
        
        # 检查并处理NaN值
        nan_mask = torch.isnan(self.embedding_matrix_gpu)
        if torch.any(nan_mask):
            # print(f"警告: 发现 {torch.sum(nan_mask)} 个NaN值，将其替换为0")
            self.embedding_matrix_gpu[nan_mask] = 0
        
        # 为了兼容sklearn的cosine_similarity，保留一个CPU版本
        self.embedding_matrix = self.embedding_matrix_gpu.cpu().numpy()
        
        # 选项：预计算KNN
        if self.precompute_knn:
            try:
                self.build_full_knn_cache(top_k=self.knn_top_k, batch_size=self.knn_batch_size)
            except Exception as e:
                print(f"预计算KNN失败（生成模型嵌入）: {e}")

    def get_similar_tokens(self, token_id: int, top_k: int = 100) -> List[int]:
        if token_id >= self.embedding_matrix_gpu.shape[0]:
            return []  # 无效的token id
        # 若启用KNN优先使用预计算（或按需KNN缓存）
        if self.precompute_knn:
            pre = self._get_precomputed_neighbors(token_id, top_k)
            if pre is not None:
                return pre

        # 使用GPU加速的相似度计算
        with torch.no_grad():
            target_vector = self.embedding_matrix_gpu[token_id:token_id+1]
            
            # 检查目标向量是否包含NaN
            if torch.any(torch.isnan(target_vector)):
                # print(f"警告: token_id {token_id} 的向量包含NaN，跳过相似度计算")
                return []
            
            # GPU上计算余弦相似度（已归一化，直接做点积）
            sim_scores = torch.mm(target_vector, self.embedding_matrix_gpu.t()).squeeze(0)
            
            # 检查相似度分数是否包含NaN
            if torch.any(torch.isnan(sim_scores)):
                # print(f"警告: token_id {token_id} 的相似度计算结果包含NaN，使用备用方案")
                # 备用方案：返回随机的top_k个token（排除自身）
                all_indices = list(range(self.embedding_matrix_gpu.shape[0]))
                all_indices.remove(token_id)
                return np.random.choice(all_indices, min(top_k, len(all_indices)), replace=False).tolist()
            
            # 获取top_k个最相似的（排除自身）
            # 将自身的相似度设为负无穷，避免选中
            sim_scores[token_id] = float('-inf')
            
            # 使用torch.topk获取最大的k个
            _, top_indices = torch.topk(sim_scores, min(top_k, sim_scores.shape[0]))
            
            result = top_indices.cpu().tolist()
            # 若启用KNN模式，按需缓存该 token 的近邻
            if self.precompute_knn:
                self._store_neighbors_map(token_id, result)
            return result

    def build_full_knn_cache(self, top_k: int, batch_size: int = 128):
        vocab_size = self.embedding_matrix_gpu.shape[0]
        k = min(top_k, vocab_size-1)
        neighbors = np.zeros((vocab_size, k), dtype=np.int32)
        with torch.no_grad():
            embT = self.embedding_matrix_gpu.t().contiguous()
            for start in tqdm(range(0, vocab_size, batch_size), desc="构建KNN(GPU)"):
                end = min(start + batch_size, vocab_size)
                queries = self.embedding_matrix_gpu[start:end]  # [b, d]
                sims = torch.mm(queries, embT)  # [b, V]
                # 排除自身
                ar = torch.arange(end - start, device=sims.device)
                sims[ar, start + ar] = float('-inf')
                # 取 topk
                _, top_idx = torch.topk(sims, k, dim=1)
                neighbors[start:end, :] = top_idx.cpu().numpy().astype(np.int32)
                del sims, top_idx, queries
                torch.cuda.empty_cache()
        self.knn_neighbors_full = neighbors
        self.knn_k_full = k

class DedicatedEmbeddingModelProvider(BaseEmbeddingProvider):
    """
    使用专门的嵌入模型来查找相似词元。
    警告：此方法在首次运行时计算成本很高，因为它需要为生成模型的整个词汇表计算嵌入。
    """
    def __init__(self, model_name: str, generation_tokenizer, device='cuda', precompute_knn: bool = False, knn_top_k: int = 200, knn_batch_size: int = 128):
        print(f"正在加载专用嵌入模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        super().__init__(model, tokenizer, precompute_knn=precompute_knn, knn_top_k=knn_top_k, knn_batch_size=knn_batch_size)

        self.generation_tokenizer = generation_tokenizer
        self._build_embedding_cache()
        if self.precompute_knn:
            try:
                self.build_full_knn_cache(top_k=self.knn_top_k, batch_size=self.knn_batch_size)
            except Exception as e:
                print(f"预计算KNN失败（专用嵌入）: {e}")

    def _build_embedding_cache(self):
        """为生成模型的词汇表构建嵌入缓存"""
        print("正在为生成模型的词汇表构建嵌入缓存...")
        print("这可能需要一些时间...")
        vocab_size = self.generation_tokenizer.vocab_size
        all_tokens_as_text = [self.generation_tokenizer.decode([i]) for i in range(vocab_size)]
        
        self.embedding_matrix = []
        batch_size = 128  # 可根据显存大小调整
        
        for i in tqdm(range(0, vocab_size, batch_size), desc="缓存嵌入"):
            batch_texts = all_tokens_as_text[i:i+batch_size]
            
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=32).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                # 使用[CLS]嵌入或平均池化
                embeddings = outputs.last_hidden_state.mean(dim=1)
            self.embedding_matrix.append(embeddings.cpu().detach())
            
        self.embedding_matrix = torch.cat(self.embedding_matrix, dim=0).numpy()
        
        # 安全的归一化：处理零向量和NaN
        norms = np.linalg.norm(self.embedding_matrix, axis=1, keepdims=True)
        # 将零范数替换为1，避免除零错误
        norms = np.where(norms == 0, 1, norms)
        self.embedding_matrix = self.embedding_matrix / norms
        
        # 检查并处理NaN值
        nan_mask = np.isnan(self.embedding_matrix)
        if np.any(nan_mask):
            # print(f"警告: 发现 {np.sum(nan_mask)} 个NaN值，将其替换为0")
            self.embedding_matrix[nan_mask] = 0
        
        print("嵌入缓存构建并归一化完成。")

    def get_similar_tokens(self, token_id: int, top_k: int = 100) -> List[int]:
        # 若启用KNN且有预计算/按需缓存则优先返回
        if self.precompute_knn:
            pre = self._get_precomputed_neighbors(token_id, top_k)
            if pre is not None:
                return pre
        # 在缓存的矩阵中直接查找，无需重新计算
        target_vector = self.embedding_matrix[token_id:token_id+1]
        
        # 检查目标向量是否包含NaN
        if np.any(np.isnan(target_vector)):
            # print(f"警告: token_id {token_id} 的向量包含NaN，跳过相似度计算")
            return []
        
        sim_scores = cosine_similarity(target_vector, self.embedding_matrix)[0]
        
        # 检查相似度分数是否包含NaN
        if np.any(np.isnan(sim_scores)):
            # print(f"警告: token_id {token_id} 的相似度计算结果包含NaN，使用备用方案")
            # 备用方案：返回随机的top_k个token（排除自身）
            all_indices = list(range(self.embedding_matrix.shape[0]))
            all_indices.remove(token_id)
            return np.random.choice(all_indices, min(top_k, len(all_indices)), replace=False).tolist()
        
        top_indices = np.argpartition(sim_scores, -top_k-1)[-top_k-1:]
        top_indices = [idx for idx in top_indices if idx != token_id]
        top_indices_sorted = sorted(top_indices, key=lambda x: sim_scores[x], reverse=True)
        result = top_indices_sorted[:top_k]
        if self.precompute_knn:
            self._store_neighbors_map(token_id, result)
        return result

    def build_full_knn_cache(self, top_k: int, batch_size: int = 128):
        """专用嵌入矩阵上构建全量KNN缓存。
        构建每个 token 的 top_k 近邻索引，最终缓存到 knn_neighbors_full。
        """
        vocab_size = self.embedding_matrix.shape[0]
        k = min(top_k, vocab_size - 1)
        neighbors = np.zeros((vocab_size, k), dtype=np.int32)

        emb_gpu = torch.from_numpy(self.embedding_matrix).to('cuda', non_blocking=True)
        embT = emb_gpu.t().contiguous()

        with torch.no_grad():
            for start in tqdm(range(0, vocab_size, batch_size), desc="构建KNN(GPU-DED)"):
                end = min(start + batch_size, vocab_size)
                queries = emb_gpu[start:end]  # [b, d]
                sims = torch.mm(queries, embT)  # [b, V]
                # 排除自身
                ar = torch.arange(end - start, device=sims.device)
                sims[ar, start + ar] = float('-inf')
                # 取 topk
                _, top_idx = torch.topk(sims, k, dim=1)
                neighbors[start:end, :] = top_idx.cpu().numpy().astype(np.int32)
                # 释放临时，清理缓存
                del sims, top_idx, queries
                torch.cuda.empty_cache()

        # 回填缓存到 CPU 结构
        self.knn_neighbors_full = neighbors
        self.knn_k_full = k
        # 释放 GPU 大矩阵
        del emb_gpu, embT
        torch.cuda.empty_cache()

# --- 增强的贪婪量化采样器 ---

class SemanticSampler:
    """
    增强版本的贪婪量化采样器，集成了语义相似度预筛选。
    
    工作流程：
    1. 对于每一步的候选生成，先使用Embedding找到与目标词元语义相似的候选池
    2. 在这个语义候选池中，应用原始GQS的概率筛选逻辑
    3. 最终生成既语义相关又概率分布相似的候选
    """
    
    def __init__(self, tokenizer, model, embedding_provider: BaseEmbeddingProvider, 
                 temperature=1.0, max_batch_size=8, semantic_top_k=200, semantic_exclude_top_ratio: float = 0.05):
        self.tokenizer = tokenizer
        self.model = model
        self.embedding_provider = embedding_provider
        self.temperature = temperature
        self.max_batch_size = max_batch_size
        self.semantic_top_k = semantic_top_k
        # 排除“过近”相似词的比例（0-1），避免过于接近原词导致可逆推
        self.semantic_exclude_top_ratio = max(0.0, min(semantic_exclude_top_ratio, 0.9))
        self.prefix_cache = None
        self.prefix_token_ids = None
        # 允许同一轮采样中缓存多条不同前缀的 KV 缓存与首步 logits，避免反复前向
        self._prefix_cache_store = {}  # {tuple(prefix_token_ids): {"past_key_values": ..., "first_logits": torch.Tensor}}
        
        # 缓存语义相似词以避免重复计算
        self.semantic_cache = {}
        
        # print(f"增强版GQS采样器初始化完成，语义候选池大小: {semantic_top_k}")
        # print(f"模型设备: {model.device}")
        # print(f"最大批处理大小: {max_batch_size}")

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_id: List[int] | int) -> str:
        return self.tokenizer.decode(token_id, skip_special_tokens=True)
    
    def _find_suffix_tokens(self, combined_token_ids: List[int], prefix_len: int, original_suffix: str) -> List[int]:
        """
        通过逐步解码匹配来找到正确的suffix token边界
        
        Args:
            combined_token_ids: 组合编码的完整token序列
            prefix_len: 原始prefix的token长度
            original_suffix: 原始suffix文本
            
        Returns:
            正确的suffix token序列
        """
        # 从prefix_len位置开始，逐步向前尝试找到匹配的suffix起始点
        for start_pos in range(prefix_len, len(combined_token_ids)):
            suffix_tokens = combined_token_ids[start_pos:]
            
            if len(suffix_tokens) == 0:
                continue
                
            # 解码当前的suffix tokens
            decoded_suffix = self.decode(suffix_tokens).strip()
            
            # 检查是否与原始suffix匹配
            if self._suffix_matches(decoded_suffix, original_suffix):
                # print(f"  找到匹配的suffix边界，起始位置: {start_pos}")
                # print(f"  解码结果: '{decoded_suffix}' vs 原始: '{original_suffix}'")
                return suffix_tokens
        
        # 如果没找到完美匹配，使用最后一段作为fallback
        print(f"  未找到完美匹配，使用fallback方案")
        return combined_token_ids[prefix_len:]
    
    def _suffix_matches(self, decoded: str, original: str) -> bool:
        """
        检查解码后的文本是否与原始suffix匹配
        
        使用宽松的匹配策略：
        1. 去除首尾空白后完全匹配
        2. 或者解码文本是原始文本的开头（处理截断情况）
        3. 或者原始文本是解码文本的开头（处理合并情况）
        """
        decoded_clean = decoded.strip()
        original_clean = original.strip()
        
        # 完全匹配
        if decoded_clean == original_clean:
            return True
            
        # 解码文本是原始文本的开头（常见于tokenizer合并词的情况）
        if original_clean.startswith(decoded_clean) and len(decoded_clean) > 0:
            return True
            
        # 原始文本是解码文本的开头（较少见，但也可能发生）
        if decoded_clean.startswith(original_clean) and len(original_clean) > 0:
            return True
            
        return False

    def _token_has_leading_space(self, token_id: int) -> bool:
        """判断该单个 token 解码后是否以空白开头（用于避免词黏连）。"""
        s = self.decode(token_id)
        return bool(s) and s[0].isspace()

    def _cache_prefix(self, prefix_token_ids: List[int]):
        """缓存前缀的KV值，并返回下一个token的logits"""
        if self.prefix_cache is None or self.prefix_token_ids != prefix_token_ids:
            with torch.no_grad():
                input_ids = torch.tensor([prefix_token_ids], device=self.model.device)
                outputs = self.model(input_ids=input_ids, use_cache=True)
                self.prefix_cache = outputs.past_key_values
                self.prefix_token_ids = prefix_token_ids
                logits = outputs.logits.cpu()
                del outputs, input_ids
                torch.cuda.empty_cache()
                return logits
        return None

    def _cache_prefix_for(self, prefix_token_ids: List[int]) -> torch.Tensor:
        """确保以指定前缀建立并激活 KV 缓存；返回首步 logits（与 _cache_prefix 一致的含义）。

        为支持多前缀并存，这里使用 _prefix_cache_store 做按前缀粒度的缓存；
        同时将 self.prefix_cache/self.prefix_token_ids 切到该前缀，便于后续 _batch_forward_with_prefix 复用。
        """
        key = tuple(prefix_token_ids)
        if key in self._prefix_cache_store:
            store = self._prefix_cache_store[key]
            self.prefix_cache = store["past_key_values"]
            self.prefix_token_ids = list(key)
            return store["first_logits"]

        with torch.no_grad():
            input_ids = torch.tensor([prefix_token_ids], device=self.model.device)
            outputs = self.model(input_ids=input_ids, use_cache=True)
            past_kv = outputs.past_key_values
            first_logits = outputs.logits.cpu()

        self._prefix_cache_store[key] = {"past_key_values": past_kv, "first_logits": first_logits}
        self.prefix_cache = past_kv
        self.prefix_token_ids = list(key)
        # 及时释放中间变量
        del outputs, input_ids
        torch.cuda.empty_cache()
        return first_logits

    def _batch_forward_with_given_prefix(self, prefix_token_ids: List[int], suffix_sequences: List[List[int]]) -> torch.Tensor:
        """在不污染其它前缀缓存的前提下，使用指定前缀对多个后缀序列做批量前向。"""
        # 备份当前激活的缓存
        prev_cache = self.prefix_cache
        prev_ids = self.prefix_token_ids
        # 切换到目标前缀（若无则建立并缓存）
        _ = self._cache_prefix_for(prefix_token_ids)
        # 基于该前缀执行批量前向
        logits = self._batch_forward_with_prefix(suffix_sequences)
        # 还原之前的缓存环境
        self.prefix_cache = prev_cache
        self.prefix_token_ids = prev_ids
        return logits

    def _filter_too_similar(self, semantic_candidates: List[int], target_token_id: int) -> List[int]:
        """从相似候选中移除过于接近的前若干比例，并移除真实目标token。
        仅依据近邻排序截断，不依赖相似度分数，保证轻量与兼容性。
        """
        if not semantic_candidates:
            return semantic_candidates
        # 移除真实token
        filtered = [tid for tid in semantic_candidates if tid != target_token_id]
        if not filtered:
            return semantic_candidates
        # 计算需要排除的前缀数量
        exclude_n = int(len(filtered) * self.semantic_exclude_top_ratio)
        if exclude_n <= 0:
            return filtered[:self.semantic_top_k]
        if exclude_n >= len(filtered):
            # 兜底：至少保留一个候选
            return filtered[-1:]
        return filtered[exclude_n:self.semantic_top_k]

    def _expand_past_key_values(self, past_key_values, target_batch_size):
        """将past_key_values从batch_size=1扩展到target_batch_size"""
        if past_key_values is None:
            return None
        
        if hasattr(past_key_values, 'key_cache') and hasattr(past_key_values, 'value_cache'):
            from transformers.cache_utils import DynamicCache
            expanded_cache = DynamicCache()
            for i in range(len(past_key_values.key_cache)):
                key = past_key_values.key_cache[i]
                value = past_key_values.value_cache[i]
                expanded_key = key.repeat(target_batch_size, 1, 1, 1)
                expanded_value = value.repeat(target_batch_size, 1, 1, 1)
                expanded_cache.update(expanded_key, expanded_value, i)
            return expanded_cache
        else:
            expanded_past = []
            for layer_past in past_key_values:
                if layer_past is None:
                    expanded_past.append(None)
                    continue
                key, value = layer_past
                expanded_key = key.repeat(target_batch_size, 1, 1, 1)
                expanded_value = value.repeat(target_batch_size, 1, 1, 1)
                expanded_past.append((expanded_key, expanded_value))
            return tuple(expanded_past)

    def _batch_forward_with_prefix(self, suffix_sequences: List[List[int]]) -> torch.Tensor:
        """使用缓存的前缀 + 批量计算多个后缀序列"""
        if not suffix_sequences:
            return self.model(
                input_ids=torch.tensor([[]], device=self.model.device),
                past_key_values=self.prefix_cache,
                use_cache=False
            ).logits
        
        if len(suffix_sequences) <= self.max_batch_size:
            return self._single_batch_forward(suffix_sequences)
        
        all_logits = []
        for i in range(0, len(suffix_sequences), self.max_batch_size):
            batch_sequences = suffix_sequences[i:i + self.max_batch_size]
            batch_logits = self._single_batch_forward(batch_sequences)
            all_logits.append(batch_logits)
        
        return torch.cat(all_logits, dim=0)
    
    def _single_batch_forward(self, suffix_sequences: List[List[int]]) -> torch.Tensor:
        """单批次前向传播"""
        batch_size = len(suffix_sequences)
        expanded_cache = self._expand_past_key_values(self.prefix_cache, batch_size)
        
        max_len = max(len(seq) for seq in suffix_sequences)
        if max_len == 0:
            vocab_size = self.model.config.vocab_size
            dummy_logits = torch.zeros((batch_size, 1, vocab_size))
            return dummy_logits
        else:
            padded_sequences = []
            attention_masks = []
            prefix_len = len(self.prefix_token_ids) if self.prefix_token_ids else 0
            
            for seq in suffix_sequences:
                if len(seq) == 0:
                    padded = [self.tokenizer.pad_token_id] * max_len
                    mask = [1] * prefix_len + [0] * max_len
                else:
                    padded = seq + [self.tokenizer.pad_token_id] * (max_len - len(seq))
                    mask = [1] * prefix_len + [1] * len(seq) + [0] * (max_len - len(seq))
                
                padded_sequences.append(padded)
                attention_masks.append(mask)
            
            batch_input = torch.tensor(padded_sequences, device=self.model.device)
            batch_mask = torch.tensor(attention_masks, device=self.model.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch_input,
                attention_mask=batch_mask,
                past_key_values=expanded_cache,
                use_cache=False
            )
            
            logits = outputs.logits.cpu()
            del outputs, batch_input, batch_mask, expanded_cache
            torch.cuda.empty_cache()
        
        return logits

    @torch.inference_mode()
    def sample(
        self,
        gamma: int,
        prefix: str,
        suffix: str,
        num_intervals: int = 10,
        dist: str = 'rel',
        stop_tokens: List[str] | None = None,
        expansion_prefix: str | None = None,
        independent_prefix: str | None = None,
    ) -> List[Tuple[str, float]]:
        """主采样方法，集成了语义预筛选

        Args:
            gamma: 每步保留的候选数
            prefix: 已知前缀（真实前缀，计算真实后缀概率的参照）
            suffix: 目标参考后缀（用于概率对齐）
            num_intervals: 概率量化区间数
            dist: 'rel' 或 'abs'，控制概率筛选方式
            stop_tokens: 当解码后的 token 含有这些子串时，认为序列结束。
                默认仅使用换行符 ("\n")。若想保留句点等符号，请不要把 '.' 放进此列表。
            expansion_prefix: 扩展前缀（可选）。若提供：
                - 用 prefix 计算真实后缀每一步的目标概率 q
                - 用 expansion_prefix 计算候选分布 p，并以 p/q 做相对筛选与权重
                若不提供，与原先逻辑保持一致（即使用 prefix 同时作为候选分布前缀）
            independent_prefix: 独立前缀（可选）。当在语义候选内找不到概率相近的候选时，
                回退到使用 independent_prefix 计算候选分布 p，再与真实 q 做对齐，
                该前缀通常为“所有位置都是占位符”的提示，以获得更开放的分布。
        """

        if stop_tokens is None:
            stop_tokens = ["\n"]  # 只按换行截断，保留'.'等正常标点

        # 清除激活缓存，初始化多前缀缓存存储（本次调用作用域）
        self.prefix_cache = None
        self.prefix_token_ids = None
        self._prefix_cache_store = {}
        
        prefix = strip_special_tokens(self.tokenizer, prefix)
        prefix_token_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        # 扩展前缀：若未提供，则退化为与真实前缀相同
        if expansion_prefix is not None:
            expansion_prefix_clean = strip_special_tokens(self.tokenizer, expansion_prefix)
            expansion_prefix_token_ids = self.tokenizer.encode(expansion_prefix_clean, add_special_tokens=False)
        else:
            expansion_prefix_token_ids = list(prefix_token_ids)
        # 独立前缀（仅在需要回退时使用）
        if independent_prefix is not None:
            independent_prefix_clean = strip_special_tokens(self.tokenizer, independent_prefix)
            independent_prefix_token_ids = self.tokenizer.encode(independent_prefix_clean, add_special_tokens=False)
        else:
            independent_prefix_token_ids = None
        suffix_token_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        
        # 验证编码并修正suffix_token_ids
        combined_token_ids = self.encode(prefix + " " + suffix)
        if len(combined_token_ids) != len(prefix_token_ids) + len(suffix_token_ids):
            # print("警告：空格编码影响了token分割，使用组合编码方式")
            # 使用简单的匹配方法找到正确的suffix边界
            suffix_token_ids = self._find_suffix_tokens(combined_token_ids, len(prefix_token_ids), suffix)
        # 兜底：若suffix经编码后为空，则直接返回参考样本，避免空结果
        if len(suffix_token_ids) == 0:
            return [(suffix, 1.0)]
        
        # 初始化
        ref_candidate = Candidate()
        candidates = []
        done_list = []
        max_token_len = len(suffix_token_ids)
        
        for i in range(max_token_len):
            # 构建“扩展前缀”下的序列列表（用于候选分布）
            all_sequences = []
            if ref_candidate is not None:
                candidates = [ref_candidate] + candidates
            for candidate in candidates:
                all_sequences.append(candidate.tokens)

            # 1) 真实前缀：计算目标概率 q（用于相对权重计算）
            if ref_candidate is not None:
                if i == 0:
                    real_batch_logits = self._cache_prefix_for(prefix_token_ids)
                    target_logits = real_batch_logits[0]
                else:
                    real_batch_logits = self._batch_forward_with_given_prefix(prefix_token_ids, [ref_candidate.tokens])
                    target_logits = real_batch_logits[0]
                logits_float_real = target_logits[-1].float() / self.temperature
                target_probs = torch.softmax(logits_float_real, dim=-1)
                tgt_token_prob = target_probs[suffix_token_ids[i]].item()
            else:
                tgt_token_prob = None

            # 2) 扩展前缀：获取候选分布 batch（对每个候选独立）
            if i == 0:
                exp_batch_logits = self._cache_prefix_for(expansion_prefix_token_ids)
            else:
                exp_batch_logits = self._batch_forward_with_given_prefix(expansion_prefix_token_ids, all_sequences)
            indep_batch_logits = None  # 延迟构建，仅在回退时使用

            # 处理所有候选
            new_candidates = []

            for j, candidate in enumerate(candidates):
                batch_idx = j
                candidate_logits = exp_batch_logits[batch_idx]

                last_pos = len(candidate.tokens) - 1
                logits_float = candidate_logits[last_pos].float() / self.temperature
                candidate_probs = torch.softmax(logits_float, dim=-1)
                
                # =====【核心改进】语义预筛选 + 概率筛选=====
                
                # 1. 语义预筛选：获取与目标词元语义相似的候选池（使用缓存）
                target_token_id = suffix_token_ids[i]
                if target_token_id not in self.semantic_cache:
                    self.semantic_cache[target_token_id] = self.embedding_provider.get_similar_tokens(
                        target_token_id, top_k=self.semantic_top_k
                    )
                # 排除“过近”的前若干比例相似词，并移除真实token
                semantic_candidates = self._filter_too_similar(self.semantic_cache[target_token_id], target_token_id)
                # 约束：与目标 token 的前导空格属性一致，避免词黏连
                need_space = self._token_has_leading_space(target_token_id)
                semantic_candidates = [tid for tid in semantic_candidates if self._token_has_leading_space(tid) == need_space]
                if not semantic_candidates:
                    semantic_candidates = [target_token_id]
                
                # 确保目标词元也在候选池中
                # 仍确保候选池非空；必要时回填真实token（低概率使用）
                if not semantic_candidates:
                    semantic_candidates = [target_token_id]
                
                # 2. 概率筛选：在语义候选池中应用GQS的概率筛选逻辑
                semantic_candidate_probs = candidate_probs[semantic_candidates]
                
                if dist == 'abs':
                    relative_qualifying_indices = filter_probs_abs(semantic_candidate_probs, tgt_token_prob, num_intervals)
                elif dist == 'rel':
                    relative_qualifying_indices = filter_probs_rel(semantic_candidate_probs, tgt_token_prob, magnitude=1/num_intervals)
                else:
                    raise ValueError(f"Invalid dist: {dist}")
                
                # 转换回原始词汇表索引
                qualifying_token_ids = [semantic_candidates[idx] for idx in relative_qualifying_indices]

                # 若语义候选内无概率相近结果，回退到独立前缀分布（若提供）
                if len(qualifying_token_ids) == 0 and independent_prefix_token_ids is not None:
                    if indep_batch_logits is None:
                        if i == 0:
                            indep_batch_logits = self._cache_prefix_for(independent_prefix_token_ids)
                        else:
                            indep_batch_logits = self._batch_forward_with_given_prefix(independent_prefix_token_ids, all_sequences)
                    fallback_candidate_logits = indep_batch_logits[batch_idx]
                    logits_float_fb = fallback_candidate_logits[last_pos].float() / self.temperature
                    candidate_probs_fb = torch.softmax(logits_float_fb, dim=-1)
                    semantic_candidate_probs_fb = candidate_probs_fb[semantic_candidates]
                    if dist == 'abs':
                        relative_fb = filter_probs_abs(semantic_candidate_probs_fb, tgt_token_prob, num_intervals)
                    elif dist == 'rel':
                        relative_fb = filter_probs_rel(semantic_candidate_probs_fb, tgt_token_prob, magnitude=1/num_intervals)
                    else:
                        raise ValueError(f"Invalid dist: {dist}")
                    qualifying_token_ids = [semantic_candidates[idx] for idx in relative_fb]
                # 二次回退：若仍为空，则从语义候选中“较近的一批”（前64个）作为候选池
                if len(qualifying_token_ids) == 0:
                    pool_n = min(64, len(semantic_candidates))
                    if pool_n > 0:
                        qualifying_token_ids = list(semantic_candidates[:pool_n])
                    else:
                        qualifying_token_ids = [int(target_token_id)]
                
                # ================================================
                
                gamma_ = min(gamma, len(qualifying_token_ids))
                
                if gamma_ == 0:
                    continue

                selected_next_token_ids = np.random.choice(
                    qualifying_token_ids, 
                    min(gamma_, len(qualifying_token_ids)),
                    replace=False
                )
                
                selected_token_probs = candidate_probs[selected_next_token_ids].numpy()
                
                # 为每个选中的token创建新候选
                for k in range(gamma_):
                    token_id = selected_next_token_ids[k]
                    token_prob = selected_token_probs[k]
                    
                    next_candidate = candidate.clone()
                    # 相对权重：计算相对于目标token的概率比值
                    relative_weight = token_prob / tgt_token_prob if tgt_token_prob > 0 else 1.0
                    next_candidate.prob *= relative_weight
                    
                    # 检查终止条件
                    dec = self.decode(token_id)
                    if any(st in dec for st in stop_tokens) or token_id == self.tokenizer.eos_token_id:
                        done_list.append(next_candidate)
                        continue
                    
                    next_candidate.tokens.append(token_id)
                    
                    if i + 1 == max_token_len:
                        done_list.append(next_candidate)
                        continue
                    
                    new_candidates.append(next_candidate)
            
            # 处理参考候选的下一步
            if ref_candidate is not None:
                # 参考候选的相对权重始终为1.0（选择目标token）
                ref_candidate.prob *= 1.0
                
                if i + 1 < len(suffix_token_ids):
                    ref_candidate.tokens.append(suffix_token_ids[i])
                else:
                    ref_candidate.tokens.append(suffix_token_ids[i])
                    done_list.append(ref_candidate.clone())
                    ref_candidate = None
            
            # 随机选择gamma个候选继续
            candidates = random.sample(new_candidates, min(gamma, len(new_candidates)))
        
        # 后处理：去重和排序
        done_map = {}
        cleaned = []
        ref_prob = 1.0
        
        for candidate in done_list:
            text = self.decode(candidate.tokens)
            text = ''.join(c for c in text if c.isprintable())
            
            if text.strip() == suffix.strip():
                ref_prob = candidate.prob
            
            key = re.sub(r"[^a-z0-9]+", "", text, flags=re.IGNORECASE)
            if key not in done_map:
                done_map[key] = True
                cleaned.append((text, candidate.prob))
        
        cleaned = sorted(cleaned, key=lambda x: abs(x[1] - ref_prob))
        return cleaned[:gamma]

    @torch.inference_mode()
    def sample_multi(
        self,
        gamma: int,
        prefix: str,
        suffix: str,
        expansion_prefixes: List[str],
        num_intervals: int = 10,
        dist: str = 'rel',
        stop_tokens: List[str] | None = None,
        independent_prefix: str | None = None,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """批量采样：在同一次调用中对多个扩展前缀完成候选分布计算与筛选。

        逻辑：
        - 使用真实前缀 prefix 计算每一步真实后缀 token 的概率 q（共享一次前向/每步一次）。
        - 对每个扩展前缀 expansion_prefixes[i]，独立进行候选分布计算与筛选，生成与 sample 相同语义的结果列表。

        Returns:
            {path_idx: [(text, relative_prob), ...]} 与单路径 sample 返回格式一致
        """

        if stop_tokens is None:
            stop_tokens = ["\n"]

        # 清除激活缓存，初始化多前缀缓存存储（本次调用作用域）
        self.prefix_cache = None
        self.prefix_token_ids = None
        self._prefix_cache_store = {}

        # 规范化真实前缀与所有扩展前缀
        prefix = strip_special_tokens(self.tokenizer, prefix)
        prefix_token_ids = self.tokenizer.encode(prefix, add_special_tokens=False)

        # 独立前缀（用于回退）
        if independent_prefix is not None:
            independent_prefix_clean = strip_special_tokens(self.tokenizer, independent_prefix)
            independent_prefix_token_ids = self.tokenizer.encode(independent_prefix_clean, add_special_tokens=False)
        else:
            independent_prefix_token_ids = None

        cleaned_expansion_token_ids: List[List[int]] = []
        for ep in expansion_prefixes:
            ep_clean = strip_special_tokens(self.tokenizer, ep) if ep is not None else prefix
            cleaned_expansion_token_ids.append(self.tokenizer.encode(ep_clean, add_special_tokens=False))

        # 规范化 suffix，并做一次边界修正（与单路径一致）
        suffix_token_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        combined_token_ids = self.encode(prefix + " " + suffix)
        if len(combined_token_ids) != len(prefix_token_ids) + len(suffix_token_ids):
            suffix_token_ids = self._find_suffix_tokens(combined_token_ids, len(prefix_token_ids), suffix)
        # 兜底：若suffix经编码后为空，则为所有路径直接返回参考样本
        if len(suffix_token_ids) == 0:
            return {idx: [(suffix, 1.0)] for idx in range(len(expansion_prefixes))}

        max_token_len = len(suffix_token_ids)

        # 1) 真实前缀：预先计算每一步的目标概率 q_i（共享给所有路径）
        tgt_probs_per_step: List[float] = []
        ref_tokens: List[int] = []
        for i in range(max_token_len):
            if i == 0:
                real_batch_logits = self._cache_prefix_for(prefix_token_ids)
                target_logits = real_batch_logits[0]
            else:
                real_batch_logits = self._batch_forward_with_given_prefix(prefix_token_ids, [ref_tokens])
                target_logits = real_batch_logits[0]

            logits_float_real = target_logits[-1].float() / self.temperature
            target_probs = torch.softmax(logits_float_real, dim=-1)
            tgt_token_prob = target_probs[suffix_token_ids[i]].item()
            tgt_probs_per_step.append(tgt_token_prob)

            # 推进真实参考序列
            ref_tokens.append(suffix_token_ids[i])

        # 2) 对每个扩展前缀，复制单路径采样逻辑（但复用共享的 tgt_probs_per_step）
        results: Dict[int, List[Tuple[str, float]]] = {}
        for path_idx, expansion_prefix_token_ids in enumerate(cleaned_expansion_token_ids):
            # 初始化 per-path 状态
            candidates: List[Candidate] = []
            ref_candidate = Candidate()
            done_list: List[Candidate] = []

            # 单独的参考序列（用于复用 sample 的行为，将参考样本也并入结果集）
            ref_tokens_local: List[int] = []

            for i in range(max_token_len):
                # 组装当前路径的序列池（包含参考候选在前）
                all_sequences: List[List[int]] = []
                if ref_candidate is not None:
                    candidates = [ref_candidate] + candidates
                for cand in candidates:
                    all_sequences.append(cand.tokens)

                # 扩展前缀：获取候选分布 batch（对每个候选独立）
                if i == 0:
                    exp_batch_logits = self._cache_prefix_for(expansion_prefix_token_ids)
                else:
                    exp_batch_logits = self._batch_forward_with_given_prefix(expansion_prefix_token_ids, all_sequences)
                indep_batch_logits = None  # 延迟构建，按需创建

                # 处理所有候选
                new_candidates: List[Candidate] = []
                tgt_token_prob = tgt_probs_per_step[i]

                for j, candidate in enumerate(candidates):
                    batch_idx = j
                    candidate_logits = exp_batch_logits[batch_idx]

                    last_pos = len(candidate.tokens) - 1
                    logits_float = candidate_logits[last_pos].float() / self.temperature
                    candidate_probs = torch.softmax(logits_float, dim=-1)

                    # ===== 语义预筛选 + 概率筛选（与单路径一致） =====
                    target_token_id = suffix_token_ids[i]
                    if target_token_id not in self.semantic_cache:
                        self.semantic_cache[target_token_id] = self.embedding_provider.get_similar_tokens(
                            target_token_id, top_k=self.semantic_top_k
                        )
                    # 排除“过近”的前若干比例相似词，并移除真实token
                    semantic_candidates = self._filter_too_similar(self.semantic_cache[target_token_id], target_token_id)
                    # 约束：与目标 token 的前导空格属性一致，避免词黏连
                    need_space = self._token_has_leading_space(target_token_id)
                    semantic_candidates = [tid for tid in semantic_candidates if self._token_has_leading_space(tid) == need_space]
                    if not semantic_candidates:
                        semantic_candidates = [target_token_id]

                    if not semantic_candidates:
                        semantic_candidates = [target_token_id]

                    semantic_candidate_probs = candidate_probs[semantic_candidates]

                    if dist == 'abs':
                        relative_qualifying_indices = filter_probs_abs(semantic_candidate_probs, tgt_token_prob, num_intervals)
                    elif dist == 'rel':
                        relative_qualifying_indices = filter_probs_rel(semantic_candidate_probs, tgt_token_prob, magnitude=1/num_intervals)
                    else:
                        raise ValueError(f"Invalid dist: {dist}")

                    qualifying_token_ids = [semantic_candidates[idx] for idx in relative_qualifying_indices]

                    # 若语义候选内无概率相近结果，回退到独立前缀分布（若提供）
                    if len(qualifying_token_ids) == 0 and independent_prefix_token_ids is not None:
                        if indep_batch_logits is None:
                            if i == 0:
                                indep_batch_logits = self._cache_prefix_for(independent_prefix_token_ids)
                            else:
                                indep_batch_logits = self._batch_forward_with_given_prefix(independent_prefix_token_ids, all_sequences)
                        fallback_candidate_logits = indep_batch_logits[batch_idx]
                        logits_float_fb = fallback_candidate_logits[last_pos].float() / self.temperature
                        candidate_probs_fb = torch.softmax(logits_float_fb, dim=-1)
                        semantic_candidate_probs_fb = candidate_probs_fb[semantic_candidates]
                        if dist == 'abs':
                            relative_fb = filter_probs_abs(semantic_candidate_probs_fb, tgt_token_prob, num_intervals)
                        elif dist == 'rel':
                            relative_fb = filter_probs_rel(semantic_candidate_probs_fb, tgt_token_prob, magnitude=1/num_intervals)
                        else:
                            raise ValueError(f"Invalid dist: {dist}")
                        qualifying_token_ids = [semantic_candidates[idx] for idx in relative_fb]
                    # 二次回退：仍为空则用语义候选前64个作为候选池
                    if len(qualifying_token_ids) == 0:
                        pool_n = min(64, len(semantic_candidates))
                        if pool_n > 0:
                            qualifying_token_ids = list(semantic_candidates[:pool_n])
                        else:
                            qualifying_token_ids = [int(target_token_id)]

                    gamma_ = min(gamma, len(qualifying_token_ids))
                    if gamma_ == 0:
                        continue

                    selected_next_token_ids = np.random.choice(
                        qualifying_token_ids,
                        min(gamma_, len(qualifying_token_ids)),
                        replace=False
                    )

                    selected_token_probs = candidate_probs[selected_next_token_ids].numpy()

                    for k in range(gamma_):
                        token_id = selected_next_token_ids[k]
                        token_prob = selected_token_probs[k]

                        next_candidate = candidate.clone()
                        relative_weight = token_prob / tgt_token_prob if tgt_token_prob > 0 else 1.0
                        next_candidate.prob *= relative_weight

                        dec = self.decode(token_id)
                        if any(st in dec for st in stop_tokens) or token_id == self.tokenizer.eos_token_id:
                            done_list.append(next_candidate)
                            continue

                        next_candidate.tokens.append(token_id)

                        if i + 1 == max_token_len:
                            done_list.append(next_candidate)
                            continue

                        new_candidates.append(next_candidate)

                # 参考候选前进一位并保留（与单路径一致）
                if ref_candidate is not None:
                    ref_candidate.prob *= 1.0
                    if i + 1 < len(suffix_token_ids):
                        ref_candidate.tokens.append(suffix_token_ids[i])
                        ref_tokens_local.append(suffix_token_ids[i])
                    else:
                        ref_candidate.tokens.append(suffix_token_ids[i])
                        ref_tokens_local.append(suffix_token_ids[i])
                        done_list.append(ref_candidate.clone())
                        ref_candidate = None

                # 下一轮候选池
                candidates = random.sample(new_candidates, min(gamma, len(new_candidates)))

            # 去重与排序（与单路径一致）
            done_map: Dict[str, bool] = {}
            cleaned: List[Tuple[str, float]] = []
            ref_prob = 1.0
            for cand in done_list:
                text = self.decode(cand.tokens)
                text = ''.join(c for c in text if c.isprintable())
                if text.strip() == suffix.strip():
                    ref_prob = cand.prob
                key = re.sub(r"[^a-z0-9]+", "", text, flags=re.IGNORECASE)
                if key not in done_map:
                    done_map[key] = True
                    cleaned.append((text, cand.prob))

            cleaned = sorted(cleaned, key=lambda x: abs(x[1] - ref_prob))
            results[path_idx] = cleaned[:gamma]

        return results

def strip_special_tokens(tokenizer, text: str) -> str:
    """去除文本中的特殊tokens"""
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False), skip_special_tokens=True)

def main():
    """测试示例"""
    import time
    
    print("=== 增强版贪婪量化采样器测试 ===")
    
    # --- 配置区域 ---
    GENERATION_MODEL_NAME = "NousResearch/Llama-2-7b-hf"
    
    # 设置专用嵌入模型的名称，设置为 None 则使用生成模型自身的嵌入（推荐）
    DEDICATED_EMBEDDING_MODEL_NAME = None
    # DEDICATED_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    
    # --- 加载模型 ---
    print(f"加载模型: {GENERATION_MODEL_NAME}")
    
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME, device_map="cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME, device_map="cuda", torch_dtype=torch.float16)
    model.eval()
    
    # --- 初始化嵌入服务提供者 ---
    if DEDICATED_EMBEDDING_MODEL_NAME:
        print(f"使用【专用】嵌入模型: {DEDICATED_EMBEDDING_MODEL_NAME}")
        embedding_provider = DedicatedEmbeddingModelProvider(
            model_name=DEDICATED_EMBEDDING_MODEL_NAME,
            generation_tokenizer=tokenizer,
            device=model.device
        )
    else:
        print("使用【生成模型自身】的嵌入。")
        embedding_provider = GenerativeModelEmbeddingProvider(model=model, tokenizer=tokenizer)
    
    # --- 创建增强采样器 ---
    sampler = SemanticSampler(
        tokenizer=tokenizer, 
        model=model, 
        embedding_provider=embedding_provider,
        temperature=1.0, 
        max_batch_size=8,
        semantic_top_k=100  # 语义候选池大小
    )
    
    # 测试案例
    test_cases = [
        {
            "prefix": f"""Your task is to guess the redacted [TARGET], which is word, in the following sentence:\n\n
            <sentence>Hey [TARGET], did you hear about the new government initiative to help people with health insurance premiums? I saw it on Twitter yesterday, but wasn't sure what it actually means for us.\n\nOh yeah, I caught that too! The government just announced a subsidy program to lower monthly payments for people who buy insurance through the marketplace. It's supposed to start next month. They posted some details on their official Facebook page, so it sounds legit.\n\n
            One possible word that fits in to [TARGET] is:""",
            "suffix": "Sarah",
            "gamma": 32,
            "num_intervals": 10,
            "dist": "rel",
        },
        {
            "prefix": f"""Your task is to guess the redacted [TARGET], which is word, in the following sentence:\n\n
            <sentence>Once upon a time there was a [TARGET]</sentence>\n\n
            One possible word that fits in to [TARGET] is:""",
            "suffix": "beautiful princess",
            "gamma": 32,
            "num_intervals": 10,
            "dist": "abs",
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- 测试案例 {i} ---")
        print(f"前缀: '{test_case['prefix'][:100]}...'")
        print(f"目标: '{test_case['suffix']}'")
        print(f"参数: gamma={test_case['gamma']}, intervals={test_case['num_intervals']}, dist={test_case['dist']}")
        
        start_time = time.time()
        
        results = sampler.sample(
            gamma=test_case['gamma'],
            prefix=test_case['prefix'],
            suffix=test_case['suffix'],
            num_intervals=test_case['num_intervals'],
            dist=test_case['dist'],
        )
        
        elapsed = time.time() - start_time
        print(f"生成时间: {elapsed:.2f}秒")
        print(f"生成结果 ({len(results)}个):")
        
        for j, (text, prob) in enumerate(results, 1):
            print(f"  {j}. '{text}' (概率权重: {prob:.4f})")
                
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()