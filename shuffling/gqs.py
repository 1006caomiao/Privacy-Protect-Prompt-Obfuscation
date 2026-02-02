from __future__ import annotations
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "/data1/huggingface"
import huggingface_hub 
huggingface_hub.login("")
import math
import random
import re
import copy
from typing import List, Tuple, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass


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


class GreedyQuantizedSampler:
    """
    贪婪量化采样器
    """
    
    def __init__(self, tokenizer, model, temperature=1.0, max_batch_size=8):
        self.tokenizer = tokenizer
        self.model = model
        self.temperature = temperature
        self.max_batch_size = max_batch_size  # 最大批次大小，防止显存爆炸
        self.prefix_cache = None
        self.prefix_token_ids = None

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, token_id: List[int] | int) -> str:
        return self.tokenizer.decode(token_id, skip_special_tokens=True)

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

    def _suffix_matches(self, decoded: str, original: str) -> bool:
        """
        宽松匹配规则以稳健定位 suffix：
        1) 去除首尾空白后完全匹配
        2) 解码文本是原始文本的开头（处理 tokenizer 合并/截断）
        3) 原始文本是解码文本的开头
        """
        decoded_clean = decoded.strip()
        original_clean = original.strip()

        if decoded_clean == original_clean:
            return True
        if original_clean.startswith(decoded_clean) and len(decoded_clean) > 0:
            return True
        if decoded_clean.startswith(original_clean) and len(original_clean) > 0:
            return True
        return False

    def _find_suffix_tokens(self, combined_token_ids: List[int], prefix_len: int, original_suffix: str) -> List[int]:
        """
        通过逐步解码匹配来找到正确的 suffix token 边界。
        若找不到完美匹配，使用从 prefix_len 起的剩余部分作为 fallback。
        """
        for start_pos in range(prefix_len, len(combined_token_ids)):
            suffix_tokens = combined_token_ids[start_pos:]
            if len(suffix_tokens) == 0:
                continue
            decoded_suffix = self.decode(suffix_tokens).strip()
            if self._suffix_matches(decoded_suffix, original_suffix):
                return suffix_tokens
        return combined_token_ids[prefix_len:]

    @torch.inference_mode()
    def sample(self, gamma: int, prefix: str, suffix: str, num_intervals=10, dist: str = 'rel', 
               exclude_reference: bool = True) -> List[Tuple[str, float]]:
        """
        主采样方法
        
        Args:
            gamma: 每步生成的候选数量
            prefix: 前缀文本
            suffix: 目标后缀文本  
            num_intervals: 概率量化区间数
            dist: 距离度量方式 ('abs' 或 'rel')
            exclude_reference: 是否排除参考样本 (True=排除, False=放在第一位)
        
        Returns:
            生成的候选列表 [(text, prob), ...]
        """
        # 清除前缀缓存，确保每次采样都重新计算
        self.prefix_cache = None
        self.prefix_token_ids = None

        # 编码输入
        prefix = strip_special_tokens(self.tokenizer, prefix)
        
        # 分别编码前缀/后缀，并使用组合编码+逐步解码匹配来稳健确定 suffix 的 token 边界
        prefix_token_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        # 先计算组合编码（前缀 + 空格 + 后缀），再用宽松匹配定位 suffix tokens
        combined_token_ids = self.encode(prefix + " " + suffix)
        suffix_token_ids = self._find_suffix_tokens(
            combined_token_ids=combined_token_ids,
            prefix_len=len(prefix_token_ids),
            original_suffix=suffix,
        )
        # 兜底：若 suffix 经编码后为空，则直接返回参考样本，避免空结果
        if len(suffix_token_ids) == 0:
            return [(suffix, 1.0)]
        
        # 初始化：参考候选和普通候选
        ref_candidate = Candidate()  # 参考路径
        candidates = []   # 其他候选（初始为空）
        done_list = []
        max_token_len = len(suffix_token_ids)
        
        for i in range(max_token_len):
            # 构建序列列表：目标序列 + 候选序列
            all_sequences = []
            # 构建候选列表（参考候选+普通候选）
            if ref_candidate is not None:
                candidates = [ref_candidate] + candidates
            # 添加候选序列到批量推理
            for candidate in candidates:
                all_sequences.append(candidate.tokens)
            
            # 计算目标概率
            tgt_token_prob = None
            if ref_candidate is not None:
                if i == 0:
                    # 第一轮：缓存前缀
                    batch_logits = self._cache_prefix(prefix_token_ids)
                else:
                    # 后续轮次：批量推理
                    batch_logits = self._batch_forward_with_prefix(all_sequences)
                target_logits = batch_logits[0]
                # 转换为float32进行softmax计算，避免Half精度问题
                logits_float = target_logits[-1].float() / self.temperature
                target_probs = torch.softmax(logits_float, dim=-1)
                tgt_token_prob = target_probs[suffix_token_ids[i]].item()
                ref_candidate.prob *= tgt_token_prob
            
            # 处理所有候选（包括参考候选）
            new_candidates = []
            
            for j, candidate in enumerate(candidates):
                batch_idx = j
                candidate_logits = batch_logits[batch_idx]
                
                # 获取该候选的下一token概率分布
                last_pos = len(candidate.tokens) - 1
                # 转换为float32进行softmax计算，避免Half精度问题
                logits_float = candidate_logits[last_pos].float() / self.temperature
                candidate_probs = torch.softmax(logits_float, dim=-1)
                
                # 过滤相似概率的tokens
                if dist == 'abs':
                    qualifying_token_ids = filter_probs_abs(candidate_probs, tgt_token_prob, num_intervals)
                elif dist == 'rel':
                    qualifying_token_ids = filter_probs_rel(candidate_probs, tgt_token_prob, magnitude=1/num_intervals)
                else:
                    raise ValueError(f"Invalid dist: {dist}")
                
                gamma_ = min(gamma, len(qualifying_token_ids))
                
                if gamma_ == 0:
                    continue
                
                # 随机选择gamma个tokens
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
                    next_candidate.prob *= token_prob / tgt_token_prob  # 概率权重调整
                    
                    # 检查终止条件
                    dec = self.decode(token_id)
                    if '.' in dec or '\n' in dec or token_id == self.tokenizer.eos_token_id:
                        done_list.append(next_candidate)
                        continue
                    
                    # 添加token到候选
                    next_candidate.tokens.append(token_id)
                    
                    # 检查是否达到最大长度
                    if i + 1 == max_token_len:
                        done_list.append(next_candidate)
                        continue
                    
                    new_candidates.append(next_candidate)
            
            # 处理参考候选的下一步（与原版逻辑一致）
            if ref_candidate is not None:
                if i + 1 < len(suffix_token_ids):
                    # 为参考候选添加下一个真实token
                    ref_candidate.tokens.append(suffix_token_ids[i])
                else:
                    # 参考序列完成，添加最后一个token并加入done_list
                    ref_candidate.tokens.append(suffix_token_ids[i])
                    done_list.append(ref_candidate.clone())
                    ref_candidate = None  # 关键：设置为None，与原版一致
            
            # 随机选择gamma个候选继续
            candidates = random.sample(new_candidates, min(gamma, len(new_candidates)))
        
        # 后处理：去重和排序
        done_map = {}
        cleaned = []
        ref_prob = 1.0
        ref_text = suffix.strip()  # 参考文本
        ref_sample = None  # 存储参考样本
        
        for candidate in done_list:
            text = self.decode(candidate.tokens)
            text = ''.join(c for c in text if c.isprintable())
            
            # 处理参考样本
            if text.strip() == ref_text:
                ref_prob = candidate.prob
                if not exclude_reference:
                    ref_sample = (text, candidate.prob)
                continue  # 参考样本单独处理，不参与普通去重
            
            # 去重
            key = re.sub(r"[^a-z0-9]+", "", text, flags=re.IGNORECASE)
            if key not in done_map:
                done_map[key] = True
                cleaned.append((text, candidate.prob))
        
        # 按与参考概率的距离排序
        cleaned = sorted(cleaned, key=lambda x: abs(x[1] - ref_prob))

        # 处理参考样本的位置
        if not exclude_reference and ref_sample is not None:
            # 将参考样本放在第一位，保持伪样本数量为gamma
            final_candidates = [ref_sample] + cleaned
            target_count = gamma + 1  # 额外的1个是参考样本
        else:
            # 排除参考样本，只要伪样本
            final_candidates = cleaned
            target_count = gamma

        # 保证结果数量达到target_count。若为空，用参考候选补齐，避免死循环
        if len(final_candidates) < target_count:
            if len(final_candidates) == 0:
                # 使用参考候选进行补齐；若 ref_sample 不存在，则使用 ref_text/ref_prob 兜底
                if not exclude_reference and ref_sample is not None:
                    seed = [ref_sample]
                else:
                    seed = [(ref_text, ref_prob)]
                repeats = (target_count + len(seed) - 1) // len(seed)
                final_candidates = (seed * repeats)[:target_count]
            else:
                extended = list(final_candidates)
                while len(extended) < target_count:
                    remaining = target_count - len(extended)
                    if not exclude_reference and ref_sample is not None:
                        # 有参考样本时，只复制伪样本部分
                        available_fakes = cleaned if len(cleaned) > 0 else [ref_sample]
                        extended.extend(available_fakes[:remaining])
                    else:
                        extended.extend(final_candidates[:remaining])
                final_candidates = extended[:target_count]
        else:
            final_candidates = final_candidates[:target_count]

        return final_candidates

    # -------------------------------------------------------------
    # 连续采样接口：根据目标数量自动冗余采样并截断/补足
    # -------------------------------------------------------------
    def generate_samples(self, prefix: str, suffix: str, target_samples: int = 100,
                         num_intervals: int = 10, dist: str = 'rel',
                         redundancy_factor: float = 1.3, exclude_reference: bool = True) -> List[Tuple[str, float]]:
        """连续采样包装函数，保证返回的样本数量精确等于target_samples。

        Args:
            prefix: 带[TARGET]占位符的前缀文本
            suffix: 真实目标词（参考路径）
            target_samples: 需要的候选数量
            num_intervals: 概率量化区间
            dist: 距离度量方式 ('abs' | 'rel')
            redundancy_factor: 冗余因子，实际采样数量 = target_samples * redundancy_factor
            exclude_reference: 是否排除参考样本 (True=排除, False=放在第一位)
        """

        redundant_target = max(target_samples, int(target_samples * redundancy_factor))

        # 先做一次冗余采样
        candidates = self.sample(
            gamma=redundant_target,
            prefix=prefix,
            suffix=suffix,
            num_intervals=num_intervals,
            dist=dist,
            exclude_reference=exclude_reference
        )

        # 如果采样结果不足，再次简单复制（理论上不会发生）
        if len(candidates) < target_samples:
            extended = list(candidates)
            while len(extended) < target_samples:
                remaining = target_samples - len(extended)
                extended.extend(candidates[:remaining])
            candidates = extended

        # 最终截断到目标数量
        return candidates[:target_samples]


def strip_special_tokens(tokenizer, text: str) -> str:
    """去除文本中的特殊tokens"""
    return tokenizer.decode(tokenizer.encode(text, add_special_tokens=False), skip_special_tokens=True)



# =============================================================
# NER解析与多位置连续采样（独立策略）
# =============================================================


@dataclass
class NERSensitiveWord:
    """NER解析得到的敏感词信息"""
    word: str           # 文本
    word_type: str      # 类型 (NAME, PLACE ...)
    positions: List[int]  # 该词在模板中的占位符顺序位置列表
    uid: str            # 唯一id = word_type_word


def parse_ner_text(ner_text: str) -> Tuple[str, List[NERSensitiveWord]]:
    """解析<redacted> word (type) </redacted>格式文本，返回模板和敏感词列表"""
    pattern = r'<redacted>\s*([^(]+?)\s*\(([^)]+)\)\s*</redacted>'

    sensitive_words: List[NERSensitiveWord] = []
    word_to_positions: Dict[str, List[int]] = {}
    position = 0

    for match in re.finditer(pattern, ner_text):
        word = match.group(1).strip()
        wtype = match.group(2).strip()
        uid = f"{word}_{wtype}"
        word_to_positions.setdefault(uid, []).append(position)
        position += 1

    for uid, pos_list in word_to_positions.items():
        word, wtype = uid.rsplit('_', 1)
        sensitive_words.append(NERSensitiveWord(
            word=word,
            word_type=wtype,
            positions=pos_list,
            uid=uid
        ))

    # 将所有敏感词替换成占位符
    template = re.sub(pattern, '<redacted>', ner_text)
    template = re.sub(r'\s+', ' ', template).strip()

    return template, sensitive_words


def build_independent_prefix(template: str) -> str:
    """独立策略下直接返回模板（所有占位符保留）"""
    return template


def postprocess_format_consistency(
    ner_text: str,
    samples: List[Tuple[str, float]],
    real_to_fake_mapping: Dict[str, List[str]],
    real_words_sequence: List[str]
) -> List[Tuple[str, float]]:
    """
    后处理：确保所有生成的样本格式与原始NER文本一致
    这个方法会：
    1. 从原始NER文本出发，按顺序替换<redacted>标签
    2. 保留原始文本的所有格式特征（空格、换行、符号等）
    3. 只压缩连续的空格（2个或以上）
    """
    PLACEHOLDER_PAT = re.compile(r'<redacted>.*?</redacted>', re.DOTALL)

    reformatted_samples: List[Tuple[str, float]] = []

    for sample_idx, (_old_text, prob) in enumerate(samples):
        # 使用原始NER文本重建
        result = ner_text

        # 按顺序替换每个<redacted>标签
        for real_word in real_words_sequence:
            fake_list = real_to_fake_mapping.get(real_word, [])

            if not fake_list:
                # 如果没有映射，使用原词
                replacement = real_word
            else:
                # 使用循环索引选择伪词
                replacement = fake_list[sample_idx % len(fake_list)]

            # 替换第一个<redacted>标签
            # 使用lambda避免反斜杠等特殊字符被正则表达式解释
            result, _ = PLACEHOLDER_PAT.subn(
                lambda _m, rep=replacement: rep,
                result,
                count=1
            )

        # 保留原始空格与格式，不压缩空格

        reformatted_samples.append((result, prob))

    return reformatted_samples


def generate_samples_from_ner(sampler: GreedyQuantizedSampler, ner_text: str,
                              target_samples: int = 100, num_intervals: int = 10, dist: str = 'rel',
                              redundancy_factor: float = 1.3, exclude_reference: bool = True) -> Tuple[List[Tuple[str, float]], Dict[str, List[str]]]:
    """使用独立策略基于NER文本批量生成伪样本，同时返回真词→伪词映射。

    返回值：
        samples: [(text, prob), ...] 生成的混淆样本
        real_to_fake_mapping: Dict[str, List[str]] 真实词到伪词列表映射
    """

    template, sensitive_words = parse_ner_text(ner_text)
    if not sensitive_words:
        # 无敏感词，直接返回原句和空映射
        return [(template, 1.0)], {}

    # 构建真实词序列
    real_words: List[str] = []
    for sw in sensitive_words:
        for pos in sw.positions:
            if len(real_words) <= pos:
                real_words.extend([''] * (pos + 1 - len(real_words)))
            real_words[pos] = sw.word

    total_positions = len(real_words)

    # 为每个位置独立生成候选（即使是同一个词也分别生成）
    generated_candidates: Dict[int, List[Tuple[str, float]]] = {}

    # 用于收集真词到伪词候选的映射
    real_to_fake_mapping: Dict[str, set] = {}

    # 构建位置到敏感词的映射
    position_to_word: Dict[int, NERSensitiveWord] = {}
    for sw in sensitive_words:
        for pos in sw.positions:
            position_to_word[pos] = sw

    # 为每个位置独立生成候选
    for position in range(total_positions):
        if position in position_to_word:
            sw = position_to_word[position]
            
            # 创建独立策略前缀
            prefix_template = build_independent_prefix(template)
            
            # 精确替换第position个<redacted>为[TARGET]
            redacted_positions = []
            temp_text = prefix_template
            start = 0
            while True:
                pos = temp_text.find('<redacted>', start)
                if pos == -1:
                    break
                redacted_positions.append(pos)
                start = pos + len('<redacted>')
            
            if position < len(redacted_positions):
                # 替换第position个<redacted>
                target_pos = redacted_positions[position]
                prompt_template = (prefix_template[:target_pos] + 
                                 '[TARGET]' + 
                                 prefix_template[target_pos + len('<redacted>'):])
            else:
                # 防御性代码：如果位置不存在，使用第一个
                prompt_template = prefix_template.replace('<redacted>', '[TARGET]', 1)
            
            prompt = f"Your task is to guess the redacted [TARGET], and its word type is '{sw.word_type}', in the following sentence: <sentence>{prompt_template}</sentence> One possible word that fits in to [TARGET] is:"

            # 为每个位置独立生成候选
            candidates = sampler.generate_samples(
                prefix=prompt,
                suffix=sw.word,
                target_samples=target_samples,
                num_intervals=num_intervals,
                dist=dist,
                redundancy_factor=redundancy_factor,
                exclude_reference=exclude_reference
            )

            generated_candidates[position] = candidates

            # 收集映射
            fake_words = [c[0] for c in candidates]
            real_to_fake_mapping.setdefault(sw.word, set()).update(fake_words)

    # ------------------------------------------------------------------
    # 拼接最终样本：使用原始 ner_text，按顺序替换 <redacted> XXX (TYPE) </redacted>
    # 保证换行/空格等排版与输入一致
    # ------------------------------------------------------------------

    redacted_pat = re.compile(r'<redacted>.*?</redacted>', re.DOTALL)

    final_samples: List[Tuple[str, float]] = []
    for i in range(target_samples):
        cur_sentence = ner_text  # 使用原始带标注文本，保留排版
        total_prob = 1.0

        for pos in range(total_positions):
            cands = generated_candidates[pos]
            idx = i % len(cands)
            fake_word, prob = cands[idx]
            total_prob *= prob

            # 仅替换当前首个 <redacted> ... </redacted>
            # 使用函数替换，避免 fake_word 中的反斜杠被解释为转义序列
            cur_sentence, n_sub = redacted_pat.subn(
                lambda _m, rep=fake_word: rep,
                cur_sentence,
                count=1
            )
            if n_sub == 0:
                break

        # 压缩连续空格，保持换行
        cur_sentence = re.sub(r" {2,}", " ", cur_sentence)
        final_samples.append((cur_sentence, total_prob))

    final_samples.sort(key=lambda x: x[1], reverse=True)

    # 将 set 转为 list，确保可 JSON 序列化
    real_to_fake_mapping = {k: list(v) for k, v in real_to_fake_mapping.items()}

    # 统一后处理格式，确保严格遵循原始NER文本的排版
    final_samples = postprocess_format_consistency(
        ner_text=ner_text,
        samples=final_samples,
        real_to_fake_mapping=real_to_fake_mapping,
        real_words_sequence=real_words
    )

    return final_samples[:target_samples], real_to_fake_mapping


# =============================================================
# 新测试函数（与progressive_generator一致的格式）
# =============================================================


def test_optimized_gqs_generator():
    import time

    print("=== 测试 OptimizedGQS 连续采样 (独立策略) ===")

    model_name = "Qwen/Qwen3-14B"
    print(f"加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.float16)
    model.eval()

    sampler = GreedyQuantizedSampler(tokenizer, model, temperature=1.0, max_batch_size=8)

    test_cases = [
        {
            "name": "简单案例",
            "text": "Hey <redacted> Sarah (NAME) </redacted>, I heard <redacted> John (NAME) </redacted> is visiting <redacted> Paris (PLACE) </redacted> next month.",
            "target_samples": 20
        },
        {
            "name": "重复词案例",
            "text": "Yesterday <redacted> Sarah (NAME) </redacted> met <redacted> John (NAME) </redacted>, and today <redacted> Sarah (NAME) </redacted> called <redacted> Mary (NAME) </redacted>.",
            "target_samples": 16
        },
        {
            "name": "复杂案例", 
            "text": "Today, as Flight\n\n Attendant <redacted> Manuel Pavlov (NAME) </redacted>, with email <redacted> manuel.pavlov2706@gmail.net (EMAIL) </redacted> and residing at <redacted> 36528 Short Circle (LOCATION) </redacted>, I encountered a perplexing situation during my flight duties.",
            "target_samples": 16
        },
        {
            "name": "日期案例",
            "text": "Formal Report on the Financial and Legal Implications of the International Business Trip to Veridian City\n\nPrepared by: <redacted> Evelyn Harper (PERSON) </redacted>\nDate: <redacted> June 12, 2024 (DATE) </redacted>\n\nIntroduction:\nThis report provides a comprehensive overview of the financial expenditures, travel arrangements, and potential legal considerations involved in the recent international business trip undertaken by Mr. <redacted> Jonathan Pierce (PERSON) </redacted>, CFO of <redacted> Alpine Technologies Inc. (ORG) </redacted>, to Veridian City, Veridia Republic. The trip took place from <redacted> April 15 (DATE) </redacted> to <redacted> April 28, 2024 (DATE) </redacted>, with the primary purpose of negotiating a joint venture agreement with Veridian-based firm NovaCorps Ltd.\n\n1. Financial Overview:\nThe total budget allocated for the trip was <redacted> $18,750.00 (MONEY) </redacted>, covering airfare, accommodation, local transportation, meals, and incidental expenses. Airfare was booked through Global Air Services under reservation number GA-98475321, at a cost of <redacted> $3,200.00 (MONEY) </redacted> for a round-trip business class ticket departing from the company headquarters in Boston (BOS) to Veridian City International Airport (VCI).\n\nAccommodation was arranged at the Grand Veridian Hotel, located at 45 Riverbend Avenue, Veridian City. The hotel stay lasted 12 nights, with a nightly rate of <redacted> $220.00 (MONEY) </redacted>, incurring a total expense of <redacted> $2,640.00 (MONEY) </redacted>. Local transportation expenses, including taxi rides and car rentals, amounted to <redacted> $580.00 (MONEY) </redacted>. Meals and other expenses were reimbursed upon submission of receipts, totaling <redacted> $1,150.00 (MONEY) </redacted>.\n\nAn unanticipated expenditure of <redacted> $1,000.00 (MONEY) </redacted> was incurred due to the expedited visa processing fees required after submission delays. The visa, issued under case number VR-2024-5992, was essential for entry into Veridia Republic and was handled by the legal affairs department.\n\n2. Travel Arrangements and Compliance:\nAll travel arrangements complied with Alpine Technologies’ internal travel policies, which mandate economy-class travel for trips under 10 hours and business class for longer flights, adherence to per diem meal allowances, and use of preferred vendors for accommodation and transport services. The trip itinerary was submitted and approved by the finance department on <redacted> March 10, 2024 (DATE) </redacted>.\n\nHowever, it was noted that the visa application process was initiated late by the employee, resulting in additional legal fees and expedited processing charges. This delay could have jeopardized the planned negotiation schedule.\n\n3. Legal Considerations:\nDuring the trip, legal counsel identified several clauses within the proposed joint venture agreement that require further review. Notably, the non-compete clause restricts Alpine Technologies from engaging with competing firms within Veridia for a period of three years. Additionally, the arbitration clause mandates dispute resolution exclusively through Veridia’s commercial courts, which may present challenges given differences in legal frameworks.\n\nThe legal affairs team has recommended renegotiation of these provisions to include more neutral arbitration venues and clearer definitions of competitive activities. Furthermore, compliance with Veridian export regulations concerning technology transfer was emphasized to avoid potential sanctions.\n\n4. Recommendations:\n- Implement earlier initiation of visa applications for future international travel to mitigate expedited processing costs.\n- Conduct thorough pre-trip legal reviews of contractual documents to identify and address restrictive clauses prior to negotiation.\n- Consider budgeting additional contingency funds for unforeseen legal expenses.\n- Evaluate alternative arbitration venues acceptable to both parties to minimize jurisdictional risks.\n\nConclusion:\nThe international business trip to Veridian City was successful in advancing Alpine Technologies’ strategic partnership goals, albeit with some financial and legal challenges. By addressing the highlighted issues, future trips can be managed more efficiently, reducing unnecessary costs and legal risks.\n\nAttachments:\n- Expense receipts and reimbursement forms\n- Visa application documentation\n- Draft joint venture agreement excerpt\n\nPrepared by:\n<redacted> Evelyn Harper (PERSON) </redacted>\nCorporate Compliance Officer\n<redacted> Alpine Technologies Inc. (ORG) </redacted>",
            "target_samples": 12
        }
    ]

    for idx, case in enumerate(test_cases, 1):
        print("\n" + "=" * 50)
        print(f"测试案例 {idx}: {case['name']}")

        start_time = time.time()
        samples, mapping = generate_samples_from_ner(
            sampler,
            ner_text=case["text"],
            target_samples=case["target_samples"],
            num_intervals=2,
            dist='rel',
            redundancy_factor=1.3,
            exclude_reference=True  # 排除参考样本，只生成伪样本
        )
        elapsed = time.time() - start_time

        print(f"生成时间: {elapsed:.2f} 秒, 共生成 {len(samples)} 条伪样本")
        for j, (txt, prob) in enumerate(samples[:10], 1):
            print(f"  {j}. {txt} (概率: {prob:.5f})")

        print("\n真词到伪词映射:")
        for real_word, fake_words in mapping.items():
            print(f"  {real_word}: {fake_words}")


# 若直接运行本文件，则执行测试
if __name__ == "__main__":
    test_optimized_gqs_generator() 