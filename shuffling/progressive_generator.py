import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "/data1/huggingface"
import re
import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from collections import defaultdict
import math
import unicodedata

# 导入现有的采样器
from semantic_sampler import SemanticSampler
import torch_npu
from torch_npu.npu import amp # 导入AMP模块
from torch_npu.contrib import transfer_to_npu    # 使能自动迁移


@dataclass
class SensitiveWord:
    """敏感词信息"""
    word: str           # 敏感词文本
    word_type: str      # 敏感词类型 (如NAME, PLACE等)
    positions: List[int] # 在文本中出现的位置列表
    uid: str            # 唯一标识符


@dataclass
class SampleCandidate:
    """样本候选"""
    text: str                    # 当前的文本状态
    prob: float                  # 累积概率
    filled_positions: List[int]  # 已填充的位置索引
    word_mapping: Dict[str, str] # 敏感词到伪词的映射


class SimpleEnhancedGenerator:
    """
    简单增强的多目标词伪样本生成器
    
    核心思路：
    1. 每个位置独立生成，按索引拼接（保持GQS简单性）
    2. 除第一个位置外，每次生成使用前缀策略：
       - 独立：所有前缀都是占位符
       - 真实：前缀填入真实词
       - top_k：前缀填入质量最好的前k%伪词（从独立生成候选中随机选择）
    
    重要说明：
    - 采样器返回的是相对似然度（与参考样本的概率比值）
    - 越接近1.0的相对似然度表示质量越好
    - independent_candidates按照|prob-1.0|排序，确保高质量候选在前
    """
    
    def __init__(self, sampler: SemanticSampler, top_k_ratio: float = 0.4, beam_size: int = 4):
        self.sampler = sampler
        self.generated_candidates = {}  # 存储每个位置的生成结果 {position: [(word, prob), ...]}
        self.word_to_fakes = {}  # 词汇到伪词映射缓存 {word_uid: [fake_words]}
        self.independent_candidates = {}  # 存储每个位置独立生成的候选 {position: [(word, prob), ...]}
        self.root_to_fakes = {}  # root到伪词映射缓存 {root: [(word, prob), ...]}
        # top_k_ratio 用于 "top_k" 策略：选择质量最好的前 k% 候选作为采样池
        self.top_k_ratio = top_k_ratio
        # beam_size 控制每一步扩展前缀数量（仅在 position>0 时启用）
        self.beam_size = beam_size
        # 用于缓存经符号回退后的NER文本
        self._normalized_ner_text = None
        
    def parse_ner_text(self, ner_text: str) -> Tuple[str, List[SensitiveWord]]:
        """解析NER标注的文本，提取敏感词并生成模板。

        额外处理：若敏感词被符号包裹（如引号、括号、标点等），
        将符号回退到原文中，只保留核心词在<redacted>标签内。
        """
        pattern = re.compile(r'<redacted>(\s*)([^(]+?)(\s*)\(([^)]+)\)(\s*)</redacted>')

        # 使用字典收集相同词和类型的所有位置
        word_info: Dict[Tuple[str, str], List[int]] = {}
        position = 0
        normalized_parts: List[str] = []
        last_end = 0

        def _is_affix_char(ch: str) -> bool:
            if not ch:
                return False
            category = unicodedata.category(ch)
            return category.startswith('P') or category.startswith('S')

        for match in pattern.finditer(ner_text):
            start, end = match.span()
            normalized_parts.append(ner_text[last_end:start])

            leading_ws, word_raw, between_ws, word_type_raw, tail_ws = match.groups()
            word_type = word_type_raw.strip()

            # 清理敏感词首尾的标点/符号
            cleaned_word = word_raw.strip()
            prefix_affix = ''
            suffix_affix = ''

            while cleaned_word and _is_affix_char(cleaned_word[0]):
                prefix_affix += cleaned_word[0]
                cleaned_word = cleaned_word[1:]

            while cleaned_word and _is_affix_char(cleaned_word[-1]):
                suffix_affix = cleaned_word[-1] + suffix_affix
                cleaned_word = cleaned_word[:-1]

            cleaned_word = cleaned_word.strip()

            # 若剥离后为空，则回退到原始文本
            if not cleaned_word:
                cleaned_word = word_raw.strip()
                prefix_affix = ''
                suffix_affix = ''

            sanitized_segment = (
                f"{prefix_affix}<redacted>{leading_ws}{cleaned_word}{between_ws}({word_type_raw}){tail_ws}</redacted>{suffix_affix}"
            )
            normalized_parts.append(sanitized_segment)
            last_end = end

            key = (cleaned_word, word_type)
            word_info.setdefault(key, []).append(position)
            position += 1

        normalized_parts.append(ner_text[last_end:])
        normalized_ner_text = ''.join(normalized_parts)
        self._normalized_ner_text = normalized_ner_text

        # 直接生成敏感词列表
        sensitive_words = []
        for (word, word_type), positions in word_info.items():
            uid = f"{word}|{word_type}"  # 使用|分隔符，避免下划线冲突
            sensitive_words.append(SensitiveWord(
                word=word,
                word_type=word_type,
                positions=positions,
                uid=uid
            ))

        # 生成模板
        template = pattern.sub('<redacted>', normalized_ner_text)
        template = re.sub(r'\s+', ' ', template).strip()

        return template, sensitive_words
    
    def build_prefix_template(self, original_template: str, position: int, strategy: str) -> str:
        """
        根据策略构建前缀模板
        
        Args:
            original_template: 原始模板
            position: 当前位置
            strategy: 策略类型 ('independent', 'real', 'top_p')
        """
        if strategy == 'independent':
            # 独立策略：保持所有占位符
            return original_template
        
        # 对于其他策略，需要替换前面位置的占位符
        result_template = original_template
        
        for i in range(position):
            if strategy == 'real':
                # 真实策略：用真实词填充
                if i < len(self.real_words):
                    replacement_word = self.real_words[i]
                else:
                    replacement_word = '<redacted>'
            elif strategy == 'top_k':
                # Top-k 策略：从质量最好的前k%候选中随机选择
                # 由于independent_candidates已按质量排序（越接近1.0越好），直接取前k个
                if i in self.independent_candidates and self.independent_candidates[i]:
                    independent_cands = self.independent_candidates[i]
                    real_word = self.real_words[i] if i < len(self.real_words) else ''
                    fake_candidates = [(word, prob) for word, prob in independent_cands if word != real_word]

                    if fake_candidates:
                        # 取前 top_k_ratio 比例的高质量候选
                        k = max(1, int(len(fake_candidates) * self.top_k_ratio))
                        pool = [word for word, _ in fake_candidates[:k]]
                        replacement_word = random.choice(pool)
                    else:
                        replacement_word = '<redacted>'
                else:
                    replacement_word = '<redacted>'
            # elif strategy in ['closest', 'distant']:
            #     # 接近/远离策略：从独立生成的候选中选择伪词
            #     if i in self.independent_candidates and self.independent_candidates[i]:
            #         independent_cands = self.independent_candidates[i]
            #         # 首先排除真实词，只考虑伪词
            #         real_word = self.real_words[i] if i < len(self.real_words) else ''
            #         fake_candidates = [(word, prob) for word, prob in independent_cands if word != real_word]
                    
            #         if fake_candidates:
            #             if strategy == 'closest':
            #                 # 选择概率最高的伪词（从独立生成中）
            #                 replacement_word = fake_candidates[0][0]
            #             else:  # distant
            #                 # 选择中等概率的伪词（从独立生成中，避免极端）
            #                 mid_idx = min(len(fake_candidates) // 3, len(fake_candidates) - 1)
            #                 replacement_word = fake_candidates[mid_idx][0]
            #         else:
            #             # 如果没有伪词，使用占位符
            #             replacement_word = '<redacted>'
            #     else:
            #         replacement_word = '<redacted>'
            else:
                replacement_word = '<redacted>'
            
            # 替换第一个<REDACTED>为选定的词
            result_template = result_template.replace('<redacted>', replacement_word, 1)
        
        return result_template
    
    def generate_for_position(self, word: SensitiveWord, position: int, template: str, 
                            target_samples: int, num_intervals: int = 10, dist: str = 'rel',
                            redundancy_factor: float = 1.3, exclude_reference: bool = True) -> List[Tuple[str, float]]:
        """
        为特定位置生成候选词
        
        Args:
            exclude_reference: 是否排除参考样本 (True=排除, False=放在第一位)
        """
        # print(f"为位置 {position} 生成: {word.word} ({word.word_type})")
        
        # 计算冗余目标
        redundant_target = int(target_samples * redundancy_factor)
        # print(f"  目标: {target_samples}, 冗余目标: {redundant_target}")
        
        # -------------------------
        # 先尝试硬映射 / 软映射
        # -------------------------

        all_candidates = None
        root = self._get_root(word.word)
        root_valid = len(root) >= 3

        # 1) 硬映射：完全相同的 uid
        if word.uid in self.word_to_fakes:
            # 由于使用的是相对概率（与真实词的比值），跨上下文复用旧概率会失真
            # 命中映射时统一将概率置为 1.0（中性），仅复用候选文本
            cached = self.word_to_fakes[word.uid][:target_samples]
            all_candidates = [(w, 1.0) for (w, _p) in cached]

        # 2) 软映射：root 完全或部分匹配
        if all_candidates is None and root_valid:
            hit_root = next((r for r in self.root_to_fakes if r in root or root in r), None)
            if hit_root:
                base = self.root_to_fakes[hit_root]
                # 同理：软映射复用时，概率置为 1.0（中性）
                # 使用自适应替换，兼顾人名与邮箱本地部分
                all_candidates = [
                    (
                        self._substitute_adaptive(
                            real_word=word.word,
                            real_root=root,
                            fake_text=fake,
                            hit_root=hit_root,
                        ),
                        1.0,
                    )
                    for fake, _p in base
                ][:target_samples]
                # 将结果写入硬映射，后续可直接复用
                self.word_to_fakes[word.uid] = all_candidates

        # 2.5) 规则映射（日期/金额统一入口）：命中时直接覆盖候选，避免自由生成的噪声
        rule_candidates = self._generate_rule_based_candidates(word.word, word.word_type, int(redundant_target))
        if rule_candidates:
            all_candidates = rule_candidates
            self.word_to_fakes[word.uid] = list(all_candidates)
            if root_valid:
                self.root_to_fakes[root] = list(all_candidates)
            self.independent_candidates[position] = list(all_candidates)

        # 3) 若仍未命中，则进入生成流程
        if all_candidates is None:
            independent_only = []  # 仅在独立策略路径中使用

            if position == 0 or len(self.word_to_fakes) == 0:
                # 首位置：仅独立生成
                prefix_template = self.build_prefix_template(template, position, 'independent')
                # 精确替换第 position 个 <redacted> 为 [TARGET]
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
                    target_pos = redacted_positions[position]
                    prompt_template = (prefix_template[:target_pos] + '[TARGET]' + prefix_template[target_pos + len('<redacted>'):])
                else:
                    prompt_template = prefix_template.replace('<redacted>', '[TARGET]', 1)

                # prompt = f"""Your task is to guess the redacted [TARGET], which is {word.word_type}, in the following sentence: 
                #         <sentence>{prompt_template}</sentence> 
                #         One possible {word.word_type} that fits in to [TARGET] is:"""
                prompt = f"""Your task is to guess the redacted [TARGET], and its word type is '{word.word_type}', in the following sentence: 
                        <sentence>{prompt_template}</sentence> 
                        One possible word that fits in to [TARGET] is:"""
                # 独立前缀用于回退（与首位置生成提示保持一致）
                independent_prefix_text = prompt
                strategy_target = max(1, int(redundant_target))
                candidates = self.sampler.sample(
                    gamma=strategy_target,
                    prefix=prompt,
                    suffix=word.word,
                    num_intervals=num_intervals,
                    dist=dist,
                    independent_prefix=independent_prefix_text
                )
                all_candidates = list(candidates)
                independent_only = list(candidates)
            else:
                # 后续位置：beam 路径 + 批量采样（保证相同 uid 一致）
                real_prefix_template = self.build_prefix_template(template, position, 'real')

                # 1) 将历史位置按 uid 分组，构造每个 uid 的候选池（硬映射 > 独立候选 > 真实兜底）
                uid_to_positions = {}
                for i in range(position):
                    uid = self.position_to_uid.get(i)
                    if uid is None:
                        continue
                    uid_to_positions.setdefault(uid, []).append(i)

                uid_to_pool = {}
                for uid, pos_list in uid_to_positions.items():
                    pool_words = []
                    if uid in self.word_to_fakes and self.word_to_fakes[uid]:
                        pool_words = [w for (w, _p) in self.word_to_fakes[uid]]
                    if not pool_words:
                        first_pos = pos_list[0]
                        if first_pos in self.independent_candidates and self.independent_candidates[first_pos]:
                            real_word_fp = self.real_words[first_pos] if first_pos < len(self.real_words) else ''
                            fake_cands_fp = [(w, p) for w, p in self.independent_candidates[first_pos] if w != real_word_fp]
                            if fake_cands_fp:
                                k_fp = max(1, int(len(fake_cands_fp) * self.top_k_ratio))
                                pool_words = [w for w, _ in fake_cands_fp[:k_fp]]
                    if not pool_words:
                        any_pos = pos_list[0]
                        fallback_word = self.real_words[any_pos] if any_pos < len(self.real_words) else '<redacted>'
                        pool_words = [fallback_word]
                    uid_to_pool[uid] = pool_words

                # 2) 组合各 uid 的选择为路径（限制总数）
                ordered_uids = [self.position_to_uid[i] for i in range(position) if self.position_to_uid.get(i) is not None]
                seen = set()
                ordered_uids = [u for u in ordered_uids if not (u in seen or seen.add(u))]

                prefix_uid_choices = [[]]
                for uid in ordered_uids:
                    pool = uid_to_pool.get(uid, ['<redacted>'])
                    if not pool:
                        pool = ['<redacted>']
                    new_choices = []
                    for choice in prefix_uid_choices:
                        for w in pool:
                            new_choices.append(choice + [(uid, w)])
                    prefix_uid_choices = new_choices
                    if len(prefix_uid_choices) > self.beam_size:
                        prefix_uid_choices = random.sample(prefix_uid_choices, self.beam_size)

                # 3) 将 uid 选择映射回具体位置，生成扩展前缀文本
                expansion_prefixes = []
                for uid_choice in prefix_uid_choices:
                    uid_to_word = {u: w for (u, w) in uid_choice}
                    expanded = template
                    for _idx in range(position):
                        uid = self.position_to_uid.get(_idx)
                        replacement = uid_to_word.get(uid, self.real_words[_idx] if _idx < len(self.real_words) else '<redacted>')
                        expanded = expanded.replace('<redacted>', replacement, 1)
                    tmp_expanded = expanded.replace('<redacted>', '[TARGET]', 1)
                    # exp_prefix_text = f"""Your task is to guess the redacted [TARGET], which is {word.word_type}, in the following sentence: 
                    #         <sentence>{tmp_expanded}</sentence> 
                    #         One possible {word.word_type} that fits in to [TARGET] is:"""
                    exp_prefix_text = f"""Your task is to guess the redacted [TARGET], and its word type is '{word.word_type}', in the following sentence: 
                            <sentence>{tmp_expanded}</sentence> 
                            One possible word that fits in to [TARGET] is:"""
                    expansion_prefixes.append(exp_prefix_text)

                # 真实前缀转换为带 [TARGET] 的 prompt 文本
                real_tmp = real_prefix_template.replace('<redacted>', '[TARGET]', 1)
                # real_prefix_text = f"""Your task is to guess the redacted [TARGET], which is {word.word_type}, in the following sentence: 
                #         <sentence>{real_tmp}</sentence> 
                #         One possible {word.word_type} that fits in to [TARGET] is:"""
                real_prefix_text = f"""Your task is to guess the redacted [TARGET], and its word type is '{word.word_type}', in the following sentence: 
                        <sentence>{real_tmp}</sentence> 
                        One possible word that fits in to [TARGET] is:"""
                # 独立前缀（所有占位符保持占位，仅替换当前为 [TARGET]）
                indep_template = self.build_prefix_template(template, position, 'independent')
                indep_tmp = indep_template.replace('<redacted>', '[TARGET]', 1)
                independent_prefix_text = f"""Your task is to guess the redacted [TARGET], and its word type is '{word.word_type}', in the following sentence: 
                        <sentence>{indep_tmp}</sentence> 
                        One possible word that fits in to [TARGET] is:"""

                # 调用批量采样：共享真实前缀，按各扩展前缀计算候选分布
                if not expansion_prefixes:
                    # 退回独立策略
                    prefix_template = self.build_prefix_template(template, position, 'independent')
                    prompt_template = prefix_template.replace('<redacted>', '[TARGET]', 1)
                    # prompt = f"""Your task is to guess the redacted [TARGET], which is {word.word_type}, in the following sentence: 
                    #         <sentence>{prompt_template}</sentence> 
                    #         One possible {word.word_type} that fits in to [TARGET] is:"""
                    prompt = f"""Your task is to guess the redacted [TARGET], and its word type is '{word.word_type}', in the following sentence: 
                            <sentence>{prompt_template}</sentence> 
                            One possible word that fits in to [TARGET] is:"""
                    strategy_target = max(1, int(redundant_target))
                    candidates = self.sampler.sample(
                        gamma=strategy_target,
                        prefix=prompt,
                        suffix=word.word,
                        num_intervals=num_intervals,
                        dist=dist,
                        independent_prefix=independent_prefix_text
                    )
                    all_candidates = list(candidates)
                else:
                    per_path_target = max(1, int(redundant_target / max(1, len(expansion_prefixes))))
                    sample_results = self.sampler.sample_multi(
                        gamma=per_path_target,
                        prefix=real_prefix_text,
                        suffix=word.word,
                        expansion_prefixes=expansion_prefixes,
                        num_intervals=num_intervals,
                        dist=dist,
                        stop_tokens=["\n"],
                        independent_prefix=independent_prefix_text
                    )
                    merged = []
                    for _idx in sample_results:
                        merged.extend(sample_results[_idx])
                    all_candidates = merged
            
            # EMAIL 专用后处理：统一固定域名，仅扰动本地部分（字母替换+数字随机化）
            if (word.word_type or '').upper() == 'EMAIL':
                try:
                    all_candidates = [
                        (self._substitute_email(real_email=word.word, fake_text=word_text), prob)
                        for (word_text, prob) in all_candidates
                    ]
                except Exception:
                    pass

            # 合并、去重、排序，处理参考样本
            seen_words = set()
            unique_candidates = []
            ref_sample = None  # 存储参考样本
            ref_word = word.word  # 当前位置的真实词
            
            for word_text, prob in all_candidates:
                # 检查是否为参考样本（真实词）
                if word_text.strip() == ref_word.strip():
                    if not exclude_reference:
                        ref_sample = (word_text, prob)
                    # 参考样本单独处理，不参与去重
                    continue
                
                if word_text not in seen_words:
                    seen_words.add(word_text)
                    unique_candidates.append((word_text, prob))
            
            # 按接近 1.0 的相对似然度排序（越接近越好）
            unique_candidates.sort(key=lambda x: abs(x[1] - 1.0))
            # print(f"  去重后候选数: {len(unique_candidates)}")
            
            # 根据exclude_reference参数决定最终结果
            if not exclude_reference and ref_sample is not None:
                # 参考样本放在第一位，保持伪样本数量为target_samples
                final_candidates = [ref_sample] + unique_candidates
                final_target = target_samples + 1  # 额外的1个是参考样本
            else:
                # 排除参考样本，只要伪样本
                final_candidates = unique_candidates
                final_target = target_samples
            
            # 若没有任何候选，回退到使用真实词占位，避免后续死循环/除零
            if final_target > 0 and len(final_candidates) == 0:
                final_candidates = [(ref_word, 1.0)]
            
            # 简单处理数量：不够就按顺序复制，够了就截断（无死循环）
            if len(final_candidates) < final_target:
                extended = list(final_candidates)
                needed = final_target - len(extended)
                if not exclude_reference and ref_sample is not None:
                    # 有参考样本时，只复制伪样本部分（若为空则回退到所有候选）
                    rep_pool = unique_candidates if len(unique_candidates) > 0 else extended[1:]
                else:
                    rep_pool = final_candidates
                if not rep_pool:
                    # 仍为空则回退到已收集的任何候选，确保可填充
                    rep_pool = extended
                if rep_pool:
                    times = (needed + len(rep_pool) - 1) // len(rep_pool)
                    extended.extend((rep_pool * times)[:needed])
                all_candidates = extended[:final_target]
            else:
                # 候选足够，直接截断
                all_candidates = final_candidates[:final_target]

            # 单点清洗：只在最终确定 all_candidates 后做一次清洗
            all_candidates = [
                (self._strip_special_if_plain(word.word, w), p) for (w, p) in all_candidates
            ]
            
            # print(f"  最终候选数: {len(all_candidates)}")
            
            # 缓存候选（硬+软）
            self.word_to_fakes[word.uid] = list(all_candidates)
            if root_valid:
                self.root_to_fakes[root] = list(all_candidates)
            
            # 保存独立生成的候选（用于后续top_k策略）
            # 按照与1.0的距离排序：越接近1.0质量越好（相对似然度最优）
            if independent_only:
                independent_only.sort(key=lambda x: abs(x[1] - 1.0))
                self.independent_candidates[position] = independent_only
            else:
                # 若没有独立候选（说明该位置走了硬/软映射路径），
                # 也写入一份排序后的候选，保证后续前缀可用
                self.independent_candidates[position] = sorted(all_candidates, key=lambda x: abs(x[1] - 1.0))
        
        # 保存结果
        self.generated_candidates[position] = all_candidates
        
        return all_candidates
    
    def generate_samples(self, ner_text: str, target_samples: int = 100,
                        num_intervals: int = 10, dist: str = 'rel',
                        redundancy_factor: float = 1.3, exclude_reference: bool = True) -> Tuple[List[Tuple[str, float]], Dict[str, List[str]]]:
        """
        生成伪样本（简单增强版本）
        
        Args:
            exclude_reference: 是否排除参考样本 (True=排除, False=放在第一位)
            
        Returns:
            Tuple containing:
            - List of (text, probability) tuples for generated samples
            - Dict mapping real words to lists of fake words: {real_word: [fake_word1, fake_word2, ...]}
        """
        # print(f"开始生成伪样本，目标数量: {target_samples}")
        
        # 1. 解析文本
        template, sensitive_words = self.parse_ner_text(ner_text)
        ner_text = getattr(self, '_normalized_ner_text', ner_text)
        # print(f"解析出 {len(sensitive_words)} 个唯一敏感词")
        # print(f"模板: {template}")
        
        if not sensitive_words:
            return [(template, 1.0)], {}
        
        # 2. 存储真实词序列（用于真实策略）
        self.real_words = []
        position_to_word = {}
        # 记录 position -> uid，用于后续 beam 路径一致性（相同实体复用同一伪词）
        self.position_to_uid = {}
        for word in sensitive_words:
            for pos in word.positions:
                position_to_word[pos] = word.word
                if len(self.real_words) <= pos:
                    self.real_words.extend([''] * (pos + 1 - len(self.real_words)))
                self.real_words[pos] = word.word
                self.position_to_uid[pos] = word.uid
        
        total_positions = len(self.real_words)
        
        # print(f"总位置数: {total_positions}, 每个位置生成 {target_samples} 个候选")
        
        # 3. 清空生成缓存
        self.generated_candidates = {}
        self.word_to_fakes = {}  # 清空词汇映射缓存
        self.root_to_fakes = {}  # 清空root映射缓存
        self.independent_candidates = {}  # 清空独立候选缓存
        
        # 4. 依次为每个位置生成候选
        for position in range(total_positions):
            word = None
            for w in sensitive_words:
                if position in w.positions:
                    word = w
                    break
            
            if word:
                candidates = self.generate_for_position(
                    word, position, template, target_samples, num_intervals, dist, redundancy_factor, exclude_reference
                )
                # print(f"  位置 {position}: 生成了 {len(candidates)} 个候选")
        
        # 5. 按索引拼接生成最终样本，保持原始格式
        redacted_pat = re.compile(r'<redacted>.*?</redacted>', re.DOTALL)
        final_samples = []
        real_to_fake_mapping = {}  # {real_word: [fake_word1, fake_word2, ...]}
        
        # 确定实际的目标样本数（考虑参考样本）
        if not exclude_reference:
            # 如果不排除参考样本，那么每个位置都有target_samples+1个候选
            actual_target = target_samples + 1
        else:
            actual_target = target_samples
        
        # 简单的索引拼接：第i个样本 = 各位置的第i个候选
        for i in range(actual_target):
            current_sentence = ner_text  # 使用原始带标注文本，保留排版
            total_prob = 1.0
            
            # 对每个位置，使用第i个候选
            for position in range(total_positions):
                if position in self.generated_candidates:
                    candidates = self.generated_candidates[position]
                    # 确保索引不超出范围
                    idx = i % len(candidates)
                    fake_word, word_prob = candidates[idx]
                    
                    # 获取该位置的真实词
                    real_word = position_to_word.get(position, '')
                    
                    # 建立真词到伪词的映射（一对多）
                    if real_word.strip() and fake_word.strip():
                        if real_word not in real_to_fake_mapping:
                            real_to_fake_mapping[real_word] = []
                        if fake_word not in real_to_fake_mapping[real_word]:
                            real_to_fake_mapping[real_word].append(fake_word)
                    
                    # 替换首个 <redacted>…</redacted>，保持顺序（使用lambda避免反斜杠转义问题）
                    current_sentence, _ = redacted_pat.subn(lambda _m, rep=fake_word: rep, current_sentence, count=1)
                    total_prob *= word_prob

            # 压缩连续空格
            # 保留原始空格与格式，不压缩空格
 
            final_samples.append((current_sentence, total_prob))
        
        # 按概率排序
        final_samples.sort(key=lambda x: x[1], reverse=True)
        
        # 后处理：使用原始NER文本和mapping重新格式化所有样本，确保格式一致
        final_samples = self._postprocess_format_consistency(
            ner_text=ner_text,
            samples=final_samples,
            real_to_fake_mapping=real_to_fake_mapping,
            real_words_sequence=self.real_words
        )
        
        # if exclude_reference:
        #     print(f"生成完成，共 {len(final_samples)} 个伪样本（已排除参考样本）")
        # else:
        #     print(f"生成完成，共 {len(final_samples)} 个样本（包含参考样本）")
        
        # # 输出词映射统计
        # print("真词到伪词映射统计：")
        # for real_word, fake_words in real_to_fake_mapping.items():
        #     print(f"  '{real_word}' -> {len(fake_words)} 个伪词: {fake_words[:5]}{'...' if len(fake_words) > 5 else ''}")
        
        return final_samples, real_to_fake_mapping


    def _postprocess_format_consistency(
        self,
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
        
        Args:
            ner_text: 原始带<redacted>标注的文本
            samples: 生成的样本列表 [(text, prob), ...]
            real_to_fake_mapping: 真词到伪词的映射
            real_words_sequence: 按位置顺序的真实词列表
            
        Returns:
            格式统一后的样本列表
        """
        PLACEHOLDER_PAT = re.compile(r'<redacted>.*?</redacted>', re.DOTALL)
        
        reformatted_samples = []
        source_text = getattr(self, '_normalized_ner_text', ner_text)
        
        for sample_idx, (old_text, prob) in enumerate(samples):
            # 使用符号回退后的NER文本重建
            result = source_text
            
            # 按顺序替换每个<redacted>标签
            for real_word in real_words_sequence:
                fake_list = real_to_fake_mapping.get(real_word, [])
                
                if not fake_list:
                    # 如果没有映射，使用原词
                    replacement = real_word
                else:
                    # 使用循环索引选择伪词
                    replacement = fake_list[sample_idx % len(fake_list)]

                # 按真实词多词大小写模式对齐伪词
                replacement = self._apply_casing_by_pattern(real_word, replacement)
                
                # 替换第一个<redacted>标签
                # 使用lambda避免反斜杠等特殊字符被正则表达式解释
                result, _ = PLACEHOLDER_PAT.subn(
                    lambda _m, rep=replacement: rep,
                    result,
                    count=1
                )
            
            # 只压缩连续的空格（2个或以上），保留换行和其他格式
            # 保留原始空格与格式，不压缩空格
            
            reformatted_samples.append((result, prob))
        
        return reformatted_samples

    # ------------------------------------------------------------------
    # 词根提取与大小写调整辅助函数
    # ------------------------------------------------------------------

    def _get_root(self, word_text: str) -> str:
        """提取更稳健的词根 (root)
        处理要点：
        1. 若包含邮件地址，截取 "@" 之前部分
        2. 在剩余字符串中提取所有长度≥3 的连续字母段并拼接
           - 'manuelpavlov2706@gmail.net' → 'manuelpavlov'
           - 'A1234567pavlov' → 'apavlov'
           - '2023PavlovManuel' → 'pavlovmanuel'
        3. 转小写返回
        """
        import re
        if not word_text:
            return ""

        # 去掉邮件域名
        word_text = word_text.split('@', 1)[0]

        # 提取长度≥3 的字母序列
        segments = re.findall(r'[A-Za-z]{3,}', word_text)
        if not segments:
            # 回退：取全部字母序列（>=1），防止完全丢失
            segments = re.findall(r'[A-Za-z]+', word_text)
            if not segments:
                return ""

        root = ''.join(segments).lower()
        return root

    def _match_casing(self, fake_word: str, real_word: str) -> str:
        """按真实词的大小写模式返回伪词"""
        if not fake_word:
            return fake_word
        if real_word.isupper():
            return fake_word.upper()
        # 多单词 Title
        if real_word.istitle():
            return fake_word.title()
        # 首字母大写
        if real_word[0].isupper() and real_word[1:].islower():
            return fake_word.capitalize()
        return fake_word

    def _apply_casing_by_pattern(self, real_text: str, fake_text: str) -> str:
        """按 real_text 的多词大小写模式对齐 fake_text。
        - real/fake 以空格切分为词；
        - 不改变 fake 的词数；
        - 仅对前 N 个词（N = min(len(fake), len(real))）逐词应用 _match_casing；
        """
        if not real_text or not fake_text:
            return fake_text

        # 邮箱不做大小写对齐（邮箱通常小写，且已由专用逻辑处理）
        if self._is_email(real_text):
            return fake_text

        real_tokens = [t for t in re.split(r"\s+", real_text.strip()) if t]
        fake_tokens = [t for t in re.split(r"\s+", fake_text.strip()) if t]

        if not real_tokens or not fake_tokens:
            return fake_text

        # 仅对齐前 N 个词，保持 fake 词数不变
        N = min(len(fake_tokens), len(real_tokens))
        adjusted = list(fake_tokens)
        for i in range(N):
            adjusted[i] = self._match_casing(fake_tokens[i], real_tokens[i])
        return ' '.join(adjusted)

    # ------------------------------------------------------------
    # 特殊符号处理：当真实词不含特殊符号而伪词包含时，移除伪词中的特殊符号
    # 仅保留字母/数字与空白，压缩多余空白
    # ------------------------------------------------------------
    def _contains_special(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        return re.search(r"[^A-Za-z0-9\s]", text or "") is not None

    def _strip_special_chars(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        cleaned = re.sub(r"[^A-Za-z0-9\s]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _strip_special_if_plain(self, real_word: str, fake_text: str) -> str:
        if not isinstance(fake_text, str):
            return fake_text
        # 真实词不含特殊符号且伪词包含时，移除伪词中的特殊符号
        if not self._contains_special(real_word) and self._contains_special(fake_text):
            cleaned = self._strip_special_chars(fake_text)
            return cleaned if cleaned else fake_text
        return fake_text

    def _substitute_root(self, real_word: str, real_root: str, fake_root: str) -> str:
        """在 real_word 中替换 real_root 为 fake_root。
        规则：
        1. 若 real_root 未直接匹配到，则返回按 real_word 大小写风格处理后的 fake_root（整体替换）。
        2. 若 real_root 匹配到：
           • 如果 root_seg 不含空格，则移除 fake_root 中的空格；
           • 根据 root_seg 的大小写（全小写 / 全大写 / 首字母大写 / Title）转换 fake_root；
        """
        import re
        lower_real = real_word.lower()
        idx = lower_real.find(real_root)

        if idx == -1:
            # 构建允许分隔符的模式，但最后一个字母后不再消费额外符号，避免覆盖到 '@'
            pattern = ''.join(
                c + (r'[^A-Za-z]*' if idx < len(real_root) - 1 else '')
                for idx, c in enumerate(real_root)
            )
            match = re.search(pattern, lower_real)
            if match:
                start, end = match.span()
                before = real_word[:start]
                root_seg = real_word[start:end]
                after = real_word[end:]
            else:
                # 仍未匹配，整体替换
                return self._match_casing(fake_root, real_word)
        else:
            before = real_word[:idx]
            root_seg = real_word[idx: idx + len(real_root)]
            after = real_word[idx + len(real_root):]

        # 1) 去除 fake_root 中所有空白，便于后续按分隔符重组
        base_fake = re.sub(r'\s+', '', fake_root)

        # 2) 根据 root_seg 的大小写风格转换 base_fake
        if root_seg.islower():
            base_fake = base_fake.lower()
        elif root_seg.isupper():
            base_fake = base_fake.upper()
        elif root_seg.istitle():
            base_fake = base_fake.title()
        elif root_seg[0].isupper() and root_seg[1:].islower():
            base_fake = base_fake.capitalize()

        # 3) 若 root_seg 含有非字母的分隔符（空格、下划线、点等），按照同样模式插入到 base_fake
        sep_tokens = re.findall(r'[^A-Za-z]+', root_seg)
        alpha_tokens = [t for t in re.split(r'[^A-Za-z]+', root_seg) if t]

        if sep_tokens:
            # 计算每个字母 token 的长度，用于切分 base_fake
            lengths = [len(t) for t in alpha_tokens]
            if sum(lengths) == 0:
                rebuilt_fake = base_fake  # fallback
            else:
                parts = []
                cursor = 0
                for l in lengths:
                    parts.append(base_fake[cursor:cursor + l])
                    cursor += l
                # 若剩余字符不足，最后一段接收余量
                if cursor < len(base_fake):
                    parts[-1] += base_fake[cursor:]

                # 重建字符串，交替插入分隔符
                rebuilt_fake = parts[0]
                for sep, token in zip(sep_tokens, parts[1:]):
                    rebuilt_fake += sep + token
        else:
            rebuilt_fake = base_fake

        return before + rebuilt_fake + after

    # ------------------------------------------------------------------
    # 自适应替换：兼顾人名与邮箱名
    # ------------------------------------------------------------------

    def _is_email(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        t = text.strip()
        if '@' not in t:
            return False
        # 宽松判定：无空白，含 @，且 @ 之后至少含一处点
        if re.search(r'\s', t):
            return False
        local, sep, domain = t.partition('@')
        if not local or not domain:
            return False
        if '.' not in domain:
            return False
        return True

    def _extract_alpha_tokens(self, text: str) -> List[str]:
        return re.findall(r'[A-Za-z]+', text or '')

    def _substitute_email(self, real_email: str, fake_text: str) -> str:
        """在邮箱文本中替换本地部分（@ 之前），保持原域名不变；
        - 字母片段：按 fake_text 的字母 token 顺序替换，并对齐大小写；
        - 数字：逐位随机化（与原数字不同），分隔符（. _ - 等）保留；
        """
        if '@' not in real_email:
            return real_email

        local, domain = real_email.split('@', 1)

        # 准备替换用 token（仅字母 token）
        if self._is_email(fake_text):
            fake_local = fake_text.split('@', 1)[0]
            fake_tokens = self._extract_alpha_tokens(fake_local)
        else:
            fake_tokens = self._extract_alpha_tokens(fake_text)
        if not fake_tokens:
            fallback = self._get_root(fake_text)
            fake_tokens = [fallback] if fallback else ['user']

        idx_ptr = 0

        def repl_alpha(m: re.Match) -> str:
            nonlocal idx_ptr
            src = m.group(0)
            tok = fake_tokens[idx_ptr % len(fake_tokens)]
            idx_ptr += 1
            if src.isupper():
                return tok.upper()
            if src.islower():
                return tok.lower()
            if src.istitle():
                return tok.title()
            return tok.lower()

        # 先替换字母片段
        new_local = re.sub(r'[A-Za-z]+', repl_alpha, local)

        # 再随机化数字（仅本地部分）
        def repl_digit(m: re.Match) -> str:
            s = m.group(0)
            out = []
            for ch in s:
                if ch.isdigit():
                    pool = [str(d) for d in range(10) if str(d) != ch]
                    out.append(random.choice(pool))
                else:
                    out.append(ch)
            return ''.join(out)

        new_local = re.sub(r'\d+', repl_digit, new_local)
        return new_local + '@' + domain

    def _align_fake_component_for_name(self, real_word: str, fake_text: str, real_root: str, hit_root: str) -> str:
        """当真实词为单词而假名为多词时，选择最合适的那一部分：
        - real_root 是 hit_root 的后缀 -> 取假名最后一词（更像姓氏）
        - real_root 是 hit_root 的前缀 -> 取假名第一词（更像名）
        - 其他 -> 默认取最后一词
        若真实词为多词，则返回与真实词词数对齐的前若干词（循环补齐）。
        """
        real_tokens = self._extract_alpha_tokens(real_word)
        fake_tokens = self._extract_alpha_tokens(fake_text)

        if not fake_tokens:
            fallback = self._get_root(fake_text)
            return fallback if fallback else fake_text

        if len(real_tokens) <= 1:
            # 单词：按前后缀启发式选一词
            if real_root and hit_root:
                if hit_root.endswith(real_root):
                    return fake_tokens[-1]
                if hit_root.startswith(real_root):
                    return fake_tokens[0]
            return fake_tokens[-1]

        # 多词：不改变 fake 词数，仅返回 fake 的前 K 词（K = min(len(fake), len(real)))
        K = min(len(fake_tokens), len(real_tokens))
        return ' '.join(fake_tokens[:K])

    def _substitute_adaptive(self, real_word: str, real_root: str, fake_text: str, hit_root: str) -> str:
        """自适应替换入口：
        - 若真实词是邮箱：替换本地部分，保留域名与结构
        - 若真实词是名字（或一般词）：
            * 若假词是邮箱：用其本地部分 tokens 生成名字并对齐词数
            * 若假词是多词：当真实词为单词时选择最合适的单词作为替换
        最终都走 _substitute_root 以继承大小写与分隔符风格。
        """
        if self._is_email(real_word):
            return self._substitute_email(real_email=real_word, fake_text=fake_text)

        # 真实词为名字/一般词
        # 生成用于替换的 fake_name 文本
        if self._is_email(fake_text):
            fake_local = fake_text.split('@', 1)[0]
            fake_tokens = self._extract_alpha_tokens(fake_local)
        else:
            fake_tokens = self._extract_alpha_tokens(fake_text)

        if not fake_tokens:
            fallback = self._get_root(fake_text)
            fake_tokens = [fallback] if fallback else []

        real_tokens = self._extract_alpha_tokens(real_word)
        if len(real_tokens) <= 1:
            chosen = self._align_fake_component_for_name(
                real_word=real_word,
                fake_text=' '.join(fake_tokens),
                real_root=real_root,
                hit_root=hit_root,
            )
            return self._substitute_root(real_word, real_root, chosen)

        # 多词情形：不改变 fake 词数，只取前 K 词
        K = min(len(fake_tokens), len(real_tokens))
        return self._substitute_root(real_word, real_root, ' '.join(fake_tokens[:K]))

    # ------------------------------------------------------------------
    # 规则映射：星期与月份
    # ------------------------------------------------------------------
    def _generate_week_month_candidates(self, word_text: str, k: int) -> List[Tuple[str, float]]:
        """当敏感词中出现星期或月份时，基于规则直接生成候选。
        - 保持英文缩写/全称、是否带点、大小写风格
        - 保持中文“星期/周”前缀与“日/天”的周日风格
        - 保持中文月份使用阿拉伯数字或中文数字的原始风格
        返回：[(candidate_text, 1.0), ...]，若未命中规则则返回空列表
        """
        import re

        if not word_text:
            return []

        # 英文星期与月份定义
        weekdays_full = [
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
        ]
        weekdays_abbr = [
            "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"
        ]
        months_full = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        months_abbr = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
        ]

        # 英文匹配（允许缩写 + 全称，缩写可带点）
        re_weekday_en = re.compile(
            r"\b(Mon(?:\.|day)?|Tue(?:\.|sday)?|Wed(?:\.|nesday)?|Thu(?:\.|rsday)?|Fri(?:\.|day)?|Sat(?:\.|urday)?|Sun(?:\.|day)?)\b",
            re.IGNORECASE,
        )
        re_month_en = re.compile(
            r"\b(Jan(?:\.|uary)?|Feb(?:\.|ruary)?|Mar(?:\.|ch)?|Apr(?:\.|il)?|May|Jun(?:\.|e)?|Jul(?:\.|y)?|Aug(?:\.|ust)?|Sep(?:\.|t(?:\.|ember)?)?|Oct(?:\.|ober)?|Nov(?:\.|ember)?|Dec(?:\.|ember)?)\b",
            re.IGNORECASE,
        )

        # 中文匹配
        re_weekday_zh = re.compile(r"(星期|周)([一二三四五六日天])")
        re_month_zh_digit = re.compile(r"([1-9]|1[0-2])月")
        re_month_zh_cn = re.compile(r"(十一|十二|十[一二]?|[一二三四五六七八九])月")

        has_en_week = bool(re_weekday_en.search(word_text))
        has_en_month = bool(re_month_en.search(word_text))
        has_zh_week = bool(re_weekday_zh.search(word_text))
        has_zh_month = bool(re_month_zh_digit.search(word_text) or re_month_zh_cn.search(word_text))

        if not (has_en_week or has_en_month or has_zh_week or has_zh_month):
            return []

        def apply_case_like(template: str, s: str) -> str:
            # 使用已有大小写匹配逻辑，去掉末尾点再判断风格
            core = template[:-1] if template.endswith('.') else template
            return self._match_casing(s, core)

        def choose_weekday_en(abbr: bool) -> str:
            base = random.choice(weekdays_abbr if abbr else weekdays_full)
            return base

        def choose_month_en(abbr: bool) -> str:
            base = random.choice(months_abbr if abbr else months_full)
            return base

        def num_to_cn(n: int) -> str:
            cn_digits = {1: "一", 2: "二", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八", 9: "九", 10: "十"}
            if n <= 10:
                return cn_digits[n]
            if n == 11:
                return "十一"
            if n == 12:
                return "十二"
            return str(n)

        candidates: List[str] = []

        # 为了得到足够的去重样本，尝试次数上限
        attempts = max(10, k * 4)
        while len(candidates) < k and attempts > 0:
            attempts -= 1
            new_text = word_text

            # 英文星期替换（在一次样本中固定选择）
            if has_en_week:
                chosen_abbr = None  # 是否缩写由首个匹配决定
                had_dot_cache = {}

                def repl_week_en(m: re.Match) -> str:
                    tok = m.group(0)
                    had_dot = tok.endswith('.')
                    core = tok[:-1] if had_dot else tok
                    lower = core.lower()
                    is_abbr = len(lower) <= 3 or not lower.endswith('day')
                    nonlocal chosen_abbr
                    if chosen_abbr is None:
                        chosen_abbr = is_abbr
                    base = choose_weekday_en(chosen_abbr)
                    out = apply_case_like(core, base)
                    return out + ('.' if had_dot else '')

                new_text = re_weekday_en.sub(repl_week_en, new_text)

            # 英文月份替换（在一次样本中固定选择）
            if has_en_month:
                chosen_abbr = None

                def repl_month_en(m: re.Match) -> str:
                    tok = m.group(0)
                    had_dot = tok.endswith('.')
                    core = tok[:-1] if had_dot else tok
                    lower = core.lower()
                    # 识别缩写（May 特殊，既是全称也是缩写，这里按全称处理）
                    is_abbr = (
                        lower in [s.lower() for s in months_abbr]
                        or (len(lower) <= 4 and lower not in ["may"]) and not lower.endswith('ber') and not lower.endswith('ary') and not lower.endswith('une') and not lower.endswith('uly')
                    )
                    nonlocal chosen_abbr
                    if chosen_abbr is None:
                        chosen_abbr = is_abbr
                    base = choose_month_en(chosen_abbr)
                    out = apply_case_like(core, base)
                    return out + ('.' if had_dot else '')

                new_text = re_month_en.sub(repl_month_en, new_text)

            # 中文星期替换（一次样本固定选一种目标星期与“日/天”风格）
            if has_zh_week:
                # 选择 1..7，其中 7 代表周日
                day_idx = random.randint(1, 7)

                def repl_week_zh(m: re.Match) -> str:
                    prefix = m.group(1)  # 星期/周
                    orig_day = m.group(2)  # 一二三四五六日/天
                    if day_idx == 7:
                        # 保持“日/天”的原始风格
                        day_char = '天' if orig_day == '天' else '日'
                    else:
                        day_char = "一二三四五六"[day_idx - 1]
                    return prefix + day_char

                new_text = re_weekday_zh.sub(repl_week_zh, new_text)

            # 中文月份替换（一次样本固定选一个目标月份，保持阿拉伯/中文数字风格）
            if has_zh_month:
                month_idx = random.randint(1, 12)

                def repl_month_zh_digit(m: re.Match) -> str:
                    return f"{month_idx}月"

                def repl_month_zh_cn(m: re.Match) -> str:
                    return f"{num_to_cn(month_idx)}月"

                new_text = re_month_zh_digit.sub(repl_month_zh_digit, new_text)
                new_text = re_month_zh_cn.sub(repl_month_zh_cn, new_text)

            if new_text not in candidates:
                candidates.append(new_text)

        return [(c, 1.0) for c in candidates[:k]]


    # ------------------------------------------------------------------
    # 规则映射：金额（Money）
    # ------------------------------------------------------------------
    def _generate_money_candidates(self, word_text: str, k: int) -> List[Tuple[str, float]]:
        """金额候选生成（仅依据 word_type==MONEY 触发）。
        - 保持原始字符串中的货币符号/代码/单位：若原文有则保留，若没有则不新增；
        - 仅在数字部分做轻微扰动；
        - 保留原本的分隔符/空格位置；
        返回：[(candidate_text, 1.0), ...]
        """
        import re
        if not word_text:
            return []

        s = word_text

        # 数字匹配：抓取所有包含千分位/小数的数字片段
        # 例："1,234.56"、"1.234,56"、"12 345"、"500"、"1.2"、"10,000"
        num_pat = re.compile(r"\d[\d,\.\s]*\d|\d")

        # 若没有数字则认为非金额
        if not num_pat.search(s):
            return []

        def mutate_number_like(num_str: str) -> str:
            """低可察觉金额扰动：
            - 若末尾存在连续的0（包括小数部分的0），默认不扰动这些0；
              仅微调其前面的最后一个非零数字；
              或者以一定概率将连续零段中的第一个0改为5（提高多样性）。
            - 若末尾无连续0，则仅微调倒数第一个非零数字（±1，受边界与首位保护）。
            - 始终保留原有分隔符与单位/符号位置，不新增符号。
            """
            # 收集数字序列
            digits = re.findall(r"\d", num_str)
            if not digits:
                return num_str

            def fill_back(new_digits_list: List[str]) -> str:
                it = iter(new_digits_list)
                out_chars = []
                for ch in num_str:
                    if ch.isdigit():
                        out_chars.append(next(it))
                    else:
                        out_chars.append(ch)
                return ''.join(out_chars)

            new_digits = list(digits)
            L = len(new_digits)

            # 预处理：若最后一个非零数字是5且不在最高位，先将其置为0
            last_nz = -1
            for i in range(L - 1, -1, -1):
                if new_digits[i] != '0':
                    last_nz = i
                    break
            if last_nz > 0 and new_digits[last_nz] == '5':
                new_digits[last_nz] = '0'

            # 计算末尾连续0长度（基于可能已更新的 new_digits）
            trailing_zeros = 0
            for d in reversed(new_digits):
                if d == '0':
                    trailing_zeros += 1
                else:
                    break

            def perturb_nonzero(idx: int) -> None:
                if idx < 0:
                    return
                old = int(new_digits[idx])
                if old == 0:
                    return
                deltas: List[int] = []
                if old < 9:
                    deltas.append(1)
                # 首位不允许变成0
                if old > 1 or idx != 0:
                    deltas.append(-1)
                if not deltas:
                    return
                delta = random.choice(deltas)
                nv = old + delta
                if idx == 0 and nv == 0:
                    nv = old
                new_digits[idx] = str(max(0, min(9, nv)))

            def apply_progressive_perturb(start_idx: int, max_steps: int = 3) -> None:
                """从右向左逐位缩小扰动（范围型）：
                - 位置 start_idx 使用 1..max_steps 的随机幅度；
                - 左侧依次使用 1..(max_steps-1)、1..(max_steps-2)… 的随机幅度；
                - 始终保证不越界，且首位不变为 0（若原首位非 0）。
                """
                if start_idx < 0:
                    return
                step = max_steps
                pos = start_idx
                while pos >= 0 and step >= 1:
                    if new_digits[pos] != '0':
                        old = int(new_digits[pos])
                        # 在 1..step 的范围内随机选择扰动幅度与方向，收集所有可行候选
                        candidates = []
                        for mag in range(1, step + 1):
                            for sign in (-1, 1):
                                nv = old + sign * mag
                                if 0 <= nv <= 9:
                                    # 保证首位不为 0（若原首位非 0）
                                    if not (pos == 0 and nv == 0 and digits[0] != '0'):
                                        candidates.append(nv)
                        if candidates:
                            nv = random.choice(candidates)
                            new_digits[pos] = str(nv)
                    pos -= 1
                    step -= 1

            if trailing_zeros > 0:
                # 末尾连续零的起始位置（在 digits 中）
                zero_run_start = L - trailing_zeros
                # 1) 同时对零段前的若干非零位做逐位扩大扰动
                target_idx = -1
                for i in range(zero_run_start - 1, -1, -1):
                    if new_digits[i] != '0':
                        target_idx = i
                        break
                if target_idx >= 0:
                    apply_progressive_perturb(target_idx, max_steps=3)
                # 2) 可选：将零段中的第一个0改为5以降低可察觉性模式
                if zero_run_start < L and random.random() < 0.6:
                    new_digits[zero_run_start] = '5'
                elif zero_run_start >= L:
                    # 极端：全是0
                    new_digits[-1] = '5'
            else:
                # 无末尾零：从最后一个非零位起做逐位扩大扰动
                target_idx = -1
                for i in range(L - 1, -1, -1):
                    if new_digits[i] != '0':
                        target_idx = i
                        break
                if target_idx >= 0:
                    apply_progressive_perturb(target_idx, max_steps=2)

            # 保护首位不为0（若原首位非0）
            if new_digits and digits and digits[0] != '0' and new_digits[0] == '0':
                new_digits[0] = digits[0]

            return fill_back(new_digits)

        def build_variant(src: str) -> str:
            # 替换所有数字片段；若原文含货币符号/代码/单位则自然保留，若原文无则不新增
            def repl(m: re.Match) -> str:
                return mutate_number_like(m.group(0))
            return num_pat.sub(repl, src)

        candidates: List[str] = []
        attempts = max(10, k * 5)
        seen = set([s])
        while len(candidates) < k and attempts > 0:
            attempts -= 1
            new_text = build_variant(s)
            if new_text not in seen and new_text.strip() != s.strip():
                seen.add(new_text)
                candidates.append(new_text)

        return [(c, 1.0) for c in candidates[:k]]

    # ------------------------------------------------------------------
    # 规则映射：通用数字类实体（PERCENT/CARDINAL/ORDINAL/QUANTITY/AGE/PHONE）
    # ------------------------------------------------------------------
    def _generate_numeric_like_candidates(self, word_text: str, word_type: str, k: int) -> List[Tuple[str, float]]:
        """对几乎纯数字的实体进行轻量扰动：
        - PERCENT: 保持%与小数位，调整数值（如 42% -> 38%/47%/41.5%）
        - CARDINAL/QUANTITY: 保持分隔符与单位，调整末位或最后一个非0位；
        - ORDINAL: 小范围调整级次（1st/2nd/3rd/4th...）；
        - AGE: 小范围±1~±3；
        - PHONE: 只调整末尾2-3位，保留分隔符与国家/区号；
        """
        import re
        s = word_text
        if not s:
            return []

        # 工具：提取并替换所有数字子串，同时保持非数字原位
        num_pat = re.compile(r"\d+")

        def apply_percent(val_str: str) -> List[str]:
            # 允许1-2位小数，范围裁剪到0..100
            try:
                v = float(val_str)
            except ValueError:
                return []
            deltas = [-5, -3, -2, -1, 1, 2, 3, 5]
            outs = []
            for d in random.sample(deltas, min(len(deltas), max(3, k))):
                nv = max(0.0, min(100.0, v + d))
                if '.' in val_str and len(val_str.split('.')[-1]) >= 2:
                    outs.append(f"{nv:.2f}")
                elif '.' in val_str:
                    outs.append(f"{nv:.1f}")
                else:
                    outs.append(str(int(round(nv))))
            return outs

        def vary_last_non_zero(digits: List[str]) -> List[str]:
            out = list(digits)
            idx = None
            for i in range(len(out)-1, -1, -1):
                if out[i] != '0':
                    idx = i
                    break
            if idx is None:
                out[-1] = '1'
            else:
                old = int(out[idx])
                choices = []
                if old < 9:
                    choices.append(old + 1)
                if old > 1 or idx != 0:
                    choices.append(old - 1)
                if choices:
                    out[idx] = str(random.choice(choices))
            return out

        def replace_with_variants(text: str, gen_variants_fn) -> List[str]:
            # 对首个数字片段生成多个变体再回填，其他片段不变（轻量规则）
            m = num_pat.search(text)
            if not m:
                return []
            start, end = m.span()
            core = m.group(0)
            variants = gen_variants_fn(core)
            outs = []
            for v in variants:
                outs.append(text[:start] + v + text[end:])
            return outs

        results: List[str] = []
        wt = (word_type or '').upper()

        if wt == 'PERCENT' and ('%' in s or 'percent' in s.lower()):
            # 仅变更数值，不改百分号/单词
            def gen(core: str) -> List[str]:
                return apply_percent(core)
            results = replace_with_variants(s, gen)
        elif wt in ['CARDINAL', 'QUANTITY']:
            # 若整段文本为“纯数字+常见分隔符”，对整段数字做多位扰动；否则仅扰动首个数字片段的最后一个非0位
            pure_numeric_like = re.sub(r"[0-9\s,\.]", "", s) == ""

            if pure_numeric_like:
                digit_positions = [i for i, ch in enumerate(s) if ch.isdigit()]
                if not digit_positions:
                    return []

                first_digit_idx = digit_positions[0]

                def rnd_variant() -> str:
                    chars = list(s)
                    D = len(digit_positions)
                    # 随机扰动比例 30%~70%
                    import math as _math
                    cnt = max(1, min(D, int(_math.ceil(D * random.uniform(0.3, 0.7)))))
                    chosen = set(random.sample(digit_positions, cnt))
                    for idx in digit_positions:
                        if idx in chosen:
                            old = chars[idx]
                            if idx == first_digit_idx and s[first_digit_idx] != '0':
                                pool = [str(d) for d in range(1, 10) if str(d) != old]
                            else:
                                pool = [str(d) for d in range(0, 10) if str(d) != old]
                            chars[idx] = random.choice(pool)
                    return ''.join(chars)

                outs: List[str] = []
                seen = set([s])
                attempts = max(10, k * 6)
                while len(outs) < k and attempts > 0:
                    attempts -= 1
                    v = rnd_variant()
                    if v not in seen and v != s:
                        seen.add(v)
                        outs.append(v)
                results = outs
            else:
                # 仅改变第一个数字片段的最后一个非0位（保留单位/分隔）
                def gen(core: str) -> List[str]:
                    digs = list(core)
                    all_outs = []
                    for _ in range(max(6, k*2)):
                        cand = vary_last_non_zero(digs)
                        val = ''.join(cand)
                        if val != core and val not in all_outs:
                            all_outs.append(val)
                        if len(all_outs) >= k:
                            break
                    return all_outs
                results = replace_with_variants(s, gen)
        elif wt == 'ORDINAL':
            # 识别 1st/2nd/3rd/4th... 小范围+/-1
            ord_pat = re.compile(r"(\d+)(st|nd|rd|th)", re.IGNORECASE)
            m = ord_pat.search(s)
            if m:
                n = int(m.group(1))
                suf = m.group(2)
                cand_ns = [max(1, n + d) for d in [-2, -1, 1, 2]]
                def suffix_of(x: int) -> str:
                    x_mod = x % 100
                    if 11 <= x_mod <= 13:
                        return 'th'
                    last = x % 10
                    return 'st' if last == 1 else 'nd' if last == 2 else 'rd' if last == 3 else 'th'
                for nn in cand_ns:
                    results.append(ord_pat.sub(f"{nn}" + suffix_of(nn), s, count=1))
        elif wt == 'AGE':
            # 年龄：±1~±3
            def gen(core: str) -> List[str]:
                try:
                    v = int(core)
                except ValueError:
                    return []
                outs = []
                for d in [-3,-2,-1,1,2,3]:
                    nv = max(0, v + d)
                    outs.append(str(nv))
                return outs
            results = replace_with_variants(s, gen)
        elif wt == 'PHONE':
            # 电话：跨全串定位最后2-3个数字并扰动，保留分隔符/国家区号/括号等
            digit_positions = [i for i, ch in enumerate(s) if ch.isdigit()]
            if len(digit_positions) > 2:
                outs: List[str] = []
                seen = set([s])
                attempts = max(10, k * 4)
                while len(outs) < k and attempts > 0:
                    attempts -= 1
                    chars = list(s)
                    tail = 3 if len(digit_positions) >= 3 and random.random() < 0.6 else 2
                    for pos in digit_positions[-tail:]:
                        chars[pos] = str(random.randint(0, 9))
                    v = ''.join(chars)
                    if v not in seen and v != s:
                        seen.add(v)
                        outs.append(v)
                results = outs
            else:
                results = []
        elif wt in ['ID', 'KEY', 'IDENTIFIER']:
            # 纯数字键名/标识：随机化整串中所有数字，保留字母与符号
            digit_positions = [i for i, ch in enumerate(s) if ch.isdigit()]
            if not digit_positions:
                return []
            outs: List[str] = []
            seen = set([s])
            attempts = max(20, k * 8)
            while len(outs) < k and attempts > 0:
                attempts -= 1
                chars = list(s)
                for pos in digit_positions:
                    old = chars[pos]
                    pool = [str(d) for d in range(10) if str(d) != old]
                    chars[pos] = random.choice(pool)
                v = ''.join(chars)
                if v not in seen and v != s:
                    seen.add(v)
                    outs.append(v)
            results = outs

        if not results:
            return []
        uniq: List[str] = []
        seen = set()
        for t in results:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return [(t, 1.0) for t in uniq[:k]]

    # ------------------------------------------------------------------
    # 规则映射：统一入口（日期/金额）
    # ------------------------------------------------------------------
    def _generate_rule_based_candidates(self, word_text: str, word_type: str, k: int) -> List[Tuple[str, float]]:
        """统一的规则候选入口。
        优先级：金额 > 日期（星期与月份）。
        命中任一类别则返回对应规则候选；都未命中返回空列表。
        """
        wt = (word_type or '').upper()
        # 针对 MONEY
        if wt in ['MONEY']:
            money = self._generate_money_candidates(word_text, k)
            if money:
                return money
        # 针对其他数字类实体：PERCENT, CARDINAL, ORDINAL, QUANTITY, AGE, PHONE
        if wt in ['PERCENT', 'CARDINAL', 'ORDINAL', 'QUANTITY', 'AGE', 'PHONE', 'ID', 'IDENTIFIER', 'KEY']:
            numeric = self._generate_numeric_like_candidates(word_text, wt, k)
            if numeric:
                return numeric
        # 日期/星期/月规则
        week_month = self._generate_week_month_candidates(word_text, k)
        if week_month:
            return week_month
        return []


def test_simple_enhanced_generator():
    """测试简单增强的生成器"""
    import time
    from semantic_sampler import GenerativeModelEmbeddingProvider, SemanticSampler
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    
    print("=== 测试简单增强的多目标词生成器 ===")
    
    # 配置
    GENERATION_MODEL_NAME = "Qwen/Qwen3-0.6B"
    
    # 加载模型（简化版，实际使用时应该从main模块导入）
    print(f"加载模型: {GENERATION_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME, device_map="cuda")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL_NAME, device_map="cuda", torch_dtype=torch.float16)
    model.eval()
    
    # 初始化嵌入提供者和采样器
    from sentence_transformers import SentenceTransformer
    from semantic_sampler import DedicatedEmbeddingModelProvider
    EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    embedding_provider = DedicatedEmbeddingModelProvider(model_name=EMBEDDING_MODEL_NAME, generation_tokenizer=tokenizer, precompute_knn=True, knn_top_k=256, knn_batch_size=512)
    # embedding_provider = GenerativeModelEmbeddingProvider(model=embedding_model, tokenizer=tokenizer)
    sampler = SemanticSampler(
        tokenizer=tokenizer,
        model=model,
        embedding_provider=embedding_provider,
        temperature=1.0,
        max_batch_size=32,
        semantic_top_k=200,
    )
    
    # 创建简单增强生成器
    generator = SimpleEnhancedGenerator(sampler)
    
    # 测试案例
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
        }
        ,
        {
            "name": "星期与月份案例",
            "text": "By . <redacted> Anthony Bond (PERSON) </redacted> . PUBLISHED: . <redacted> 08:20 EST, 7 February 2013 (TIME) </redacted> . | . UPDATED: . <redacted> 08:30 EST, 7 February 2013 (TIME) </redacted> . A teenage soap star today denied raping and molesting a 14-year-old boy during a three-month campaign of sexual abuse. The 17-year-old is accused of forcing the boy to perform oral sex on him at a south London theatre. The TV actor, who cannot be named for legal reasons, is also accused of fondling his victim in the changing rooms of a London stage school and in nearby public toilets. Accused: A teenage soap star today denied raping and molesting a 14-year-old boy during a three-month campaign of sexual abuse. He appeared at <redacted> Blackfriars Crown Court (ORG) </redacted> . The alleged attacks took place while the pair attended the same stage school <redacted> between July and September 2010 (DATE) </redacted>. The teenager appeared this morning at <redacted> Blackfriars Crown Court (ORG) </redacted> to formally enter not guilty pleas to one count of rape and three counts of sexual assault. Wearing a grey jacket with brown leather elbow patches, skinny blue jeans with turn-ups, and an open collar white shirt, the actor spoke to confirm his name and say 'not guilty' four times. Judge <redacted> John Hillen (PERSON) </redacted> set the trial date for <redacted> 18 March (DATE) </redacted> and released the teenager on bail until the start of the two-week trial. 'Today's hearing was to find out whether you plead guilty or not guilty, and you pleaded not guilty', he said. 'You are released on bail, and if you fail to attend for trial it will take in your absence.' The teenager from Surrey pleaded not guilty to rape and sexual assault in <redacted> July 2010 (DATE) </redacted> and two counts of sexual assault in <redacted> September 2010 (DATE) </redacted>. Sorry we are unable to accept comments for legal reasons.",
            "target_samples": 12
        }
        ,
        {
            "name": "金融案例",
            "text": "Formal Report on the Financial and Legal Implications of the International Business Trip to Veridian City\n\nPrepared by: <redacted> Evelyn Harper (PERSON) </redacted>\nDate: <redacted> June 12, 2024 (DATE) </redacted>\n\nIntroduction:\nThis report provides a comprehensive overview of the financial expenditures, travel arrangements, and potential legal considerations involved in the recent international business trip undertaken by Mr. <redacted> Jonathan Pierce (PERSON) </redacted>, CFO of <redacted> Alpine Technologies Inc. (ORG) </redacted>, to Veridian City, Veridia Republic. The trip took place from <redacted> April 15 (DATE) </redacted> to <redacted> April 28, 2024 (DATE) </redacted>, with the primary purpose of negotiating a joint venture agreement with Veridian-based firm NovaCorps Ltd.\n\n1. Financial Overview:\nThe total budget allocated for the trip was <redacted> $18,750.00 (MONEY) </redacted>, covering airfare, accommodation, local transportation, meals, and incidental expenses. Airfare was booked through Global Air Services under reservation number GA-98475321, at a cost of <redacted> $3,200.00 (MONEY) </redacted> for a round-trip business class ticket departing from the company headquarters in Boston (BOS) to Veridian City International Airport (VCI).\n\nAccommodation was arranged at the Grand Veridian Hotel, located at 45 Riverbend Avenue, Veridian City. The hotel stay lasted 12 nights, with a nightly rate of <redacted> $220.00 (MONEY) </redacted>, incurring a total expense of <redacted> $2,640.00 (MONEY) </redacted>. Local transportation expenses, including taxi rides and car rentals, amounted to <redacted> $580.00 (MONEY) </redacted>. Meals and other expenses were reimbursed upon submission of receipts, totaling <redacted> $1,150.00 (MONEY) </redacted>.\n\nAn unanticipated expenditure of <redacted> $1,000.00 (MONEY) </redacted> was incurred due to the expedited visa processing fees required after submission delays. The visa, issued under case number VR-2024-5992, was essential for entry into Veridia Republic and was handled by the legal affairs department.\n\n2. Travel Arrangements and Compliance:\nAll travel arrangements complied with Alpine Technologies’ internal travel policies, which mandate economy-class travel for trips under 10 hours and business class for longer flights, adherence to per diem meal allowances, and use of preferred vendors for accommodation and transport services. The trip itinerary was submitted and approved by the finance department on <redacted> March 10, 2024 (DATE) </redacted>.\n\nHowever, it was noted that the visa application process was initiated late by the employee, resulting in additional legal fees and expedited processing charges. This delay could have jeopardized the planned negotiation schedule.\n\n3. Legal Considerations:\nDuring the trip, legal counsel identified several clauses within the proposed joint venture agreement that require further review. Notably, the non-compete clause restricts Alpine Technologies from engaging with competing firms within Veridia for a period of three years. Additionally, the arbitration clause mandates dispute resolution exclusively through Veridia’s commercial courts, which may present challenges given differences in legal frameworks.\n\nThe legal affairs team has recommended renegotiation of these provisions to include more neutral arbitration venues and clearer definitions of competitive activities. Furthermore, compliance with Veridian export regulations concerning technology transfer was emphasized to avoid potential sanctions.\n\n4. Recommendations:\n- Implement earlier initiation of visa applications for future international travel to mitigate expedited processing costs.\n- Conduct thorough pre-trip legal reviews of contractual documents to identify and address restrictive clauses prior to negotiation.\n- Consider budgeting additional contingency funds for unforeseen legal expenses.\n- Evaluate alternative arbitration venues acceptable to both parties to minimize jurisdictional risks.\n\nConclusion:\nThe international business trip to Veridian City was successful in advancing Alpine Technologies’ strategic partnership goals, albeit with some financial and legal challenges. By addressing the highlighted issues, future trips can be managed more efficiently, reducing unnecessary costs and legal risks.\n\nAttachments:\n- Expense receipts and reimbursement forms\n- Visa application documentation\n- Draft joint venture agreement excerpt\n\nPrepared by:\n<redacted> Evelyn Harper (PERSON) </redacted>\nCorporate Compliance Officer\n<redacted> Alpine Technologies Inc. (ORG) </redacted>",
            "target_samples": 12
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试案例 {i}: {test_case['name']}")
        
        # 生成样本
        start_time = time.time()
        samples, real_to_fake_mapping = generator.generate_samples(
            test_case['text'],
            target_samples=test_case['target_samples'],
            num_intervals=2,
            dist='rel',
            redundancy_factor=1.2,  # 使用冗余
            exclude_reference=True  # 排除参考样本
        )
        elapsed = time.time() - start_time
        
        print(f"生成时间: {elapsed:.2f} 秒, 共生成 {len(samples)} 条伪样本")
        for j, (txt, prob) in enumerate(samples[:10], 1):
            print(f"  {j}. {txt} (概率: {prob:.5f})")
        
        print("\n真词到伪词映射：")
        for real_word, fake_words in real_to_fake_mapping.items():
            print(f"  '{real_word}' -> {fake_words}")


if __name__ == "__main__":
    test_simple_enhanced_generator() 
