import random
import re
import unicodedata
from typing import Dict, List, Tuple

from progressive_generator import SensitiveWord


class EmbeddingSimpleGenerator:
    """
    基于嵌入近邻的极简基线：
    - 仅依赖 embedding 提供的近邻，不做概率筛选
    - 不包含缓存、规则映射或复杂后处理
    """

    def __init__(
        self,
        tokenizer,
        embedding_provider,
        neighbor_pool_size: int = 128,
        exclude_top_ratio: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.embedding_provider = embedding_provider
        self.neighbor_pool_size = max(1, neighbor_pool_size)
        self.exclude_top_ratio = max(0.0, min(exclude_top_ratio, 0.9))
        self._normalized_ner_text = None

    # ------------------------------------------------------------------
    # NER 文本解析
    # ------------------------------------------------------------------
    def parse_ner_text(self, ner_text: str) -> Tuple[str, List[SensitiveWord]]:
        pattern = re.compile(r'<redacted>(\s*)([^(]+?)(\s*)\(([^)]+)\)(\s*)</redacted>')

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

        sensitive_words: List[SensitiveWord] = []
        for (word, word_type), positions in word_info.items():
            uid = f"{word}|{word_type}"
            sensitive_words.append(
                SensitiveWord(
                    word=word,
                    word_type=word_type,
                    positions=positions,
                    uid=uid,
                )
            )

        template = pattern.sub('<redacted>', normalized_ner_text)
        template = re.sub(r'\s+', ' ', template).strip()

        return template, sensitive_words

    # ------------------------------------------------------------------
    # 嵌入近邻候选
    # ------------------------------------------------------------------
    def _get_neighbors(self, token_id: int) -> List[int]:
        try:
            neighbors = self.embedding_provider.get_similar_tokens(
                token_id, top_k=self.neighbor_pool_size
            )
        except Exception:
            neighbors = []

        filtered = [tid for tid in neighbors if tid != token_id]
        if filtered:
            cut = int(len(filtered) * self.exclude_top_ratio)
            if cut >= len(filtered):
                cut = 0
            filtered = filtered[cut:]
        return filtered or [token_id]

    def _sample_candidate_text(self, token_ids: List[int]) -> str:
        if not token_ids:
            return ""
        sampled_ids = []
        for tid in token_ids:
            neighbors = self._get_neighbors(tid)
            sampled_ids.append(random.choice(neighbors))
        decoded = self.tokenizer.decode(sampled_ids, skip_special_tokens=True)
        return decoded.strip()

    def _generate_word_candidates(
        self,
        word: SensitiveWord,
        target_samples: int,
        redundancy_factor: float,
    ) -> List[Tuple[str, float]]:
        attempts = max(1, int(target_samples * redundancy_factor))
        token_ids = self.tokenizer.encode(word.word, add_special_tokens=False)
        if not token_ids:
            return [(word.word, 1.0)]

        candidates = []
        seen = set()

        for _ in range(attempts * 4):
            candidate = self._sample_candidate_text(token_ids)
            if not candidate:
                continue
            if candidate.lower() == word.word.lower():
                continue
            if candidate not in seen:
                seen.add(candidate)
                candidates.append((candidate, 1.0))
            if len(candidates) >= attempts:
                break

        if not candidates:
            candidates = [(word.word, 1.0)]

        if len(candidates) < target_samples:
            expanded = list(candidates)
            idx = 0
            while len(expanded) < target_samples and expanded:
                expanded.append(expanded[idx % len(expanded)])
                idx += 1
            candidates = expanded
        else:
            candidates = candidates[:target_samples]

        return candidates

    # ------------------------------------------------------------------
    # 样本生成主流程
    # ------------------------------------------------------------------
    def generate_samples(
        self,
        ner_text: str,
        target_samples: int = 20,
        num_intervals: int = 10,  # 与其它接口保持一致
        dist: str = 'rel',
        redundancy_factor: float = 1.2,
        exclude_reference: bool = True,
    ) -> Tuple[List[Tuple[str, float]], Dict[str, List[str]]]:
        del num_intervals, dist  # 未使用，保持接口兼容

        template, sensitive_words = self.parse_ner_text(ner_text)
        ner_text = getattr(self, '_normalized_ner_text', ner_text)

        if not sensitive_words:
            return [(template, 1.0)], {}

        position_to_word: Dict[int, str] = {}
        real_words: List[str] = []
        for sw in sensitive_words:
            for pos in sw.positions:
                position_to_word[pos] = sw.word
                if len(real_words) <= pos:
                    real_words.extend([''] * (pos + 1 - len(real_words)))
                real_words[pos] = sw.word

        total_positions = len(real_words)

        generated_candidates: Dict[int, List[Tuple[str, float]]] = {}

        for position in range(total_positions):
            target_word = None
            for sw in sensitive_words:
                if position in sw.positions:
                    target_word = sw
                    break
            if target_word is None:
                continue

            candidates = self._generate_word_candidates(
                target_word,
                target_samples=target_samples,
                redundancy_factor=redundancy_factor,
            )

            if exclude_reference:
                candidates = [(w, p) for (w, p) in candidates if w.strip() != target_word.word.strip()]
                if not candidates:
                    candidates = [(target_word.word, 1.0)]

            generated_candidates[position] = candidates[:target_samples] or [(target_word.word, 1.0)]

        if not generated_candidates:
            return [(template, 1.0)], {}

        redacted_pat = re.compile(r'<redacted>.*?</redacted>', re.DOTALL)
        final_samples: List[Tuple[str, float]] = []
        real_to_fake_mapping: Dict[str, List[str]] = {}

        for i in range(target_samples):
            current_sentence = ner_text
            total_prob = 1.0

            for position in range(total_positions):
                if position not in generated_candidates:
                    continue
                candidate_list = generated_candidates[position]
                idx = i % len(candidate_list)
                fake_word, word_prob = candidate_list[idx]

                real_word = position_to_word.get(position, '')
                if real_word.strip():
                    real_to_fake_mapping.setdefault(real_word, [])
                    if fake_word not in real_to_fake_mapping[real_word]:
                        real_to_fake_mapping[real_word].append(fake_word)

                current_sentence, _ = redacted_pat.subn(
                    lambda _m, rep=fake_word: rep,
                    current_sentence,
                    count=1,
                )
                total_prob *= word_prob

            final_samples.append((current_sentence, total_prob))

        final_samples.sort(key=lambda x: x[1], reverse=True)
        return final_samples, real_to_fake_mapping

