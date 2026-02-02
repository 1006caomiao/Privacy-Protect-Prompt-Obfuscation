import random
import re
from types import SimpleNamespace
from typing import Dict, List, Set, Tuple

from progressive_generator import SensitiveWord, SimpleEnhancedGenerator


class SemanticNeighborRandomGenerator(SimpleEnhancedGenerator):
    """
    基于嵌入语义近邻的随机基线生成器。
    不依赖语言模型进行概率筛选，直接从 embedding 空间选取候选。
    """

    def __init__(
        self,
        sampler,
        neighbor_pool_size: int = 256,
        exclude_top_ratio: float = 0.1,
        top_k_ratio: float = 0.4,
    ):
        super().__init__(sampler=sampler, top_k_ratio=top_k_ratio, beam_size=1)
        self.embedding_provider = getattr(sampler, "embedding_provider", None)
        if self.embedding_provider is None:
            raise ValueError("sampler 必须提供 embedding_provider")
        self.tokenizer = getattr(sampler, "tokenizer", None)
        if self.tokenizer is None:
            raise ValueError("sampler 必须提供 tokenizer")
        self.neighbor_pool_size = max(1, neighbor_pool_size)
        self.exclude_top_ratio = max(0.0, min(exclude_top_ratio, 0.9))
        self._neighbor_cache: Dict[int, List[int]] = {}

    def _get_semantic_neighbors(self, token_id: int) -> List[int]:
        if token_id in self._neighbor_cache:
            return self._neighbor_cache[token_id]
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
        if not filtered:
            filtered = [token_id]
        self._neighbor_cache[token_id] = filtered
        return filtered

    def _build_semantic_neighbor_candidates(self, word_text: str, max_candidates: int) -> List[str]:
        token_ids = self.tokenizer.encode(word_text, add_special_tokens=False)
        if not token_ids:
            return []
        candidates: List[str] = []
        seen: Set[str] = set()
        attempts = max(16, max_candidates * 5)
        while len(candidates) < max_candidates and attempts > 0:
            attempts -= 1
            new_ids = []
            for token_id in token_ids:
                neighbors = self._get_semantic_neighbors(token_id)
                choice = random.choice(neighbors) if neighbors else token_id
                new_ids.append(choice)
            decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            cleaned = decoded.strip()
            if not cleaned:
                continue
            if cleaned.lower() == word_text.lower():
                continue
            if cleaned not in seen:
                seen.add(cleaned)
                candidates.append(cleaned)
        return candidates

    def generate_for_position(
        self,
        word: SensitiveWord,
        position: int,
        template: str,
        target_samples: int,
        num_intervals: int = 10,
        dist: str = 'rel',
        redundancy_factor: float = 1.3,
        exclude_reference: bool = True,
    ) -> List[Tuple[str, float]]:
        redundant_target = int(target_samples * redundancy_factor)
        all_candidates = None
        root = self._get_root(word.word)
        root_valid = len(root) >= 3
        independent_only: List[Tuple[str, float]] = []

        if word.uid in self.word_to_fakes:
            cached = self.word_to_fakes[word.uid][:target_samples]
            all_candidates = [(w, 1.0) for (w, _p) in cached]

        if all_candidates is None and root_valid:
            hit_root = next((r for r in self.root_to_fakes if r in root or root in r), None)
            if hit_root:
                base = self.root_to_fakes[hit_root]
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
                self.word_to_fakes[word.uid] = all_candidates
                if root_valid:
                    self.root_to_fakes[root] = list(all_candidates)

        rule_candidates = self._generate_rule_based_candidates(
            word.word, word.word_type, int(redundant_target)
        )
        if rule_candidates:
            all_candidates = rule_candidates
            self.word_to_fakes[word.uid] = list(all_candidates)
            if root_valid:
                self.root_to_fakes[root] = list(all_candidates)
            self.independent_candidates[position] = list(all_candidates)

        if all_candidates is None:
            target_count = max(1, int(redundant_target))
            neighbor_texts = self._build_semantic_neighbor_candidates(word.word, target_count)
            if not neighbor_texts:
                neighbor_texts = [word.word]
            processed: List[Tuple[str, float]] = []
            seen_processed: Set[str] = set()
            for candidate_text in neighbor_texts:
                adjusted = self._apply_casing_by_pattern(word.word, candidate_text)
                cleaned = self._strip_special_if_plain(word.word, adjusted)
                if not cleaned:
                    continue
                if cleaned.lower() == word.word.lower():
                    continue
                if cleaned not in seen_processed:
                    seen_processed.add(cleaned)
                    processed.append((cleaned, 1.0))
            if not processed:
                processed = [(word.word, 1.0)]
            all_candidates = processed
            independent_only = list(all_candidates)
        else:
            if not independent_only:
                independent_only = list(all_candidates)

        if (word.word_type or '').upper() == 'EMAIL':
            try:
                all_candidates = [
                    (
                        self._substitute_email(real_email=word.word, fake_text=word_text),
                        prob,
                    )
                    for (word_text, prob) in all_candidates
                ]
            except Exception:
                pass

        seen_words = set()
        unique_candidates = []
        ref_sample = None
        ref_word = word.word

        for word_text, prob in all_candidates:
            if word_text.strip() == ref_word.strip():
                if not exclude_reference:
                    ref_sample = (word_text, prob)
                continue
            if word_text not in seen_words:
                seen_words.add(word_text)
                unique_candidates.append((word_text, prob))

        unique_candidates.sort(key=lambda x: abs(x[1] - 1.0))

        if not exclude_reference and ref_sample is not None:
            final_candidates = [ref_sample] + unique_candidates
            final_target = target_samples + 1
        else:
            final_candidates = unique_candidates
            final_target = target_samples

        if final_target > 0 and len(final_candidates) == 0:
            final_candidates = [(ref_word, 1.0)]

        if len(final_candidates) < final_target:
            extended = list(final_candidates)
            needed = final_target - len(extended)
            if not exclude_reference and ref_sample is not None:
                rep_pool = unique_candidates if len(unique_candidates) > 0 else extended[1:]
            else:
                rep_pool = final_candidates
            if not rep_pool:
                rep_pool = extended
            if rep_pool:
                times = (needed + len(rep_pool) - 1) // len(rep_pool)
                extended.extend((rep_pool * times)[:needed])
            all_candidates = extended[:final_target]
        else:
            all_candidates = final_candidates[:final_target]

        all_candidates = [
            (self._strip_special_if_plain(word.word, w), p) for (w, p) in all_candidates
        ]

        self.word_to_fakes[word.uid] = list(all_candidates)
        if root_valid:
            self.root_to_fakes[root] = list(all_candidates)

        if independent_only:
            independent_only.sort(key=lambda x: abs(x[1] - 1.0))
            self.independent_candidates[position] = independent_only
        else:
            self.independent_candidates[position] = sorted(
                all_candidates, key=lambda x: abs(x[1] - 1.0)
            )

        self.generated_candidates[position] = all_candidates

        return all_candidates

    def _postprocess_format_consistency(
        self,
        ner_text: str,
        samples: List[Tuple[str, float]],
        real_to_fake_mapping: Dict[str, List[str]],
        real_words_sequence: List[str]
    ) -> List[Tuple[str, float]]:
        """
        后处理：确保所有生成的样本格式与原始NER文本一致
        """
        import re

        PLACEHOLDER_PAT = re.compile(r'<redacted>.*?</redacted>', re.DOTALL)

        reformatted_samples = []
        source_text = getattr(self, '_normalized_ner_text', ner_text)

        for sample_idx, (old_text, prob) in enumerate(samples):
            result = source_text

            for real_word in real_words_sequence:
                fake_list = real_to_fake_mapping.get(real_word, [])

                if not fake_list:
                    replacement = real_word
                else:
                    replacement = fake_list[sample_idx % len(fake_list)]

                replacement = self._apply_casing_by_pattern(real_word, replacement)

                result, _ = PLACEHOLDER_PAT.subn(
                    lambda _m, rep=replacement: rep,
                    result,
                    count=1
                )

            reformatted_samples.append((result, prob))

        return reformatted_samples

    # ------------------------------------------------------------------
    # 词根提取与大小写调整辅助函数
    # ------------------------------------------------------------------

    def _get_root(self, word_text: str) -> str:
        """提取更稳健的词根 (root)"""
        import re

        if not word_text:
            return ""

        word_text = word_text.split('@', 1)[0]
        segments = re.findall(r'[A-Za-z]{3,}', word_text)
        if not segments:
            segments = re.findall(r'[A-Za-z]+', word_text)
            if not segments:
                return ""

        root = ''.join(segments).lower()
        return root

    def _match_casing(self, fake_word: str, real_word: str) -> str:
        if not fake_word:
            return fake_word
        if real_word.isupper():
            return fake_word.upper()
        if real_word.istitle():
            return fake_word.title()
        if real_word[0].isupper() and real_word[1:].islower():
            return fake_word.capitalize()
        return fake_word

    def _apply_casing_by_pattern(self, real_text: str, fake_text: str) -> str:
        """按 real_text 的多词大小写模式对齐 fake_text。"""
        import re

        if not real_text or not fake_text:
            return fake_text

        if self._is_email(real_text):
            return fake_text

        real_tokens = [t for t in re.split(r"\s+", real_text.strip()) if t]
        fake_tokens = [t for t in re.split(r"\s+", fake_text.strip()) if t]

        if not real_tokens or not fake_tokens:
            return fake_text

        N = min(len(fake_tokens), len(real_tokens))
        adjusted = list(fake_tokens)
        for i in range(N):
            adjusted[i] = self._match_casing(fake_tokens[i], real_tokens[i])
        return ' '.join(adjusted)

    # ------------------------------------------------------------
    # 特殊符号处理
    # ------------------------------------------------------------
    def _contains_special(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        import re
        return re.search(r"[^A-Za-z0-9\s]", text or "") is not None

    def _strip_special_chars(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        import re
        cleaned = re.sub(r"[^A-Za-z0-9\s]", "", text)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _strip_special_if_plain(self, real_word: str, fake_text: str) -> str:
        if not isinstance(fake_text, str):
            return fake_text
        if not self._contains_special(real_word) and self._contains_special(fake_text):
            cleaned = self._strip_special_chars(fake_text)
            return cleaned if cleaned else fake_text
        return fake_text

    def _substitute_root(self, real_word: str, real_root: str, fake_root: str) -> str:
        import re

        lower_real = real_word.lower()
        idx = lower_real.find(real_root)

        if idx == -1:
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
                return self._match_casing(fake_root, real_word)
        else:
            before = real_word[:idx]
            root_seg = real_word[idx: idx + len(real_root)]
            after = real_word[idx + len(real_root):]

        base_fake = re.sub(r'\s+', '', fake_root)

        if root_seg.islower():
            base_fake = base_fake.lower()
        elif root_seg.isupper():
            base_fake = base_fake.upper()
        elif root_seg.istitle():
            base_fake = base_fake.title()
        elif root_seg[0].isupper() and root_seg[1:].islower():
            base_fake = base_fake.capitalize()

        sep_tokens = re.findall(r'[^A-Za-z]+', root_seg)
        alpha_tokens = [t for t in re.split(r'[^A-Za-z]+', root_seg) if t]

        if sep_tokens:
            lengths = [len(t) for t in alpha_tokens]
            if sum(lengths) == 0:
                rebuilt_fake = base_fake
            else:
                parts = []
                cursor = 0
                for l in lengths:
                    parts.append(base_fake[cursor:cursor + l])
                    cursor += l
                if cursor < len(base_fake):
                    parts[-1] += base_fake[cursor:]

                rebuilt_fake = parts[0]
                for sep, token in zip(sep_tokens, parts[1:]):
                    rebuilt_fake += sep + token
        else:
            rebuilt_fake = base_fake

        return before + rebuilt_fake + after

    # ------------------------------------------------------------------
    # 自适应替换相关
    # ------------------------------------------------------------------
    def _is_email(self, text: str) -> bool:
        if not isinstance(text, str):
            return False
        import re

        t = text.strip()
        if '@' not in t:
            return False
        if re.search(r'\s', t):
            return False
        local, sep, domain = t.partition('@')
        if not local or not domain:
            return False
        if '.' not in domain:
            return False
        return True

    def _extract_alpha_tokens(self, text: str) -> List[str]:
        import re
        return re.findall(r'[A-Za-z]+', text or '')

    def _substitute_email(self, real_email: str, fake_text: str) -> str:
        if '@' not in real_email:
            return real_email

        import re

        local, domain = real_email.split('@', 1)

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

        new_local = re.sub(r'[A-Za-z]+', repl_alpha, local)

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
        real_tokens = self._extract_alpha_tokens(real_word)
        fake_tokens = self._extract_alpha_tokens(fake_text)

        if not fake_tokens:
            fallback = self._get_root(fake_text)
            return fallback if fallback else fake_text

        if len(real_tokens) <= 1:
            if real_root and hit_root:
                if hit_root.endswith(real_root):
                    return fake_tokens[-1]
                if hit_root.startswith(real_root):
                    return fake_tokens[0]
            return fake_tokens[-1]

        K = min(len(fake_tokens), len(real_tokens))
        return ' '.join(fake_tokens[:K])

    def _substitute_adaptive(self, real_word: str, real_root: str, fake_text: str, hit_root: str) -> str:
        if self._is_email(real_word):
            return self._substitute_email(real_email=real_word, fake_text=fake_text)

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

        K = min(len(fake_tokens), len(real_tokens))
        return self._substitute_root(real_word, real_root, ' '.join(fake_tokens[:K]))

    # ------------------------------------------------------------------
    # 规则映射
    # ------------------------------------------------------------------
    def _generate_week_month_candidates(self, word_text: str, k: int) -> List[Tuple[str, float]]:
        import re

        if not word_text:
            return []

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

        re_weekday_en = re.compile(
            r"\b(Mon(?:\.|day)?|Tue(?:\.|sday)?|Wed(?:\.|nesday)?|Thu(?:\.|rsday)?|Fri(?:\.|day)?|Sat(?:\.|urday)?|Sun(?:\.|day)?)\b",
            re.IGNORECASE,
        )
        re_month_en = re.compile(
            r"\b(Jan(?:\.|uary)?|Feb(?:\.|ruary)?|Mar(?:\.|ch)?|Apr(?:\.|il)?|May|Jun(?:\.|e)?|Jul(?:\.|y)?|Aug(?:\.|ust)?|Sep(?:\.|t(?:\.|ember)?)?|Oct(?:\.|ober)?|Nov(?:\.|ember)?|Dec(?:\.|ember)?)\b",
            re.IGNORECASE,
        )
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
        attempts = max(10, k * 4)
        while len(candidates) < k and attempts > 0:
            attempts -= 1
            new_text = word_text

            if has_en_week:
                chosen_abbr = None

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

            if has_en_month:
                chosen_abbr = None

                def repl_month_en(m: re.Match) -> str:
                    tok = m.group(0)
                    had_dot = tok.endswith('.')
                    core = tok[:-1] if had_dot else tok
                    lower = core.lower()
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

            if has_zh_week:
                day_idx = random.randint(1, 7)

                def repl_week_zh(m: re.Match) -> str:
                    prefix = m.group(1)
                    orig_day = m.group(2)
                    if day_idx == 7:
                        day_char = '天' if orig_day == '天' else '日'
                    else:
                        day_char = "一二三四五六"[day_idx - 1]
                    return prefix + day_char

                new_text = re_weekday_zh.sub(repl_week_zh, new_text)

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

    def _generate_money_candidates(self, word_text: str, k: int) -> List[Tuple[str, float]]:
        import re

        if not word_text:
            return []

        s = word_text

        num_pat = re.compile(r"\d[\d,\.\s]*\d|\d")

        if not num_pat.search(s):
            return []

        def mutate_number_like(num_str: str) -> str:
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

            last_nz = -1
            for i in range(L - 1, -1, -1):
                if new_digits[i] != '0':
                    last_nz = i
                    break
            if last_nz > 0 and new_digits[last_nz] == '5':
                new_digits[last_nz] = '0'

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
                if start_idx < 0:
                    return
                step = max_steps
                pos = start_idx
                while pos >= 0 and step >= 1:
                    if new_digits[pos] != '0':
                        old = int(new_digits[pos])
                        candidates = []
                        for mag in range(1, step + 1):
                            for sign in (-1, 1):
                                nv = old + sign * mag
                                if 0 <= nv <= 9:
                                    if not (pos == 0 and nv == 0 and digits[0] != '0'):
                                        candidates.append(nv)
                        if candidates:
                            nv = random.choice(candidates)
                            new_digits[pos] = str(nv)
                    pos -= 1
                    step -= 1

            if trailing_zeros > 0:
                zero_run_start = L - trailing_zeros
                target_idx = -1
                for i in range(zero_run_start - 1, -1, -1):
                    if new_digits[i] != '0':
                        target_idx = i
                        break
                if target_idx >= 0:
                    apply_progressive_perturb(target_idx, max_steps=3)
                if zero_run_start < L and random.random() < 0.6:
                    new_digits[zero_run_start] = '5'
                elif zero_run_start >= L:
                    new_digits[-1] = '5'
            else:
                target_idx = -1
                for i in range(L - 1, -1, -1):
                    if new_digits[i] != '0':
                        target_idx = i
                        break
                if target_idx >= 0:
                    apply_progressive_perturb(target_idx, max_steps=2)

            if new_digits and digits and digits[0] != '0' and new_digits[0] == '0':
                new_digits[0] = digits[0]

            return fill_back(new_digits)

        def build_variant(src: str) -> str:
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

    def _generate_numeric_like_candidates(self, word_text: str, word_type: str, k: int) -> List[Tuple[str, float]]:
        import re

        s = word_text
        if not s:
            return []

        num_pat = re.compile(r"\d+")

        def apply_percent(val_str: str) -> List[str]:
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
            def gen(core: str) -> List[str]:
                return apply_percent(core)
            results = replace_with_variants(s, gen)
        elif wt in ['CARDINAL', 'QUANTITY']:
            pure_numeric_like = re.sub(r"[0-9\s,\.]", "", s) == ""

            if pure_numeric_like:
                digit_positions = [i for i, ch in enumerate(s) if ch.isdigit()]
                if not digit_positions:
                    return []

                first_digit_idx = digit_positions[0]

                def rnd_variant() -> str:
                    chars = list(s)
                    D = len(digit_positions)
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
            ord_pat = re.compile(r"(\d+)(st|nd|rd|th)", re.IGNORECASE)
            m = ord_pat.search(s)
            if m:
                n = int(m.group(1))
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

    def _generate_rule_based_candidates(self, word_text: str, word_type: str, k: int) -> List[Tuple[str, float]]:
        wt = (word_type or '').upper()
        if wt in ['MONEY']:
            money = self._generate_money_candidates(word_text, k)
            if money:
                return money
        if wt in ['PERCENT', 'CARDINAL', 'ORDINAL', 'QUANTITY', 'AGE', 'PHONE', 'ID', 'IDENTIFIER', 'KEY']:
            numeric = self._generate_numeric_like_candidates(word_text, wt, k)
            if numeric:
                return numeric
        week_month = self._generate_week_month_candidates(word_text, k)
        if week_month:
            return week_month
        return []


class EmbeddingRandomFillGenerator(SemanticNeighborRandomGenerator):
    """
    基于嵌入语义近邻的随机填充基线。
    不调用语言模型筛选，仅从嵌入候选池中随机抽样。
    """

    def __init__(
        self,
        tokenizer,
        embedding_provider,
        neighbor_pool_size: int = 256,
        exclude_top_ratio: float = 0.1,
        random_pool_multiplier: float = 2.0,
    ):
        sampler_stub = SimpleNamespace(
            tokenizer=tokenizer,
            embedding_provider=embedding_provider,
        )
        super().__init__(
            sampler=sampler_stub,
            neighbor_pool_size=neighbor_pool_size,
            exclude_top_ratio=exclude_top_ratio,
            top_k_ratio=1.0,
        )
        self.random_pool_multiplier = max(1.0, random_pool_multiplier)

    def _build_semantic_neighbor_candidates(self, word_text: str, max_candidates: int) -> List[str]:
        # 扩大候选池再随机抽样，提升多样性
        pool_target = max(int(max_candidates * self.random_pool_multiplier), max_candidates)
        base_candidates = super()._build_semantic_neighbor_candidates(word_text, pool_target)
        if len(base_candidates) <= max_candidates:
            return base_candidates
        return random.sample(base_candidates, max_candidates)

