from typing import List, Dict, Tuple
import re

def load_jsonl(file_path: str, stream: bool = False):
    """加载JSONL文件"""
    import json
    
    if stream:
        def generator():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        return generator()
    else:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

def save_jsonl(data: List[Dict], file_path: str):
    """保存数据到JSONL文件"""
    import json
    import os
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def split_string_to_words(text: str) -> List[str]:
    """将字符串分割成单词列表，包含标点符号，特别处理<PII>标记"""
    if not text:
        return []
    # 优先匹配<PII>标记，然后匹配普通单词和标点符号
    pattern = r'<PII>|\b\w+\b|[<>=/!@#$%^&*()?":{}|\\`~;_+-]'
    try:
        return re.findall(pattern, text)
    except TypeError:
        return []
    
def extract_sensitive_info(original_text: str, masked_text: str) -> Tuple[List[str], List[str]]:
    """
    从原始文本和masked文本中提取敏感信息
    使用完整词汇分词（包含标点符号），以词为单位进行敏感信息标记
    
    Args:
        original_text: 原始文本
        masked_text: 包含<PII>标记的文本
        
    Returns:
        Tuple[List[str], List[str]]: (原始词列表, 敏感词列表)
    """
    # 使用包含标点的分词方式，以完整词汇为单位
    orig_words = split_string_to_words(original_text)
    mask_words = split_string_to_words(masked_text)
    
    # 移除<PII>标记，得到非敏感词列表
    non_sensitive_words = [word for word in mask_words if word != "<PII>"]
    
    # 提取敏感词
    sensitive_words = []
    
    # 简单的贪心匹配算法
    non_sensitive_idx = 0
    for i, orig_word in enumerate(orig_words):
        if (non_sensitive_idx < len(non_sensitive_words) and 
            orig_word.lower() == non_sensitive_words[non_sensitive_idx].lower()):
            # 匹配到非敏感词，跳过
            non_sensitive_idx += 1
        else:
            # 未匹配到，为敏感词
            sensitive_words.append(orig_word)
    
    return orig_words, sensitive_words