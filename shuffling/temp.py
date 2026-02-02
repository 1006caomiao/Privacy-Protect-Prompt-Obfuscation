import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "/data1/huggingface"
import evaluate
evaluate.load("bleu")
evaluate.load("rouge")
evaluate.load("bertscore")