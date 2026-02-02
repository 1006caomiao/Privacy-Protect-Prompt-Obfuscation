#!/usr/bin/env python3
"""
运行敏感样本混淆流水线的示例脚本
"""

import subprocess
import sys
import os

def main():
    """运行敏感样本混淆流水线"""
    
    # 流水线参数配置
    config = {
        # 输入输出文件
        "input_file": "./processed_datasets/synthetic_dataset/synthetic_dataset_gpt-mini_0519_test.jsonl",
        "output_file": "./results/obfuscated_samples_output.jsonl", 
        "log_file": "./results/obfuscation_pipeline.log",
        
        # 处理参数
        "num_records": 50,  # 处理50条记录作为示例
        "device": "cuda",
        
        # 模型参数
        "generation_model": "NousResearch/Llama-2-7b-hf",
        
        # 混淆生成参数
        "num_obfuscated_samples": 10,
        "num_intervals": 10,
        "dist": "rel",
        "redundancy_factor": 1.2,
    }
    
    # 构建命令
    cmd = [
        sys.executable, "privacy_obfuscation_pipeline.py",
        "--input_file", config["input_file"],
        "--output_file", config["output_file"],
        "--log_file", config["log_file"],
        "--num_records", str(config["num_records"]),
        "--device", config["device"],
        "--generation_model", config["generation_model"],
        "--num_obfuscated_samples", str(config["num_obfuscated_samples"]),
        "--num_intervals", str(config["num_intervals"]),
        "--dist", config["dist"],
        "--redundancy_factor", str(config["redundancy_factor"]),
    ]
    
    print("运行敏感样本混淆流水线...")
    print(f"输入文件: {config['input_file']}")
    print(f"输出文件: {config['output_file']}")
    print(f"处理记录数: {config['num_records']}")
    print(f"生成模型: {config['generation_model']}")
    print()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(config["output_file"]), exist_ok=True)
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("流水线执行成功!")
        print("输出:")
        print(result.stdout)
        
        if result.stderr:
            print("警告/错误信息:")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"流水线执行失败，退出码: {e.returncode}")
        print("错误输出:")
        print(e.stderr)
        print("标准输出:")
        print(e.stdout)
        sys.exit(1)
    except Exception as e:
        print(f"运行流水线时出现错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 