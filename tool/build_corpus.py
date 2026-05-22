# https://gemini.google.com/app/194e66f1fd209670
import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def format_magicoder_item(item):
    """將 Magicoder 格式轉換為標準的對話格式 (ChatML/OpenAI 風格)"""
    # Magicoder 欄位通常為 'instruction' 和 'response'
    instruction = item.get("instruction", "").strip()
    response = item.get("response", "").strip()
    
    if not instruction or not response:
        return None
        
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response}
        ],
        "source": "magicoder_oss_75k"
    }

def format_gsm8k_item(item):
    """將 GSM8K 數學數據轉換為相同的對話格式，強化模型的邏輯推理能力"""
    question = item.get("question", "").strip()
    answer = item.get("answer", "").strip()
    
    if not question or not answer:
        return None
        
    return {
        "messages": [
            {"role": "user", "content": f"Please solve this math problem step by step:\n{question}"},
            {"role": "assistant", "content": answer}
        ],
        "source": "gsm8k_math"
    }

def save_to_jsonl(data, file_path):
    """將資料儲存為 JSONL 檔案"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"成功儲存: {file_path} (共 {len(data)} 條數據)")

def main():
    output_dir = Path("./processed_corpus")
    output_dir.mkdir(exist_ok=True)
    
    all_corpus = []
    
    # 1. 載入並處理 Magicoder 代碼數據
    print("正在下載並載入 Magicoder-OSS-Instruct-75K...")
    try:
        magicoder_ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        print("開始處理 Magicoder 數據...")
        for item in tqdm(magicoder_ds, desc="Magicoder"):
            formatted = format_magicoder_item(item)
            # 過濾掉太長的文本（可依據你的小模型 Context Length 調整，此處粗估字數）
            if formatted and len(formatted["messages"][0]["content"]) < 4000:
                all_corpus.append(formatted)
    except Exception as e:
        print(f"Magicoder 載入失敗: {e}。請檢查網路或 Hugging Face 連線。")

    # 2. 混合數學數據 (強烈建議混入 5%~10% 的數學來激發代碼小模型的邏輯)
    print("\n正在下載並載入 GSM8K 數學數據集以強化邏輯能力...")
    try:
        gsm8k_ds = load_dataset("openai/gsm8k", "main", split="train")
        for item in tqdm(gsm8k_ds, desc="GSM8K Math"):
            formatted = format_gsm8k_item(item)
            if formatted:
                all_corpus.append(formatted)
    except Exception as e:
        print(f"GSM8K 載入失敗: {e}，跳過數學混合。")

    # 3. 洗牌並切分數據集 (Shuffle & Split)
    print("\n正在進行資料隨機洗牌與切分...")
    random.seed(42)  # 固定隨機種子以確保結果可重現
    random.shuffle(all_corpus)
    
    # 拿 95% 作為訓練，5% 作為驗證
    split_idx = int(len(all_corpus) * 0.95)
    train_data = all_corpus[:split_idx]
    eval_data = all_corpus[split_idx:]
    
    # 4. 輸出檔案
    save_to_jsonl(train_data, output_dir / "train.jsonl")
    save_to_jsonl(eval_data, output_dir / "eval.jsonl")
    
    print("\n=== 語料庫建置完成 ===")
    print(f"總數據量: {len(all_corpus)} 條")
    print(f"檔案儲存於: {output_dir.resolve()}")

if __name__ == "__main__":
    main()