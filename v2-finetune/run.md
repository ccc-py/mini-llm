# llm4 v2 -- run.sh

```
(.venv) cccuser@cccimacdeiMac v3 % ./run.sh
=== 開始生成合成訓練資料 (虛擬武林) ===
-> 詞表大小: 112
-> 儲存純文字檔 (pretrain.txt, finetune.txt)...
✅ 武林資料產生完成！
載入 Pretrain 資料，長度: 200160
使用硬體: cpu | 開始 Pre-training...
Pretrain Step    0 | Loss: 4.8902
Pretrain Step  100 | Loss: 0.2659
Pretrain Step  200 | Loss: 0.2379
Pretrain Step  300 | Loss: 0.2271
Pretrain Step  400 | Loss: 0.2221
Pretrain Step  500 | Loss: 0.2271
Pretrain Step  599 | Loss: 0.2223
預訓練完成！模型已儲存為 pretrain.pt
載入 Finetune 資料，長度: 50138
成功載入 pretrain.pt 權重！
使用硬體: cpu | 開始 Fine-tuning...
Finetune Step    0 | Loss: 6.4075
Finetune Step  100 | Loss: 0.3357
Finetune Step  200 | Loss: 0.2328
Finetune Step  299 | Loss: 0.2092
微調完成！模型已儲存為 finetune.pt

==================================================
測試對話 (自動抓取訓練集第一句進行測試)
==================================================
📝 抽取到的題目: <Q>令狐沖的武功是什麼？<A>
🎯 預期的解答: 獨孤九劍
--------------------------------------------------
🤖 AI 實際輸出:
<Q>令狐沖的武功是什麼？<A>獨孤九劍
<Q>段譽在哪裡練武？<A>大理
<Q>什麼武功可以剋制玄冥神掌？<A>九陽神功
<Q>一陽指？<A>六脈神劍
<Q>楊過辟邪劍法華山
<Q>去哪裡可以找到楊過？<A>絕情谷
<Q>九陽神功
==================================================
```
