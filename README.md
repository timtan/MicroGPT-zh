# MicroGPT 中文姓名版

這是 Karpathy MicroGPT 的純 Python 教學實作，使用字元級 tokenizer 學習並生成中文姓名。

## 執行

```bash
uv sync
uv run python microgpt.py
```

程式會讀取 `input.txt`，訓練 1,000 steps，接著逐字生成 20 個新姓名。

## 資料

`input.txt` 包含 500 筆為教學用途組合出的合成中文姓名，不包含真實人物名單或個人資料。格式是一行一個姓名，UTF-8 編碼。你可以直接替換成自己的資料，但建議保持字典在數百到約一千個字內，因為這個純 Python 實作的輸出層成本會隨字典大小增加。
