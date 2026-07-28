# MicroGPT 中文姓名版

這是 Karpathy MicroGPT 的純 Python 教學實作，使用字元級 tokenizer 學習並生成中文姓名。

## 執行

```bash
uv sync
uv run python microgpt.py
```

程式會讀取 `input.txt`，使用兩層 Transformer 訓練 5,000 steps，將權重與優化器狀態存入本機暫存的 `checkpoint.pkl`，接著逐字生成 20 個新姓名。模型每 1,000 steps 更新 checkpoint，每 5,000 steps 顯示全資料平均 loss 與取樣結果。

指定總訓練步數、從中斷的 checkpoint 繼續，或只做 inference：

```bash
uv run python microgpt.py --steps 15000 --checkpoint checkpoints/my-experiment.pkl
uv run python microgpt.py --resume --steps 15000 --checkpoint checkpoints/my-experiment.pkl
uv run python microgpt.py --inference --checkpoint checkpoints/jinyong-2l-e16-h4-15k-seed42.pkl --temperature 0.6 --top-p 0.9
```

`--checkpoint` 可讓不同實驗使用不同檔案；未指定時仍使用 Git 忽略的根目錄 `checkpoint.pkl`。`--resume` 僅用於未達原 target steps 的中斷訓練，並且 `--steps` 必須與 checkpoint 的原 target 一致；已完成的 checkpoint 不會藉此延長訓練。`--top-p 1.0` 代表不篩掉低機率字元，與原本的完整機率分布取樣相同。請只載入自己信任的 pickle checkpoint。

## Git LFS 模型檔

`checkpoints/*.pkl` 由 Git LFS 追蹤。新的 clone 或 worktree 若尚未取得權重，請執行：

```bash
git lfs install
git lfs pull
```

每個要保留的實驗應使用不同的描述性檔名，不要覆寫已追蹤的 checkpoint。目前模型的完整參數、結果與後續實驗範本請見 [`EXPERIMENTS.md`](EXPERIMENTS.md)。

## 資料

`input.txt` 包含 723 筆金庸 15 部小說中的人物本名，參考 [WuxiaSociety 的分作品角色表](https://wuxiasociety.com/jin-yong-characters/)與中文維基百科角色列表整理。名單統一使用臺灣繁體中文，排除稱謂、法號、綽號、化名及排行稱呼，並跨作品去重；格式為 UTF-8 編碼、每行一個名稱。你可以直接替換成自己的資料，但建議保持字典在數百到約一千個字內，因為這個純 Python 實作的輸出層成本會隨字典大小增加。
