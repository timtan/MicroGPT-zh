# 訓練實驗紀錄

這份文件記錄可重現的訓練參數、checkpoint 與結果。每次要保留的新實驗應新增一節，並將權重存成新的 `checkpoints/<experiment-id>.pkl`，不要覆寫舊檔。

## jinyong-2l-e16-h4-15k-seed42

### 身分與檔案

| 欄位 | 值 |
|---|---|
| 狀態 | 已完成 15,000 steps，可純 inference；目前不可以 `--resume` 延長 target |
| 訓練日期 | 2026-07-27（Asia/Taipei） |
| Checkpoint | `checkpoints/jinyong-2l-e16-h4-15k-seed42.pkl` |
| Git 儲存 | Git LFS |
| 檔案大小 | 834,844 bytes（約 815 KiB） |
| SHA-256 | `840734a52b687e8a17a8b8aea06b9ddad620ee31451ef9d007cc652a96b18182` |
| Dataset SHA-256 | `6f534cba7a39884405c89c8f7de68a9b927417ff299bcf18accae36d6284e33c` |
| 環境 | Python 3.13.13、uv 0.11.18、Darwin arm64 |

### 資料與模型

| 類別 | 參數 |
|---|---|
| 資料 | 723 筆金庸 15 部小說人物本名，UTF-8、去重，排除稱謂、法號、綽號、化名與排行稱呼 |
| Tokenizer | 字元級，760 個資料字元 + BOS，vocab size 761 |
| Sequence | 最長名稱 5 字，block size 6 |
| Transformer | 2 layers、embedding 16、4 heads、head dimension 4 |
| 結構 | causal self-attention、RMSNorm、ReLU MLP、residual connections；無 bias、dropout |
| 參數量 | 30,592 |

### 訓練參數

| 參數 | 值 |
|---|---|
| Random seed | 42（資料 shuffle 與參數初始化） |
| Steps | 15,000，單筆姓名每 step |
| Optimizer | Adam；learning rate 0.01、beta1 0.85、beta2 0.99、epsilon 1e-8 |
| LR schedule | 以 15,000 為 target 的線性衰減，訓練結束時接近 0 |
| Checkpoint cadence | 每 1,000 steps 原子更新同一檔案 |
| Evaluation cadence | 每 5,000 steps 計算 723 筆平均 training loss |
| 里程碑取樣 | temperature 0.6、top-p 0.9、sample seed 42、20 筆 |
| 總耗時 | 3,365.41 秒（約 56 分 5 秒，含里程碑評估與取樣） |

### 結果

| 里程碑 | 平均 training loss | 唯一名稱 | 訓練集命中 | 重複 |
|---:|---:|---:|---:|---|
| 5,000 | 4.0167 | 20 / 20 | 1 | 無 |
| 10,000 | 3.7521 | 17 / 20 | 2 | `石中` ×3、`石中君` ×2 |
| 15,000 | 2.7758 | 19 / 20 | 9 | `武三娘` ×2 |

第 15,000 步單筆 loss 為 2.7565。loss 隨訓練下降，但 10,000 steps 起出現高機率字鏈集中，15,000 steps 的訓練集命中升至 9 / 20。若目標是強烈的金庸角色感，15k checkpoint 可用；若目標是創造未見過的名稱，5k 的多樣性較好。

5,000-step 樣本：

> 楊容蒙、慕容斐、王春容、史伯芳、張松林、尼沅春、莫泉師、關淩淑、郭兒、周野王、洪人遜、張春容、張康、常火龍、楊容朝、李岱素、林根布、張春秋、張魯錦、梅芳芹

10,000-step 樣本：

> 范兆和、胡琳之、趙志乾、王道乾、石中、王正、石中君、王道揚、李闊臺、趙雲、范里、吳六龍、王道、石中、趙志平、范火德、王道生、楊難公、石中、石中君

15,000-step 樣本：

> 陸乘風、胡國清、馬伯通、李文、溫方、武三娘、楊不復、李文君、武三娘、殷無壽、崔百迅、史仲強、張康、殷無福、任魯虎、單伯山、陸菲青、水岱、王元通、巴顏

只有最終 15k 權重被保留；5k 與 10k checkpoint 在同一次訓練中被後續里程碑覆寫，上表與樣本是當時的紀錄。

### Checkpoint 內容與使用

checkpoint version 1 內含：

- 30,592 個模型參數、Adam `m` / `v`、Python random state。
- `completed_steps=15000`、`target_steps=15000`。
- dataset hash、字元表、layers、embedding、heads 與 block size。

它不另存 learning rate、Adam beta/epsilon、LR schedule、temperature、top-p 與執行環境；這些資訊以本文件為準。由於格式為 pickle，請勿載入來路不明的 checkpoint。

純 inference：

```bash
uv run python microgpt.py \
  --inference \
  --checkpoint checkpoints/jinyong-2l-e16-h4-15k-seed42.pkl \
  --temperature 0.6 \
  --top-p 0.9
```

## 未來實驗範本

1. 選定不會與既有檔案重複的 ID，建議格式為 `<dataset>-<layers>l-e<embedding>-h<heads>-<steps>k-seed<seed>`。
2. 以 `--checkpoint checkpoints/<experiment-id>.pkl` 訓練，並將實驗設定加入本文件。
3. 至少記錄：資料 hash/筆數、架構、optimizer/LR schedule、seed、steps、平均 loss、耗時、取樣參數、樣本、唯一/重複/訓練集命中。
4. 記錄 checkpoint 路徑、大小、SHA-256、格式版本與是否可續訓。

## 台灣姓名預訓練與 B/C 微調

共同模型使用 2 layers、embedding 16、4 heads、824 tokens（823 個字元 + BOS）、32,608 個參數。B/C 使用完全相同的共同字典與預訓練權重；A 的舊字典只有 761 tokens，因此 A 對 B/C 是實用基準，不是完全控制變因的比較。

| 階段 | 資料 | Steps | LR | Checkpoint |
|---|---|---:|---:|---|
| 預訓練 | 19,655 筆台灣一般姓名 | 5,000 | 0.01 | `taiwan-pretrain-2l-e16-h4-5k-seed42.pkl` |
| B | 723 筆金庸姓名 | 3,000 | 0.002 | `experiment-b-jinyong-finetune-3k-seed42.pkl` |
| C | 70% 金庸 + 30% 一般姓名，共 10,000 rows | 3,000 | 0.002 | `experiment-c-mixed-finetune-3k-seed42.pkl` |

預訓練 checkpoint 的 256 筆平均 training loss 為 3.2028；B 為 4.2969；C 為 4.1611。固定以 temperature 0.6、top-p 0.9、sample seed 20260728 各抽 500 筆：

| 模型 | 唯一 | 重複 draws | 平均長度 | 三字名 | 金庸命中 | 一般姓名命中 |
|---|---:|---:|---:|---:|---:|---:|
| A | 315 / 500 | 185 | 2.736 | 358 | 226 | 7 |
| B-3k | 477 / 500 | 23 | 2.328 | 164 | 6 | 18 |
| C-3k | 466 / 500 | 34 | 2.438 | 219 | 1 | 97 |

B/C 大幅降低直接背誦，但 3,000-step 微調明顯偏向過早輸出 BOS，雙字名過多。C 保留較強的台灣一般姓名結構，武俠風格則比 B 弱。

## B 延伸金庸微調 10k

從 B-3k 權重開始新的延伸階段，重設 Adam，learning rate 0.001，以 10,000 steps 線性衰減。總權重更新歷史為一般姓名 5k + 金庸 3k + 金庸延伸 10k。每 1,000 steps 保存獨立快照，並以相同的 temperature 0.6、top-p 0.9、sample seed 20260728 抽 200 筆：

| 延伸 steps | 唯一 | 空輸出 | 平均長度 | 三字名 | 金庸命中 | 一般命中 |
|---:|---:|---:|---:|---:|---:|---:|
| 1k | 198 | 0 | 2.340 | 68 | 1 | 1 |
| 2k | 195 | 1 | 2.510 | 104 | 2 | 11 |
| 3k | 195 | 2 | 2.640 | 132 | 1 | 1 |
| 4k | 193 | 4 | 2.665 | 135 | 3 | 2 |
| 5k | 198 | 0 | 2.700 | 140 | 4 | 0 |
| **6k** | **192** | **2** | **2.705** | **141** | **0** | **0** |
| 7k | 193 | 2 | 2.655 | 135 | 3 | 3 |
| 8k | 195 | 0 | 2.670 | 134 | 5 | 1 |
| 9k | 193 | 0 | 2.555 | 109 | 4 | 2 |
| 10k | 194 | 3 | 2.670 | 135 | 6 | 0 |

推薦 `experiment-b-jinyong-extended-at-06k-seed42.pkl`：三字比例最高，本批對兩份訓練集均零命中，人工樣本也相對最佳，例如「韓文威、李立傑、安大海、張伯通、馮劍生、陳元平、單國駿」。5k 快照有更高唯一率且無空輸出，可作重視穩定性的備選。7k 後品質開始平台或退化，9k 的三字比例明顯跌到 54.5%。

推薦 checkpoint SHA-256：`4fa76b2989da70db28b3907e6b36710ae48806ad0a7654246a37e15119bd575c`。完整 10k checkpoint SHA-256：`ec7706a2c48c8550824295313b8f49b21c68c32ce0af8e289364059b5ed01453`。各里程碑的原始 200 筆結果位於 `data/results/experiment_b_extended_*_200.json`。
