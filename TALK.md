# MicroGPT 演講圖稿

全場以 **`microgpt_en.py`** 為準：英文名字資料集（`names.txt`，32,033 個名字），
**1 層** transformer、16 維、4 個 head、context 長度 16、vocab 27。

貫穿全場的例子：生成 **`emma`**。

> 這份圖稿只講架構本身。所有方框裡的數值都是**示意值**，用來說明形狀與流向，
> 不是某個 checkpoint 的實際輸出。真實權重的觀察（位置向量學到什麼、head 的分工、
> 中文資料集的延伸實驗）放在最後的「補充」一節，講完之後再談。

---

## 圖 0 — 開場：這個模型有多小

先建立尺度感，讓觀眾知道等一下看到的每個數字都是真的、可以手算的。

```
             ┌───────────────────────────────────────────────┐
             │            整個模型的全部知識                  │
             ├───────────────────────────────────────────────┤
  wte        │   27 × 16   把字母變成向量（查表）        432   │
  wpe        │   16 × 16   位置 0~15 的向量              256   │
  lm_head    │   27 × 16   把向量變回字母（反查表）      432   │
             ├───────────────────────────────────────────────┤
  layer 0    │  Wq Wk Wv Wo   各 16×16                 1,024   │
             │  MLP  16→64→16                          2,048   │
             ├───────────────────────────────────────────────┤
             │  合計                                    4,192  │
             └───────────────────────────────────────────────┘

  對照：GPT-2 small = 124,000,000（3 萬倍）
        現代大模型 ≈ 100,000,000,000+
```

**講稿要點**：四千個數字，一個 200 行、零依賴的 Python 檔案。它學會了「英文名字長什麼樣子」。
等一下我們會把它拆到只剩加法和乘法。

> 值得指出：**只有一層。** 完整的 Transformer 不需要「很多層」才成立 ——
> 一層就已經包含了 attention、MLP、殘差這三件事的全部。多層只是把同一件事再做一次。

**配碼（microgpt_en.py）**

```
75   n_layer = 1     # depth of the transformer …        # ★ n_layer：疊幾層
76   n_embd = 16     # width of the network …            # ★ n_embd：x 的寬度＝16
77   block_size = 16 # maximum context length …          # ★ block_size：最多看幾格（圖 3b 主場）
78   n_head = 4      # number of attention heads         # ★ n_head：attention 分幾頭
79   head_dim = n_embd // n_head # derived dimension …   # ★ head_dim：每頭分到 16÷4＝4 維
80   matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
     # ★ matrix：造一個矩陣；★ Value：「一個會記帳的數字」（先這樣理解，圖 10 收帳）
81   state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
     # ★ state_dict：全部知識的收納櫃；★ vocab_size：27，字母加一個特殊符號（圖 2 細講）
82   for i in range(n_layer):
83       state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
⋯
87       state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
⋯
89   params = [p for mat in state_dict.values() for row in mat for p in row] # …
     # ★ params：攤平成一條 list 的全部 4,192 個數字
90   print(f"num params: {len(params)}")
```

**★ 先講變數**：五個超參數 `n_layer`／`n_embd`／`block_size`／`n_head`／`head_dim`
（左圖那五個矩陣的長寬就是它們）；`matrix`＝造矩陣的工具；`Value`＝會記帳的數字（帳圖 10 講）；
`state_dict`＝知識收納櫃；`params`＝攤平的 4,192 個數字；`vocab_size`＝27（由來圖 2 馬上講）。

---

## 預習 — 進架構之前的心理準備

> 受眾設定：對 Machine Learning 很有熱忱、但覺得它還很難的人。
> 這幾小節不講新架構，只講清楚兩件事：
> **(預 1) 全場只有一個主角。(預 2–預 4) 這個模型只有三種運算——一種一頁，有圖有程式。**
> 建議位置：圖 0（尺度感）之後、圖 1（地圖）之前。
> 節奏預算：預 1 約 2.5 分、預 2 約 2 分、預 3a 約 2.5 分、預 3b 約 2.5 分、預 4 約 3 分（含 rmsnorm）——
> 開場到圖 1 約 14 分鐘（60–90 分鐘的場可行；45 分鐘場請用文末的降級模式）。
> 這幾節都用同一個例子：那個「第 3 格的 m」的 x——預 1 介紹它，預 2–4 是它會遇到的三種運算。

> **配碼慣例（全檔適用）**：每節的「配碼」區塊＝這一節投影片右欄要放的程式碼。
> 行號＝`microgpt_en.py` 的真實行號，內容逐字（整行省略以 `⋯` 表示；過長的行尾原註解可截短為 `…`）。
> 行尾 **`# ★ …`** 的註解是講稿加註（**源碼裡沒有**），標示這個變數**在演講中第一次被正式介紹**——
> 投影片上要跟著印，講的時候看到 ★ 就先停下來，用一句話介紹這個變數再往下。
> 若變數在更早的節先露過臉，那一節的「★ 先講變數」會標「先不停留，圖 N 主場」。
> 沒有行號的行（`#` 說明、└ 括線）是圖解註記，不是源碼。

### 預 1 — 全場只有一個主角：一個 16 維向量

```
  它長這樣——就是一個裝著 16 個小數的 Python list：

     x = [ 0.13, -0.82, 0.55, … , -0.07 ]     ← 共 16 個
           └────────────┬────────────┘
              這 16 個數字，濃縮了「這個字母、
              以及它前面所有字母」到此刻的理解

  整個程式 = 對同一個 x 不停「加料」。以生成 emma 走到第 3 步為例：

     出生：wte 查表          「我是字母 m」                    （圖 3）
       │   ＋ wpe
       ▼
     知道位置                「我是第 3 格的 m」               （圖 3）
       │   ＋ attention 算出來的修正
       ▼
     拌入前文                「我前面是 e、m」                 （圖 4）
       │   ＋ MLP 算出來的修正
       ▼
     想清楚了                「下一個八成是 a」                 （圖 5）

  ★ 注意每一步的動詞都是「＋」——三個加號，加料是字面上的加法（圖 6 收這條線）
```

**講稿要點**

- 別被「16 維向量」四個字嚇到：**它就是一個長度 16 的 list**，沒有更多了。
- 整個模型跑一輪 = 這個 list 被加料**三次**（＋位置、＋attention 的修正、＋MLP 的修正）。
  你等一下看到的每一張圖，都只是在回答「這一步往 x 裡加了什麼料」。
- 加完三次料、**進出口（lm_head）之前**，兩件事已經同時濃縮在這 16 個數字裡：
  「這個字此刻該表達的意義」和「它對下一個字母的預測」——
  而且圖 7 會揭曉：這兩件事其實是**同一件事**。
- 尾鉤：**這個 x 從頭到尾只會被三種運算對待。**下一頁開始，一種一頁。

**配碼**：無——這一節只有圖。x 的程式碼本體（就是一個 Python list）在預 2 第一次亮相。

### 預 2 — ① 相加：向量＋向量（把兩份理解疊在一起）

> 框架先一句話：**這個模型從頭到尾只有三種運算。**接下來一種一頁，
> 都用同一個例子示範——那個「第 3 格的 m」。

```
   「我是字母 m」         [ 0.3, -0.1,  0.7, … ]     ← wte 查出來的一列
 ＋「我在第 3 格」        [ 0.1,  0.4, -0.2, … ]     ← wpe 查出來的一列
 ──────────────────────────────────────────────
   「我是第 3 格的 m」    [ 0.4,  0.3,  0.5, … ]     ← 兩張便利貼疊起來

  相加的第二個用途：把「修正量」疊上去（attention／MLP 算出來的料）

     x ＋ 修正量 ─▶ 更新後的 x        殘差就是 x += 修正，不是 x = 修正
                                    （疊上去、不覆蓋——圖 6 收這條線）
```

**講稿要點**

- 相加的意義：**把兩份理解疊在一起**——字母的意義＋位置的感覺＝帶著位置的字母意義。
- 對寫過程式的人一句話：殘差就是 `x += 修正`，不是 `x = 修正`。
  全場沒有任何一步會「洗掉」x。
- 句型順帶學：`zip`＝兩個 list 並排走；list comprehension＝對每一格做同一件事。
  逐格相加，就這麼多。
- 尾鉤：加法只會**疊**、不會**變**——想從「第 3 格的 m」變出「我在找母音」這種新理解，
  得靠第二種運算：相乘。

**配碼（microgpt_en.py）**

```
111      x = [t + p for t, p in zip(tok_emb, pos_emb)] # …    # ★ x、tok_emb、pos_emb
⋯
134          x = [a + b for a, b in zip(x, x_residual)]       # ★ x_residual
⋯
141          x = [a + b for a, b in zip(x, x_residual)]
```

**★ 先講變數**：`x`＝主角本人（那個 16 維 list）；`tok_emb`＝wte 查出來的字母向量、
`pos_emb`＝wpe 查出來的位置向量（出生地 L109-110，圖 3 再看）；
`x_residual`＝加料前先留一份的 x 備份——「疊上去」的技術名字叫**殘差**（圖 6 主場）。

### 預 3a — ② 相乘（一）：兩個向量的內積

> 這一頁只講一件事，而且用真的數字算完一次。
> 很多人只是很久沒算數學了，不是不會——把它算一次，後面全部都通。

```
  兩個向量，各 16 個數字：

     x ＝ [  0.4,   0.3,   0.5,  … ]     ← 主角，那個「第 3 格的 m」
     w ＝ [  0.2,  -0.1,   0.9,  … ]     ← 另一個 16 維向量

  內積＝逐項相乘，再全部加總：

     0.4×0.2 ＋ 0.3×(-0.1) ＋ 0.5×0.9 ＋ …  ＝  1.23
       0.08        -0.03        0.45                 ▲
                                            兩個向量進去，出來只有「一個數字」

  Python 就是這一行：

     sum(wi * xi for wi, xi in zip(w, x))
         └ 逐項相乘 ┘     └ zip：兩個 list 並排走 ┘
     sum(...)＝把它們全部加起來
```

**講稿要點**

- **兩個 16 維向量進去、一個數字出來。**整個內積就這件事，沒有別的。
- 這個數字代表什麼：**兩個向量有多同方向**。同方向→大正數、無關→接近 0、反方向→負數。
- 先埋：等一下 attention 判斷「我跟前面那一格像不像」，用的是同一行程式。
- 尾鉤：一個內積只給我們一個數字——但主幹上要的是 16 個數字。所以要做 16 次。

### 預 3b — ② 相乘（二）：矩陣就是一疊向量

> **Python 裡沒有「矩陣」這種型別。**所以這一頁把數學跟程式並排放：
> 矩陣在紙上長什麼樣、在程式裡就是什麼寫法。

```
  一個矩陣 w，在 Python 裡就是「list 裡面裝 list」：

     w = [ [ 0.2, -0.1,  0.9, … ],    ← 第 0 列，16 個數字 ─┐
           [ 0.7,  0.3, -0.4, … ],    ← 第 1 列             │  每一列自己
             ⋮                                              │  就是一個
           [-0.5,  0.8,  0.1, … ] ]   ← 第 15 列          ─┘  16 維向量

  x 拿去跟「每一列」各做一次預 3a 的內積：

     x · 第 0 列   ──▶   0.87
     x · 第 1 列   ──▶  -1.20          16 列 → 16 個數字
        ⋮                              ＝又是一個 16 維向量
     x · 第 15 列  ──▶   0.34

  程式：外層每一列做一次，內層就是預 3a 那一行

     def linear(x, w):
         return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
                 └──── 預 3a：一列一次內積 ────┘   └ wo 逐一取出每一列 ┘

  矩陣是怎麼生出來的（同樣是巢狀 list）：

     matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std))
                  for _ in range(nin)] for _ in range(nout)]
              # 外層跑 nout 次＝幾列；內層跑 nin 次＝一列有幾個數字

  全場總共用 7 次，矩陣不同、動作永遠一樣：
    × Wq（圖 4a-0）    × fc1（圖 5）    × lm_head（圖 7）    …
```

**講稿要點**

- 一句話：**矩陣就是一疊向量，相乘就是拿 x 跟每一個都做一次內積。**
- 所以「16 維 × 矩陣」能算出什麼，取決於矩陣裡那些列是什麼——**而那些列是訓練學出來的**。
- 形狀怎麼看：矩陣有 64 列，出來就是 64 個數字（fc1 就是這樣把 16 撐成 64）。
- 這一頁不深講任何一個矩陣的內容（各自的主場：圖 4a-0、圖 5、圖 7）。
- 尾鉤：算出來的數字**正負混雜、大小不一**——誰來過濾？第三種運算。

**★ 先講變數**：`linear(x, w)`＝把 x 跟矩陣 w 的每一列做內積的函數
（`wo`＝其中一列、`wi`／`xi`＝內積時逐項對上的兩個數）；
7 次出現＝q、k、v、Wo、fc1、fc2、lm_head，每次都是這個動作。

### 預 4 — ③ 非線性（主角是 relu）

```
  relu＝一道門檻：不夠強的訊號，閉嘴

    進來    −1.2    +3.4    −0.8    +0.1
              │       │       │       │
            ──┴───────┴───────┴───────┴──    負的一律歸零，正的原樣通過
              │       │       │       │
    出去      0     +3.4      0     +0.1

  為什麼非有它不可（一句話版，完整論證圖 5 還債）：
    沒有 relu，fc1 × fc2 可以事先乘成一個 16×16 矩陣——疊一百層等於一層
```

**講稿要點**

- **relu 就是神經網路裡的 if。**沒有 if，程式只是一條算式；有了 if，才會「分情況」。
  （圖 5 會把 MLP 讀成「64 條 if-then 規則」——就是這個 if。）
- 相加是疊、相乘是換角度，都還是線性的——**非線性是唯一能製造「分情況」的地方**。
- 非線性家族還有個 softmax（把一排分數擠成加總＝1 的比例，attention 與抽樣那邊見）；
  再加上下面的 rmsnorm——這兩個都是配角，**「只有三種運算」仍然成立**。

還有一個 rmsnorm，**音控台**。它不改變 x 說什麼，只管音量：

```
  rmsnorm＝音控台：只調音量、不動方向

     [ 3.2, -1.6,  4.8, … ]   ← 加料疊久了、乘過幾個矩陣，音量會漂
               │   ÷ 整體音量（16 個數字的均方根）
               ▼
     [ 0.8, -0.4,  1.2, … ]   ← 方向沒變，音量回到標準
```

- **為什麼需要音控**：相加一直疊、相乘會放大縮小——**x 的音量會漂移**。
  音量一漂，內積評分就沒有共同的尺度，softmax 還會被極端值綁架。
- 所以**每次拿 x 去作答之前，先過一次音控**：進主幹前（圖 3）、attention 作答前（圖 4）、
  MLP 作答前（圖 5）——位置永遠一樣：**作答前**。
- rmsnorm **零參數**：4,192 個數字裡沒有半個屬於它。它不是知識，是水電。
- 收尾（原預 5 的三個「只有」，壓成一句）：**運算只有三種、數字只有 4,192 個**——
  之後每張圖都可以問一句：「這一步用的是哪一種運算？」
  先劇透：Attention＝用這三種去「搬」前文；MLP＝用這三種來「想」（圖 4 正式開講）。

**配碼（microgpt_en.py）**

```
50       def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
         # ★ self.data：Value 的數值本體；後面兩個括號＝記帳用（微分 0 或 1，圖 10 收）
⋯
103  def rmsnorm(x):                            # ★ rmsnorm：音控台（零參數）
104      ms = sum(xi * xi for xi in x) / len(x)     # ms：目前的平均音量（均方）
105      scale = (ms + 1e-5) ** -0.5                # scale：調回標準音量的倍率
106      return [xi * scale for xi in x]            # 只動音量，不動方向
⋯
139          x = [xi.relu() for xi in x]   # 16 個數字各自過門檻
```

**★ 先講變數**：`self.data`——一個 `Value` 就是「一個會記帳的數字」（圖 0 出場過），
`.data` 是它的數值本體；帳記什麼、給誰看，圖 10 才講。
`rmsnorm`＝音控台：`ms` 量出目前音量、`scale` 算出倍率、逐格乘回去——三行、零參數。

---

## 圖 1 — Overview：一次前向，只產生一個字母

```
  文字      ①Tokenizer      ②Embedding       ③Block × 1        ④lm_head      ⑤抽樣
   │            │                │                │                 │            │
   ▼            ▼                ▼                ▼                 ▼            ▼
 「em」  ─▶  字元→id  ─▶   id → 16 維向量  ─▶ ┌──────────┐ ─▶  16 → 27   ─▶  溫度
              [e]=4         wte[id]+wpe[pos]   │Attention │      分數        加權隨機
                                               │   ＋     │                     │
                              ▲                │   MLP    │                     │
                              │                └────┬─────┘                     ▼
                              │                  │  ▲                        「m」
                              │                  ▼  │                           │
                              │            ╔═══════════════╗                    │
                              │            ║   KV cache    ║                    │
                              │            ║ 過去每個字母的 ║ ◀── 每輪 append    │
                              │            ║   k 和 v      ║                    │
                              │            ╚═══════════════╝                    │
                              │                                                 │
                              └─────────────────────────────────────────────────┘
                                    抽到的字母回頭當下一輪的輸入
                                            （自迴歸）

  形狀：  字母 ─▶ 整數 ─▶ 16 ─▶ 16 ─▶ 16 ─▶ 27 ─▶ 1 個字母
                             └── 中間全程都是 16 維 ──┘
```

**講稿要點**

- 回收預 1：預 1 看的是 x 的**內在履歷**，這一張是**工廠的外觀**——五站流水線。
- 中間**全程 16 維不變**。模型不是「越算越大」，而是同一個 16 維向量被反覆修改。
- **①和④是一對鏡像**：wte 是「id → 向量」，lm_head 是「向量 → 每個 id 的分數」。
  整個模型 = 把字變成向量 → 在向量空間裡想事情 → 把向量變回字。
- ⑤是**全流程唯一有隨機性的地方**。前面全是確定性的乘加。
  同一個模型每次跑出不同名字，原因只在這一格。
- 雙線框的 KV cache 是**跨輪留下來的狀態**，不是流過去的資料 —— 顏色要跟主線區分。

**配碼（microgpt_en.py・推論迴圈＝整條流水線）**

```
189  for sample_idx in range(20):                          # ★ sample_idx：第幾個名字（要生 20 個）
190      keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
         # ★ keys、values：KV cache 本體——兩個只進不出的 list（圖 9 主場）
191      token_id = BOS                                    # ★ token_id：現在手上這個字母的編號
192      sample = []                                       # ★ sample：已經生出來的字母們
193      for pos_id in range(block_size):
194          logits = gpt(token_id, pos_id, keys, values)  # ★ gpt：整個模型＝一個函數；★ logits：27 個分數
195          probs = softmax([l / temperature for l in logits])
             # ★ probs：27 個機率；temperature 圖 8 主場，先不停留
196          token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
197          if token_id == BOS:
198              break
199          sample.append(uchars[token_id])
```

**★ 先講變數**：`keys`／`values`＝過去每格算好的 k、v（圖 9 收）；`token_id`＝現在手上的字母編號；
`gpt`＝整個模型本體（一個函數，圖 3 進門）；`logits`＝27 個分數、`probs`＝27 個機率；
`sample`＝收集輸出。先露臉但不停留：`temperature`（圖 8 主場）、`BOS`／`uchars`（下一張圖 2 馬上講）。

---

## 圖 2 — Tokenizer：這裡沒有魔法

```
  訓練資料裡出現過的所有字元，排序後編號（microgpt_en.py:24）：

     a=0  b=1  c=2  d=3  e=4  f=5  ...  y=24  z=25     ← 共 26 個
                                                         BOS = 26
                                                         ─────────
                                                         vocab = 27

  「emma」 ──▶ [4, 12, 12, 0]

  加上開頭與結尾（同一個符號！）：

     BOS    e     m     m     a    BOS
      26    4    12    12     0     26
      ▲                             ▲
      │                             │
    「開始了」                   「結束了」
```

**講稿要點**：一個字母一個 token，沒有 BPE、沒有 subword —— 就是 `sorted(set(...))` 一行。
BOS 同時當開頭和結尾，生成時抽到它就停手（`microgpt_en.py:197-198`）。
**「模型怎麼知道要停？」的答案就是：停止也是一個它要學會預測的 token。**

**配碼（microgpt_en.py）**

```
24   uchars = sorted(set(''.join(docs))) # unique characters …   # ★ uchars：字元表；★ docs：全部 32,033 個名字
25   BOS = len(uchars) # token id for a special Beginning …      # ★ BOS：開頭兼結尾的特殊符號＝26
26   vocab_size = len(uchars) + 1 # total number of unique …     # 圖 0 那個 27 的由來
⋯
157      tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]   # ★ tokens：一筆名字變成的編號串
⋯
196          token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
197          if token_id == BOS:
198              break
```

**★ 先講變數**：`docs`＝整個資料集（一行一個名字）；`uchars`＝出現過的字元排序表；
`BOS`＝第 27 個 token，開頭兼結尾；`tokens`＝「emma」變成的 [26, 4, 12, 12, 0, 26]（`doc`＝這一筆名字）。

---

## 圖 3 — Embedding：從符號進入語意空間

```
   token id = 4（「e」）           位置 pos = 1
          │                            │
          ▼                            ▼
    wte 的第 4 列                  wpe 的第 1 列
   [ 16 個數字 ]                  [ 16 個數字 ]
          │                            │
          └──────────  相加  ──────────┘
                        │
                        ▼
                 [ 16 個數字 ]
                        │
                    rmsnorm  ← 把長度標準化
                        │
                        ▼
              進入主幹（residual stream）
```

**講稿要點**：這是「符號」變成「意義」的轉折點（`microgpt_en.py:109-112`）。
這也是**預 2「相加」的第一次正式上場**——L111 那個「＋」。
加上 `wpe` 是模型唯一知道「我在第幾個字母」的管道 ——
把它拿掉，`emma` 和 `amme` 對模型完全相同。

> `wpe` 不是公式，**它就是普通參數**：跟 `wte` 一樣放進 `state_dict`、
> 一樣被攤平進 `params`、訓練迴圈一視同仁地更新。
> 「第 0 格」和「第 3 格」的意思是模型自己從資料裡學出來的。

**配碼（microgpt_en.py・gpt() 的第一站）**

```
108  def gpt(token_id, pos_id, keys, values):
109      tok_emb = state_dict['wte'][token_id] # token embedding    # 預 2 見過：wte 的一列
110      pos_emb = state_dict['wpe'][pos_id] # position embedding
111      x = [t + p for t, p in zip(tok_emb, pos_emb)] # …          # 預 2 的相加，正式上場
112      x = rmsnorm(x) # note: not redundant due to …              # 預 4 的音控台，正式上工
```

（`rmsnorm` 預 4 介紹過。全場它上工三次：這裡、attention 作答前（L117）、MLP 作答前（L137）——
永遠是「作答前」。源碼註解那句 not redundant：殘差旁路帶的是沒過音控的 x，所以這裡再過一次不多餘。）

---

## 圖 3b — context length 上限就是 wpe 矩陣的列數

```
  wpe 是一張只有 block_size = 16 列的表

     pos  0 │ [16 個數字] │
     pos  1 │ [16 個數字] │
        ⋮   │      ⋮      │
     pos 15 │ [16 個數字] │
     ───────┼─────────────┤
     pos 16 │     ✗       │ ← 沒有這一列，查不到，直接 crash
```

`block_size` 取訓練資料裡最長的名字（15 個字母，`microgpt_en.py:77`）。
第 17 個位置**不是效果變差，是物理上不存在**。

| 做法 | 位置資訊哪來 | 誰在用 |
|---|---|---|
| **學出來的**（本專案） | 就是普通參數，訓練出來 | GPT-2、本專案 |
| **正弦函數** | 寫死的 sin/cos 公式，不訓練 | 原始 Transformer 論文 |
| **RoPE** | 不用加，直接把 q/k 旋轉一個角度 | Llama 等現代模型 |

**講稿收尾**：「你聽過『這個模型 context 是 128K』—— 在最小的模型裡，那件事就長這樣：
一張只有 16 列的表。」
（現代模型改用 RoPE，正是因為旋轉角度可以外推到訓練時沒看過的長度。）

**配碼（microgpt_en.py）**

```
77   block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
⋯
81   state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
     # wpe 就只有 block_size＝16 列
⋯
110      pos_emb = state_dict['wpe'][pos_id] # position embedding   # pos_id 超過 15 → 這行直接 IndexError
```

---

## 圖 4 — Transformer Block：一橫一縱

進細節之前，先把語感講清楚。一個 Block 裡面只有兩個動作，而且方向完全不同。

```
              這一格的 x (16)
                    │
   ┌────────────────▼────────────────┐
   │           Attention             │  ← 橫向：跟「前面所有 token」講話
   │      「我需要前面哪些資訊？」        │     這是整個模型唯一能跨 token
   │                                 │     交換資訊的地方
   └────────────────┬────────────────┘
                    │   現在 x 裡面混進了前面 token 的意義
   ┌────────────────▼────────────────┐
   │              MLP                │  ← 縱向：一個人關起門來想
   │   「有了這些資訊，我懂了什麼？」      │     完全不看別的 token
   │                                 │
   └────────────────┬────────────────┘
                    ▼
              更新後的 x (16)
```

把位置攤開來看，這兩個方向就更明顯。先看**橫的**：

```
  正在算第 4 格（第二個 m），前面三格早就算完了

        BOS         e          m          m  ◀── 現在只算這一格
         │          │          │          │
        k,v        k,v        k,v       q,k,v
         │          │          │          │
         └──────────┴──────────┴────────▶ ●
                                          │
          前三格的 k,v 直接沿用             把前面每一格的意義
          （不重算，見圖 9 KV cache）        按比例拌進「我」自己
```

拌完之後，換**縱的** —— 注意這裡只剩一根箭頭：

```
        BOS         e          m          m
         ·          ·          ·          │   ← 前三格這一輪完全沒動
                                          ▼
                                     ┌─────────┐
                                     │   MLP   │  ← 只吃自己這一個 16 維向量
                                     └────┬────┘
                                          ▼
                                    更新後的 x (16)
                                          │
                                          ▼
                                   lm_head → 下一個字母
```

**講稿要點**

- **「一橫一縱」就是整個 Transformer 的骨架。** Attention 負責**搬運**，
  MLP 負責**加工**。兩者交替，就是全部。
  （這句話等一下在圖 6 還會再出現一次，那時候它會有更深的意思。）
- **只有 Attention 會跨 token。** 注意這裡的重點不是「自己融合自己」，
  而是**這一格去把前面每一個 token 的意義按比例拌進來**。
  把 Attention 拿掉，模型就退化成「每個字母各自獨立預測下一個字母」——
  完全不知道前面出現過什麼。
- **一輪只有一格在算。** 橫向那張圖裡站著四個 token，但真正在做運算的只有最右邊那一格；
  前面三格貢獻的是**算過就冰起來的 k,v**。
  縱向那張圖之所以只剩一根箭頭，就是這個原因 —— 這裡先埋，圖 9 再收。
- **MLP 的橫向是斷開的。** 這不是設計上的巧思，是程式碼裡看得到的事實：
  MLP 那三行（`microgpt_en.py:138-140`）只吃 `x` 這一個向量，
  手上根本沒有其他位置的資料。
- **箭頭往最後一格匯聚，而且只從左邊來**（causal）：第 4 格讀得到第 1、2、3 格，
  讀不到還沒出現的第 5 格。因為任務是「猜下一個」，看得到答案就是作弊。

> 小字但誠實：上面兩張畫的是**推論時**的樣子。**訓練時**整個名字一次餵進去，
> 每一格都會各自走一遍 MLP —— 但即使那樣，各格之間仍然完全不交談。

> 接下來四小節（4a-0 → 4a → 4b → 4c），全部都在拆解上面那個 Attention 方框裡到底發生了什麼。

**配碼（microgpt_en.py・一個 Block 的全部，先看骨架）**

```
114      for li in range(n_layer):
115          # 1) Multi-head Attention block
116          x_residual = x
117          x = rmsnorm(x)
118          q = linear(x, state_dict[f'layer{li}.attn_wq'])   # q？——下一節 4a-0 專門介紹，先看結構
⋯
133          x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])   # ★ x_attn：四個 head 的輸出接起來（16 維）
134          x = [a + b for a, b in zip(x, x_residual)]
135          # 2) MLP block
136          x_residual = x
137          x = rmsnorm(x)
138          x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
139          x = [xi.relu() for xi in x]
140          x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
141          x = [a + b for a, b in zip(x, x_residual)]
```

**★ 先講變數**：`x_attn`＝attention 這半的最終輸出、要拼回 16 維的那條 list（拼法圖 4c）。
`q` 先露臉不停留——4a-0 主場。

**配碼（攤開那一頁：橫的讀全部、縱的只吃自己）**

```
121          keys[li].append(k)
122          values[li].append(v)
⋯
129              attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
130              attn_weights = softmax(attn_logits)
131              head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
⋯
138          x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
139          x = [xi.relu() for xi in x]
140          x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
```

---

## 圖 4a-0 — 先認識 q、k、v：同一個 x 的三種角度

**場景**：拆開 Attention 方框之前，先把三個新角色介紹完。
這一節**零數字**——動機在這裡，數字下一張（圖 4a）才登場。

```
        rmsnorm 後的 x（16 個數字）＝這一格的「m」此刻的理解
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
       × Wq         × Wk         × Wv     ← 三個不同的 16×16 矩陣（相乘連用三次）
         │            │            │
         ▼            ▼            ▼
       q (16)       k (16)       v (16)   ← 之後各切 4 段，每個 head 拿 4 個（圖 4c）
    「我想找什麼」  「我是什麼」  「找到我，我給你什麼」
      （搜尋詞）    （書背標題）    （書的內容）
         │            │            │
         │            └─ k、v 存進 KV cache：前文每一格都留過一份 ─┘
         └── q 只服務這一輪，用完即丟
```

```
  一個 head 的七步，每一步只為了一個效果：

   ┌───────────┬────────────────┬─────────────────────────────┐
   │ 轉化       │ 形狀            │ 想得到什麼                    │
   ├───────────┼────────────────┼─────────────────────────────┤
   │ rmsnorm   │ 16 → 16        │ 先把音量歸一，別讓誰壓過誰       │
   │ × Wq      │ 16 → 16（切 4） │ 抽出「搜尋詞」（上圖）           │
   │ × Wk      │ 16 → 16（切 4） │ 掛出「書背標題」（上圖）         │
   │ × Wv      │ 16 → 16（切 4） │ 備好「書的內容」（上圖）         │
   │ q·k 內積   │ 每格前文 1 分   │ 我跟每一格前文有多對頻           │
   │ softmax   │ 分數 → 比例     │ 擠成加總＝1 的注意力（非線性）  │
   │ Σ wₜ·vₜ   │ 比例 × v → 4   │ 按比例把前文內容拌回「我」       │
   └───────────┴────────────────┴─────────────────────────────┘

   （x 的來歷：wte＋wpe＋前文——預 1／圖 3 講過，這裡不重來）
```

```
  全檔最長的一行，先拆好（microgpt_en.py:129）：

     attn_logits = [ sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                     for t in range(len(k_h)) ]
       ＝ 對過去每一格 t：搜尋詞 q 跟它的書背 k 逐項對答案（內積），再 ÷√4

     q_h、k_h ＝ q、k 切給這個 head 的那 4 維（怎麼切：圖 4b／4c）
     圖 4a 馬上一格一格走一遍
```

**講稿要點**

- 三張紙條都是**同一個 x 自己算出來的**（配碼 L118-120），不是外面發的。
- 為什麼要三個角度、不拿 x 直接比 x？——「我在找的」跟「我是誰」通常不是同一件事：
  m 在找母音，但它自己是子音。三個矩陣，各司其職。
- k、v 是「留給後人查的」，q 是「當下用完就丟的」——這個不對稱，圖 9 的 KV cache 會收。

**配碼（microgpt_en.py）**

```
118          q = linear(x, state_dict[f'layer{li}.attn_wq'])   # ★ q：搜尋詞（16 維）
119          k = linear(x, state_dict[f'layer{li}.attn_wk'])   # ★ k：書背標題
120          v = linear(x, state_dict[f'layer{li}.attn_wv'])   # ★ v：書的內容
⋯
129              attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
                 # ★ q_h、k_h：q、k 切給這個 head 的 4 維（就是左圖拆過的那一行）
```

**★ 先講變數**：`q`／`k`／`v`＝同一個 x 乘上三個不同矩陣的結果（搜尋詞／書背標題／書的內容）；
`q_h`／`k_h`＝切給單一 head 的那 4 維切片。

---

## 圖 4a — 一個 attention head

**場景**：把圖 4 的 Attention 方框放大。已經有 `em`，正在決定第三個字母。看第 0 層第 0 個 head。
head 裡每個向量只有 **4 個數字**（16 ÷ 4），可以手算。

> 方框內為示意值，用來說明流程與比例，不是實測輸出。

```
   查詢字母「m」的 q = [ 4 個數字 ]
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
   內積 ÷ √4=2          內積 ÷ 2             內積 ÷ 2
       │                    │                    │
  k(BOS)=[4 個數字]    k(e)=[4 個數字]      k(m)=[4 個數字]
       │                    │                    │
       ▼                    ▼                    ▼
   logit=+2.7           logit=+0.5           logit=-1.2
       │                    │                    │
       └──────────────── softmax ────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       ▼                    ▼                    ▼
     w=0.85               w=0.13               w=0.02
   ██████████████████      ███                   ·
       │                    │                    │
       ▼                    ▼                    ▼
     v(BOS)               v(e)                 v(m)   ← 各 4 個數字
       │                    │                    │
       └──────── 0.85·v(BOS) + 0.13·v(e) + 0.02·v(m) ────────┐
                                                             ▼
                                                    head_out（4 個數字）
```

**講稿要點**

1. **q 只有一個，k/v 有一串**——這個不對稱是整張圖的骨架（圖 9 的 KV cache 會收）。
   三張紙條是誰、從哪來，4a-0 剛講完——這裡直接看數字怎麼流。
2. **÷ √4 就是 ÷ 2。** 維度越高內積越大，除掉避免 softmax 太極端。
3. **softmax 的輸出加起來 = 1。** 「注意力」這個名字就從這來 ——
   它是一組必須分完的比例，多看這裡就得少看那裡。
4. **序列長度在這裡消失了。** 進去是 4 維的 q，過去有 3 個字母或 300 個字母，
   出來永遠是 4 個數字。

**配碼（microgpt_en.py・head 內部三行）**

```
118          q = linear(x, state_dict[f'layer{li}.attn_wq'])
119          k = linear(x, state_dict[f'layer{li}.attn_wk'])
120          v = linear(x, state_dict[f'layer{li}.attn_wv'])
⋯
129              attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
                 # ★ attn_logits：跟每一格前文的「對頻分數」（+2.7／+0.5／−1.2）
130              attn_weights = softmax(attn_logits)   # ★ attn_weights：分好的注意力比例，加總＝1
131              head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
                 # ★ head_out：按比例搬回來的 4 個數字
132              x_attn.extend(head_out)
```

**★ 先講變數**：`attn_logits`＝比對分數（每格前文一個）；`attn_weights`＝softmax 後的注意力比例；
`head_out`＝這個 head 搬運的成果（4 維）。

---

## 圖 4b — 為什麼要切成四個 head

同一時刻、同一層，四個 head 各自跑一遍圖 4a，可以看向完全不同的位置：

```
  【生成 emma 的第 4 個字母時（已有 e, m, m）】     ← 示意分佈

              BOS       e        m        m
  head 0  │        │        │        │████████│   盯著剛剛那個字母
  head 1  │ ███████│        │        │        │   盯著開頭（我在名字的哪裡？）
  head 2  │        │███████ │        │        │   盯著第一個字母
  head 3  │ ██     │ ██     │ ███    │ ███    │   分散：整體長度感
```

**講稿要點**

- **單頭做不到這件事** —— 16 個輸出維度會被迫共用同一組注意力權重，
  只能「全看剛剛那個」或「全看開頭」，或糊成一團平均。
- 切 head **不花任何參數、不花任何計算量**。
  Wq/Wk/Wv/Wo 不管切幾個 head 都是 16×16。差別只在 softmax 被套在哪裡：
  - 單頭 16 維：`softmax(h₀+h₁+h₂+h₃)`
  - 四頭 4 維：`softmax(h₀)`, `softmax(h₁)`, `softmax(h₂)`, `softmax(h₃)`
  - （16 維的內積在數學上**恰好等於**四個 head 內積的和 ——
    所以差別完全來自 softmax 這個非線性。）
- 代價：每個 head 只有 4 維，衡量「像不像」的解析度低很多。
  **取捨 = 一個高解析度的問題 vs 四個低解析度的問題。**

**配碼（microgpt_en.py・「切」就是一行切片）**

```
79   head_dim = n_embd // n_head # derived dimension of each head
⋯
124          for h in range(n_head):
125              hs = h * head_dim                                # ★ hs：這個 head 的切片起點（0、4、8、12）
126              q_h = q[hs:hs+head_dim]
127              k_h = [ki[hs:hs+head_dim] for ki in keys[li]]    # ki／vi：cache 裡每一格前文的 k、v
128              v_h = [vi[hs:hs+head_dim] for vi in values[li]]
⋯
132              x_attn.extend(head_out)
```

**★ 先講變數**：`hs`＝head start，第 h 個 head 從第幾格開始切（h×4）。

---

## 圖 4c — 接線圖：head 怎麼裝回 Block

```
  主幹 x (16) ────────────────────────────────────┐ 原封不動繞過去
       │                                          │
    rmsnorm                                       │
       │                                          │
       ├──▶ Wq ──▶ q(16) ─┐                       │
       ├──▶ Wk ──▶ k(16) ─┼── 切片成 4 段，各 4 維  │
       └──▶ Wv ──▶ v(16) ─┘   （只是切，沒有魔法）   │
                          │                       │
                 k,v ──▶ ╔═══════════╗            │
                         ║ KV cache  ║            │
                         ╚═══════════╝            │
                          │                       │
              ┌─────┬─────┴─────┬─────┐           │
              │head0│head1│head2│head3│  ← 各自一張圖 4a
              └──┬──┴──┬──┴──┬──┴──┬──┘     彼此完全不講話
                 4     4     4     4                │
                 └─────┴──┬──┴─────┘                │
                     接起來 = 16                     │
                          │                         │
                         Wo   ◀── 唯一讓四個 head    │
                          │      交流的地方          │
                          ▼                         │
                        （＋）◀──────────────────────┘
                          │
                          ▼  進入 MLP
```

**講稿要點**

- 上半部（x → Wq/Wk/Wv → 切片）**4a-0 都看過了**——這張圖只看下半：**怎麼接回主幹**
  （拼回 16 → Wo → ⊕ 回匯流排）。
- 「切成 4 個 head」在程式裡就是 `q[hs:hs+head_dim]` 一行切片（`microgpt_en.py:126`）。
  很多教材把 multi-head 畫得很神祕，其實就是把 16 個數字分成 4 組。
- **Wo 是翻譯層。** head 寫在第 8 格的數字，跟主幹第 8 格的意義毫無關係 ——
  那只是切片的記帳方式。Wo 把「四個 head 各說各話」翻譯成主幹聽得懂的語言。
  它也是四個 head 第一次能被聯合判斷的地方。

**配碼（microgpt_en.py・attention 區塊全文）**

```
116          x_residual = x
117          x = rmsnorm(x)
118          q = linear(x, state_dict[f'layer{li}.attn_wq'])
119          k = linear(x, state_dict[f'layer{li}.attn_wk'])
120          v = linear(x, state_dict[f'layer{li}.attn_wv'])
121          keys[li].append(k)                                # 這一格的 k、v 存檔（圖 9 收）
122          values[li].append(v)
123          x_attn = []
124          for h in range(n_head):
125              hs = h * head_dim
126              q_h = q[hs:hs+head_dim]                       # 「切」就這一行
⋯
132              x_attn.extend(head_out)
133          x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])   # Wo：唯一讓四個 head 交流的地方
134          x = [a + b for a, b in zip(x, x_residual)]             # 旁路在這裡 ⊕ 回來
```

---

## 圖 5 — MLP：64 條 if-then 規則

> 方框內為示意值。這張圖要傳達的是「撐胖 → 篩選 → 壓瘦」的形狀，不是具體數字。

```
       主幹 x (16) ───────────────────────────┐
            │                                 │
         rmsnorm                              │
            │                                 │
         fc1 (16 → 64)   ← 64 個「偵測器」      │
            │              每一列問一個問題：   │
            │              「你身上有沒有 X？」  │
            ▼                                 │
      ┌──────────────────────────────┐        │
      │ -1.2  +3.4  -0.8  +0.1  ...  │        │
      └──────────────────────────────┘        │
            │                                 │
          relu            ← 門檻：             │
            │               沒有 → 0（閉嘴）    │
            ▼               有   → 按強度通過   │
      ┌──────────────────────────────┐        │
      │   0   +3.4    0   +0.1  ...  │        │
      └──────────────────────────────┘        │
            │                                 │
         fc2 (64 → 16)   ← 64 條「回應」        │
            │              「偵測到 X，就往主幹  │
            ▼               加上這個方向」      │
          （＋）◀───────────────────────────────┘
            │
            ▼
```

**講稿要點**

- 回到預 3b：**fc1 就是一個 64 列的矩陣**——每一列都是一個偵測器，跟 x 做一次內積問一題。
- **為什麼要先撐胖再壓瘦？** 如果拿掉 relu，兩個矩陣相乘會塌成一個 16×16 矩陣，
  整段完全白做。**這段的存在意義 100% 來自那個 relu**（預 4 欠的完整論證，在這裡還）。
- 把 MLP 看成**查表式的記憶體**：64 條 if-then 規則平行檢查，觸發的加總起來寫回主幹。
  中間要夠寬，是因為要放得下夠多條規則。（預 4 說的「relu 就是 if」——這裡就是那 64 個 if。）
- 以英文名字想像：某個神經元可能專門偵測「剛剛連續兩個子音」，
  一觸發就往主幹加上「接下來該是母音了」的方向。

**配碼（microgpt_en.py・MLP）**

```
50       def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
⋯
87       state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)   # 64 列＝64 個偵測器
88       state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)   # 64 條回應
⋯
136          x_residual = x
137          x = rmsnorm(x)
138          x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
139          x = [xi.relu() for xi in x]
140          x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
141          x = [a + b for a, b in zip(x, x_residual)]
```

---

## 圖 6 — 殘差：整個模型其實是一個總和

```
  ┌─────────────────── 16 線道的匯流排 ───────────────────┐
  │                                                       │
  │  embedding ──┬──────────────┬──────────▶ lm_head      │
  │              ▲              ▲                         │
  │              │              │                         │
  │            attn0          mlp0                        │
  │              ▲              ▲                         │
  │              └──────────────┘                         │
  │           每一段：讀匯流排 → 算修正量 → 加回去          │
  └───────────────────────────────────────────────────────┘

  展開來看，餵給 lm_head 的向量字面上就是：

      x_final = 嵌入(字母 + 位置)
              + attention 的修正
              + MLP 的修正
```

**講稿要點（我認為這是理解 Transformer 為什麼能疊很深的關鍵）**

- 沒有任何一段「取代」主幹，全部都是**往上疊加**。它是一個總和，不是一連串變形。
- **每一段都可以棄權。** 沒話說就輸出接近 0，主幹原封不動流過。所以層數多不會傷害模型。
- **梯度有一條高速公路。** 加法的微分是 1 —— 程式碼裡看得一清二楚：
  `__add__` 的 local grad 就是 `(1, 1)`（`microgpt_en.py:41`）。
  反向傳播經過殘差時原封不動。
- 這裡也順帶回答「那為什麼要多層？」：把 `n_layer` 從 1 改成 2，
  就只是在這條匯流排上**再插兩個加號**。架構一個字都不用改（`microgpt_en.py:114`）。

**全場最好用的一句話**：
> Attention 決定「去哪裡拿資訊」，MLP 決定「拿到之後要想什麼」，
> 殘差確保「兩者都只是往一條共用的主幹上添加，而不是覆蓋」。

**配碼（microgpt_en.py・梯度的高速公路）**

```
39       def __add__(self, other):                       # ★ other：另一個加數（Value 或普通數字）
40           other = other if isinstance(other, Value) else Value(other)
41           return Value(self.data + other.data, (self, other), (1, 1))
             # (1, 1)＝加法對兩邊的微分——反向傳播原封不動通過
⋯
114      for li in range(n_layer):    # 想多層？迴圈多跑一圈＝匯流排上多插兩個加號
⋯
134          x = [a + b for a, b in zip(x, x_residual)]
⋯
141          x = [a + b for a, b in zip(x, x_residual)]
```

---

## 圖 7 — lm_head：向量變回字母

```
    最終的主幹向量 (16 個數字)
              │
              │  跟 lm_head 的每一列做內積
              │  （每一列代表一個 token）
              ▼
    ┌──────────────────────────────┐
    │  27 個分數（logits）           │
    │  一個 token 一個分數            │
    └──────────────────────────────┘

    「誰的方向跟我最像，分數就最高」  ← 純粹是查字典 / 比對相似度
```

**先回答一個很常見的疑問**：前面 Attention 也做了、MLP 也做了，
那 lm_head 到底還在做什麼？

答案是 —— **它不是再想一次，它是「開口」**。

```
    lm_head 的第 i 列 = 字母 i 在向量空間裡的「看板」

    logit_i = lm_head[i] · x = |lm_head[i]| × |x| × cos(夾角)
                                    ▲                  ▲
                            這個字母的「先天票數」    方向像不像
```

**講稿要點**

- 最後一個矩陣：**27 列，每一列問「你是字母 i 嗎？」**——一列就是一塊看板。
- **它是翻譯，不是思考。** MLP 算完之後，主幹上那 16 個數字仍然是模型的**內部語言** ——
  第 8 維不代表「字母 m」，它沒有名字，只有模型自己懂。
  lm_head 是全流程唯一把內部語言翻回**外部符號**的地方。翻譯不需要智慧，只需要一本字典。
- **它在架構上就沒有能力思考。** 就一行 `linear(x, lm_head)`（`microgpt_en.py:143`）：
  純線性、單一位置，沒有 relu、也看不到別的 token。
- **它是 wte 的鏡像。** `wte` 是 id → 向量，`lm_head` 是向量 → 27 個分數，
  兩個矩陣形狀一模一樣（27 × 16）。最後那一步就是把主幹向量拿去跟 27 塊看板一一比對。
  順帶一個細節：**列的長度本身就是一種先驗** ——
  常見的字母可以把自己的向量拉長，不管 x 指哪邊都先拿到基礎分。這就是 bias，藏在長度裡。
- **（最值得講的一點）正因為它笨，前面才被逼得很聰明。**
  訓練的壓力全部要穿過 lm_head 才傳得回去。既然出口只會做一次內積，
  那唯一能讓 loss 下降的方式，就是逼前面的 Attention + MLP
  把主幹向量準備到「**只差一次內積就能讀出答案**」的程度。

> 所以：**「我累積的意義」和「我對下一個字母的預測」在那個 16 維向量上是同一件事。**
> 這不是架構規定的，是被 lm_head 的笨逼出來的。

**一句話的比喻**：
> MLP 是在腦中想清楚；lm_head 是**張嘴的那一瞬間** ——
> 想法再豐富，出口只有 27 個字母，必須挑一個。
> 壓縮發生在這裡，思考不發生在這裡。

**配碼（microgpt_en.py・出口）**

```
81   state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
     # lm_head 與 wte 同形（27 × 16）——鏡像
⋯
94   def linear(x, w):
95       return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
⋯
143      logits = linear(x, state_dict['lm_head'])   # 「翻譯」的全部：就一行內積，沒有 relu
144      return logits
```

---

## 圖 8 — 抽樣：唯一的隨機

```
  27 個 logits
       │
   ÷ temperature     0.5 → 拉大差距，更保守（microgpt_en.py:187 的預設）
       │             1.5 → 壓平分佈，更隨機
    softmax
       │
   加權隨機抽一個  ◀── ★ 全流程唯一的隨機性
       │
       ▼
     一個字母
```

```
  【只有 BOS 時，模型認為第一個字母該是什麼】   ← 示意分佈

     a  ████████████████
     m  ████████████
     j  ██████████
     k  ████████
     ⋮
     x  ▏                ← 幾乎沒有名字以 x 開頭

  【看過 e, m, m, a 之後】

     <END>  ██████████████████████████████████████████
                    ← 它很確定名字結束了
```

**講稿要點**：模型從沒被告知「字母有分母音子音」、沒被告知「名字通常幾個字母」。
它排出來的分佈完全來自「猜下一個字母」這一件事的副產品。
**這就是「學習」看得見的樣子。**
（`temperature` 在圖 1 的配碼露過臉，這裡才是主場。）

**配碼（microgpt_en.py・抽樣）**

```
187  temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
⋯
193      for pos_id in range(block_size):
194          logits = gpt(token_id, pos_id, keys, values)
195          probs = softmax([l / temperature for l in logits])   # 溫度只是把分數除一下
196          token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
             # 全流程唯一的隨機性，就這一行
197          if token_id == BOS:
198              break
```

---

## 圖 9 — 自迴歸：把全部串起來

```
  輪次 1        輪次 2        輪次 3        輪次 4        輪次 5
  ────────      ────────      ────────      ────────      ────────
  輸入 BOS      輸入「e」      輸入「m」      輸入「m」      輸入「a」
  pos = 0       pos = 1       pos = 2       pos = 3       pos = 4
     │             │             │             │             │
     ▼             ▼             ▼             ▼             ▼
  ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐
  │ GPT │       │ GPT │       │ GPT │       │ GPT │       │ GPT │
  └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘       └──┬──┘
     │             │             │             │             │
  抽出「e」─┐   抽出「m」─┐   抽出「m」─┐   抽出「a」─┐   抽出 <END>
            │            │            │            │           │
            └────────────┘────────────┘────────────┘           ▼
                       餵回下一輪                             停止

  KV cache 逐輪長大（模型不用重算過去）：

     輪次 1 │ k₀ v₀                                  （1 組）
     輪次 2 │ k₀ v₀ │ k₁ v₁                          （2 組）
     輪次 3 │ k₀ v₀ │ k₁ v₁ │ k₂ v₂                  （3 組）
     輪次 4 │ k₀ v₀ │ k₁ v₁ │ k₂ v₂ │ k₃ v₃          （4 組）
              └──────── 只有最右邊那組是新算的 ────────┘

  最終輸出：emma
```

**講稿要點**

- **一次前向只產生一個字母。** 這是最多人搞錯的地方 ——
  ChatGPT 一個字一個字吐出來，不是打字動畫，是它真的一次只算得出一個 token。
- **KV cache 的意義**：每一輪只有「當下這個字母」的 k、v 是新的，過去的直接沿用
  （`microgpt_en.py:121-122` 的 `append`）。
  沒有 cache 的話每輪都要重算整段，長度平方級的浪費。
- 這也是為什麼 **q 只有一個而 k/v 有一串** —— 回頭呼應 4a-0／4a
  （4a-0 埋的那句：k、v 留給後人查，q 用完即丟）。

**配碼（microgpt_en.py・生成一個名字）**

```
121          keys[li].append(k)     # 每輪只把「新的」k、v 加進 cache（gpt() 內部）
122          values[li].append(v)
⋯
190      keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]   # 每個名字歸零重來
191      token_id = BOS
192      sample = []
193      for pos_id in range(block_size):
194          logits = gpt(token_id, pos_id, keys, values)
195          probs = softmax([l / temperature for l in logits])
196          token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
197          if token_id == BOS:
198              break
199          sample.append(uchars[token_id])
```

---

## 圖 10 — 訓練：唯一的目標是猜下一個字母

```
  訓練資料一筆：「emma」

  BOS    e     m     m     a    BOS
   └──▶  ?
         └──▶  ?
               └──▶  ?
                     └──▶  ?
                           └──▶  ?

   每個位置都問一次：「下一個字母是什麼？」
   猜錯多少 = loss（把正確答案的機率取 -log，microgpt_en.py:167）
                            │
                            ▼
                    loss.backward()
                    每個參數都問：
                   「我往哪個方向動一點，loss 會變小？」
                            │
                            ▼
                       Adam 更新
                      4,192 個數字
                     各挪動一小步
                            │
                            ▼
                     重複 num_steps 次
```

**講稿要點**：訓練從頭到尾只有一個任務 —— **猜下一個字母**。
沒有人教它母音子音、沒有人教它名字長度。這些全部是猜下一個字母時「順便」學會的。

- 注意 `losses` 是**整個名字每個位置各算一次**再平均（`microgpt_en.py:163-169`）：
  一筆資料其實提供了 5 個學習訊號，不是 1 個。
- Adam 那一段（`microgpt_en.py:176-182`）沒有任何黑魔法，
  就是「梯度的移動平均 ÷ 梯度平方的移動平均開根號」。

**配碼（microgpt_en.py・訓練一步的全部）**

```
156      doc = docs[step % len(docs)]                    # ★ step／num_steps：第幾步／共 1,000 步
157      tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
158      n = min(block_size, len(tokens) - 1)            # ★ n：這筆資料有幾個「猜下一個」的位置
⋯
163      for pos_id in range(n):
164          token_id, target_id = tokens[pos_id], tokens[pos_id + 1]   # ★ target_id：正確答案＝下一個字母
165          logits = gpt(token_id, pos_id, keys, values)
166          probs = softmax(logits)
167          loss_t = -probs[target_id].log()            # ★ loss_t／loss：猜錯多少（正解機率取 −log）
168          losses.append(loss_t)
169      loss = (1 / n) * sum(losses) # final average loss …
⋯
172      loss.backward()                                 # 每個參數都問：往哪動一點，loss 會變小？
⋯
175      lr_t = learning_rate * (1 - step / num_steps) # …   # ★ lr_t：越練越小步
176      for i, p in enumerate(params):                  # ★ p：4,192 個數字，一個一個來
177          m[i] = beta1 * m[i] + (1 - beta1) * p.grad
             # ★ m／v：Adam 的兩本流水帳（這個 v 跟 attention 的 v 只是撞名，無關！）
             # ★ p.grad：梯度＝「往哪動 loss 會變小」的答案
178          v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
⋯
181          p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
182          p.grad = 0
```

**★ 先講變數**：`step`／`num_steps`＝訓練進度；`n`＝這筆有幾個學習訊號；
`target_id`＝正確答案；`loss_t`／`loss`＝猜錯的代價；`lr_t`＝步伐（線性遞減）；
`p`＝逐一走訪的每個參數、`p.grad`＝它的梯度（圖 0 的 Value 帳本在此收帳）；
`m`／`v`＝Adam 的兩本流水帳——**提醒：這個 `v` 跟 attention 的 `v` 純粹撞名**，
作者為了省字母重用了，講的時候先點破，聽眾就不會回頭亂連結。

---

## 建議的演講順序

| # | 節 | 訊息 |
|---|---|---|
| 1 | 圖 0 | 四千個數字、一層，全部可以手算 |
| 2 | 預 1 | 全場只有一個主角：16 維向量 x，被加料三次 |
| 3 | 預 2 | ① 相加：向量＋向量＝把兩份理解疊起來 |
| 4 | 預 3a | ② 相乘（一）：兩個向量的內積——逐項相乘再加總 |
| 5 | 預 3b | ② 相乘（二）：矩陣就是一疊向量，Python 裡是巢狀 list |
| 6 | 預 4 | ③ 非線性＝神經網路裡的 if（relu；另附 rmsnorm＝音控） |
| 7 | 圖 1 | 全景，先給地圖（工廠外觀 vs 預 1 的內在履歷） |
| 8 | 圖 2 | Tokenizer 沒有魔法 |
| 9 | 圖 3 | 符號 → 意義，位置是學出來的（相加上場） |
| 10 | 圖 3b | context length 上限的真面目 |
| 11 | 圖 4 | **Attention 搬運、MLP 加工**（先給語感再進細節） |
| 12 | 圖 4a-0 | 先認識 q/k/v：三個矩陣、三種角度（零數字） |
| 13 | 圖 4a | **一個 head**（核心，只看數字） |
| 14 | 圖 4b | 為什麼要四個 head |
| 15 | 圖 4c | 接線：怎麼裝回主幹 |
| 16 | 圖 5 | MLP＝64 列的矩陣、64 條 if-then 規則 |
| 17 | 圖 6 | 殘差＝總和（最深刻的一張） |
| 18 | 圖 7 | lm_head＝最後一個矩陣（27 列），翻譯不思考 |
| 19 | 圖 8 | 唯一的隨機性 |
| 20 | 圖 9 | 自迴歸，收攏全局 |
| 21 | 圖 10 | 這一切是怎麼學來的 |

**時間不夠時怎麼辦（降級模式，不砍節）**：
預 2／預 4 改成只講圖、不逐行講碼（各壓到 60–90 秒）；**預 3a／預 3b 完整保留**——
矩陣相乘是全場數學門檻最高的一步，圖 4a-0、圖 5、圖 7 都靠這兩頁撐著。
**圖 4a-0 不可砍**：它之於圖 4a，就像圖 4 之於 4a–4c——動機沒鋪好，數字就白算。
真的還要砍節，順序照舊：先砍圖 4c（接線細節），再砍圖 3b，再砍圖 10。
圖 4、4a、4b、6、8 是五根柱子，不要動——
尤其圖 4 不能為了省時間跳過，它是 4a-0／4a／4b／4c 的前提。

---

## 補充（講完之後，有興趣再談）

以下都不進主線。它們的前提是聽眾已經理解上面的架構。

1. **真實權重裡看得到什麼**
   把訓練好的 checkpoint 攤開來看 `wpe` 的長度與夾角、看四個 head 在同一時刻的實際
   attention 分佈。工具：`scripts/dump_position.py`、`scripts/dump_attention.py`。
   這裡才適合說「模型自己學會了沒人教它的東西」，因為有數字撐著。

2. **換成中文資料集**（`microgpt.py`）
   一個字一個 token 的 tokenizer 在中文上 vocab 直接跳到 700+，
   而 `block_size` 反而變小（名字只有 2~5 個字）。
   同一份程式碼、不同語言，參數量的分佈完全變了樣。

3. **訓練得更久會發生什麼**
   loss 繼續下降，但模型開始**背答案**（生成結果直接命中訓練集原文）。
   這是 overfitting 活生生的樣子，也是很好的收尾討論。
   細節見 `EXPERIMENTS.md`。
